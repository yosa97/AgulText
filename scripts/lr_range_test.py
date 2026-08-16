"""
lr_range_test.py — LR range test satu-ramp (Smith 2015 / fastai lr_find).

Pendekatan BERBEDA dari winner (lr_search.py — multi-trial grid 3 tahap):
  Winner : banyak trial pendek pada LR berbeda, restore bobot antar trial,
           persempit grid dalam 3 pass (coarse → refine → polish).
  Kita   : SATU ramp kontinyu — LR naik geometris dari lr_lo ke lr_hi selama
           N step, catat loss ter-smooth (EMA), berhenti dini saat divergen.
           LR dipilih dari kurva loss-vs-log(LR): titik loss minimum dibagi
           faktor keamanan, lalu di-blend geometris dengan estimasi config.

  Keuntungan: satu kali jalan (bukan K trial), overhead restore cuma sekali,
  waktu terprediksi. Kelemahan: kurang presisi dari grid search penuh —
  ditutupi dengan blend ke estimasi lookup yang sudah teruji.

Sekaligus mengukur t_per_step (median wall-time per micro-step) yang dipakai
train_instruct.py untuk time-aware epoch planning.

DDP: setiap rank menjalankan ramp pada shard datanya sendiri secara independen
(bobot antar rank boleh menyimpang — semuanya di-restore setelah selesai).
Hanya KEPUTUSAN LR yang disinkronkan via broadcast dari rank-0.

Dipanggil dari train_instruct.py SETELAH trainer dibuat, SEBELUM trainer.train()
(optimizer & scheduler trainer dibuat lazy di train(), jadi mengubah
trainer.args.learning_rate sesudah range test tetap efektif).
"""

import datetime
import gc
import math
import os
import time
from typing import Callable, Optional

import torch

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

# Parameter ramp
_MAX_STEPS        = 40     # maksimum micro-step untuk ramp
_MIN_STEPS        = 12     # di bawah ini hasil tidak bermakna → skip
_SPAN_DECADES     = 2.0    # ramp dari est/10^1 ke est*10^1 (total 2 dekade)
_EMA_BETA         = 0.75   # smoothing loss
_DIVERGE_FACTOR   = 2.5    # loss_smooth > factor × best → berhenti dini
_SAFETY_DIV       = 2.0    # LR dipilih = argmin_loss_lr / safety
_BLEND_WEIGHT     = 0.5    # bobot geometris hasil test vs estimasi config
_MAX_BUDGET_SECS  = 420.0  # hard cap waktu untuk seluruh ramp
_BUDGET_FRACTION  = 0.06   # atau 6% dari sisa waktu, mana yang lebih kecil
_MIN_REMAIN_SECS  = 1200.0 # skip kalau sisa waktu < 20 menit


def _secs_until(end_time: str) -> float:
    try:
        end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        return max(0.0, (end - datetime.datetime.now()).total_seconds())
    except Exception:
        return 0.0


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


@torch.no_grad()
def _snapshot_trainable(model) -> dict:
    return {
        n: p.data.detach().cpu().clone()
        for n, p in _unwrap(model).named_parameters()
        if p.requires_grad
    }


@torch.no_grad()
def _restore_trainable(model, snap: dict) -> None:
    for n, p in _unwrap(model).named_parameters():
        if p.requires_grad and n in snap:
            p.data.copy_(snap[n].to(p.device, dtype=p.dtype))


def _make_probe_optimizer(params, lr: float):
    """Optimizer probe — samakan keluarga dengan optimizer training utama
    (paged_adamw_8bit) agar LR yang ditemukan langsung transfer. Fallback ke
    torch AdamW bila bitsandbytes tidak tersedia."""
    try:
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.0)
    except Exception:
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, foreach=False)


def run_lr_range_test(
    trainer,
    *,
    base_lr: float,
    end_time: str,
    log: Optional[Callable] = None,
) -> dict:
    """Jalankan range test. Return dict:
        {"lr": float|None, "t_per_step": float|None, "steps": int, "reason": str}
    lr=None berarti test di-skip / gagal → caller pakai base_lr apa adanya.
    Bobot model SELALU dikembalikan ke kondisi awal sebelum return.
    """
    if log is None:
        log = lambda m: print(m, flush=True)

    result = {"lr": None, "t_per_step": None, "steps": 0, "reason": ""}

    remain = _secs_until(end_time)
    if remain < _MIN_REMAIN_SECS:
        result["reason"] = f"sisa waktu {remain:.0f}s terlalu pendek"
        log(f"[lr_range] skip: {result['reason']}")
        return result

    budget = min(_MAX_BUDGET_SECS, remain * _BUDGET_FRACTION)

    model = trainer.model
    device = next(_unwrap(model).parameters()).device

    # ── Samakan profil memori dengan training ────────────────────────────────
    # Probe berjalan SEBELUM trainer.train(); HF Trainer baru mengaktifkan
    # gradient checkpointing di dalam train(). Tanpa ini, aktivasi full-graph
    # untuk batch penuh OOM di model >~1B (terbukti: 1.7B × bs100 × 512 token
    # = 79GB). enable_input_require_grads sudah dipanggil saat load model.
    if getattr(trainer.args, "gradient_checkpointing", False):
        try:
            _um = _unwrap(model)
            if hasattr(_um, "gradient_checkpointing_enable") and not getattr(
                _um, "is_gradient_checkpointing", False
            ):
                _um.gradient_checkpointing_enable()
                log("[lr_range] gradient checkpointing diaktifkan untuk probe")
        except Exception as _gc_err:
            log(f"[lr_range] gagal aktifkan grad checkpointing ({_gc_err}), lanjut")

    try:
        loader = trainer.get_train_dataloader()
    except Exception as e:
        result["reason"] = f"dataloader gagal: {e}"
        log(f"[lr_range] skip: {result['reason']}")
        return result

    n_avail = len(loader)
    n_steps = min(_MAX_STEPS, n_avail)
    if n_steps < _MIN_STEPS:
        result["reason"] = f"hanya {n_avail} batch tersedia (< {_MIN_STEPS})"
        log(f"[lr_range] skip: {result['reason']}")
        return result

    # Ramp geometris: lr_lo → lr_hi melewati base_lr di tengah
    lr_lo = base_lr / (10 ** (_SPAN_DECADES / 2))
    lr_hi = base_lr * (10 ** (_SPAN_DECADES / 2))
    mult = (lr_hi / lr_lo) ** (1.0 / max(1, n_steps - 1))

    log(
        f"[lr_range] mulai: {n_steps} step, lr {lr_lo:.2e} → {lr_hi:.2e} "
        f"(×{mult:.3f}/step), budget {budget:.0f}s"
    )

    # Snapshot bobot awal (CPU) — dikembalikan setelah ramp
    try:
        snap = _snapshot_trainable(model)
    except Exception as e:
        result["reason"] = f"snapshot gagal: {e}"
        log(f"[lr_range] skip: {result['reason']}")
        return result

    trainable = [p for p in _unwrap(model).parameters() if p.requires_grad]
    opt = _make_probe_optimizer(trainable, lr_lo)

    was_training = model.training
    model.train()

    ema_loss: Optional[float] = None
    best_ema = float("inf")
    best_lr: Optional[float] = None
    step_times: list = []
    lr_cur = lr_lo
    steps_done = 0
    t_start = time.monotonic()

    try:
        for step, batch in enumerate(loader):
            if step >= n_steps:
                break
            if time.monotonic() - t_start > budget:
                log(f"[lr_range] budget waktu habis di step {step}")
                break

            for g in opt.param_groups:
                g["lr"] = lr_cur

            t0 = time.monotonic()
            batch = trainer._prepare_inputs(batch)
            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") else out[0]
            if not torch.isfinite(loss):
                log(f"[lr_range] loss non-finite di lr={lr_cur:.2e}, berhenti")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step_times.append(time.monotonic() - t0)

            raw = float(loss.detach().item())
            ema_loss = raw if ema_loss is None else (
                _EMA_BETA * ema_loss + (1 - _EMA_BETA) * raw
            )
            # Koreksi bias EMA agar step awal tidak underweight
            ema_corr = ema_loss / (1 - _EMA_BETA ** (step + 1))

            if ema_corr < best_ema:
                best_ema = ema_corr
                best_lr = lr_cur

            # Divergensi: smooth loss jauh di atas best → LR sudah kelewat batas
            if steps_done >= _MIN_STEPS and ema_corr > best_ema * _DIVERGE_FACTOR:
                log(
                    f"[lr_range] divergen di lr={lr_cur:.2e} "
                    f"(ema {ema_corr:.3f} > {_DIVERGE_FACTOR}×{best_ema:.3f})"
                )
                break

            steps_done += 1
            lr_cur *= mult
    except Exception as e:
        import traceback as _tb
        log(
            f"[lr_range] error saat ramp ({type(e).__name__}: {e!r}) — "
            f"bobot di-restore, pakai base_lr"
        )
        log("[lr_range] traceback:\n" + _tb.format_exc())
    finally:
        # SELALU restore bobot awal, apapun yang terjadi
        opt.zero_grad(set_to_none=True)
        del opt
        _restore_trainable(model, snap)
        del snap
        if not was_training:
            model.eval()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["steps"] = steps_done

    # t_per_step: median — buang step pertama (warmup CUDA/compile)
    if len(step_times) >= 4:
        st = sorted(step_times[1:])
        result["t_per_step"] = st[len(st) // 2]

    if steps_done < _MIN_STEPS or best_lr is None:
        result["reason"] = f"hanya {steps_done} step selesai — hasil tidak dipakai"
        log(f"[lr_range] {result['reason']}")
        best_lr = None
    else:
        # LR di titik loss minimum cenderung terlalu agresif → bagi safety factor,
        # lalu blend geometris dengan estimasi config (log-space midpoint berbobot)
        picked = best_lr / _SAFETY_DIV
        blended = math.exp(
            _BLEND_WEIGHT * math.log(picked)
            + (1 - _BLEND_WEIGHT) * math.log(base_lr)
        )
        result["lr"] = blended
        log(
            f"[lr_range] selesai: argmin_lr={best_lr:.2e} → picked={picked:.2e} "
            f"→ blend dengan est {base_lr:.2e} = {blended:.2e} "
            f"({steps_done} step, t_per_step={result['t_per_step']})"
        )

    # DDP: sinkronkan keputusan — semua rank pakai nilai rank-0
    if torch.distributed.is_initialized():
        vals = torch.tensor(
            [
                result["lr"] if result["lr"] is not None else -1.0,
                result["t_per_step"] if result["t_per_step"] is not None else -1.0,
            ],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.broadcast(vals, src=0)
        result["lr"] = float(vals[0].item()) if vals[0].item() > 0 else None
        result["t_per_step"] = float(vals[1].item()) if vals[1].item() > 0 else None
        torch.distributed.barrier()

    return result
