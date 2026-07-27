"""
final_dev_train.py — Dev pass akhir dengan keamanan interpolasi bobot.

Mengapa dev pass: validator menilai model pada TEST SET yang sepenuhnya
terpisah. Dev set hanya dipakai untuk memilih checkpoint terbaik — setelah
itu "menganggur" → bisa dipakai sebagai nudge training terakhir.

PELAJARAN dari tournament sebelumnya (versi lama BUGGED):
  1. Evaluation gate mengevaluasi di dev set yang BARU SAJA dilatih →
     loss turun buatan → gate selalu lolos → model overfit tersimpan.
  2. Update SGD per micro-batch tanpa accumulation → ratusan update →
     efektif LR jauh lebih panas dari yang dikalibrasi.

DESAIN BARU — keamanan STRUKTURAL, bukan evaluasi:
  a. Gradient accumulation mengikuti training utama → effective batch sama,
     LR terkalibrasi benar.
  b. Cap jumlah optimizer update (maks 25) → perturbasi terbatas apapun
     ukuran dev set.
  c. INTERPOLASI BOBOT (gaya WiSE-FT, Wortsman et al. 2022):
       final = (1-α)·pre_dev + α·post_dev, α = 0.3
     Model yang disimpan tidak pernah lebih jauh dari 30% langkah menuju
     hasil dev pass → worst case dibatasi secara matematis. TIDAK ADA eval
     gate sama sekali (sumber bug lama).

Perbedaan dari winner (dev_pass.py):
  - Winner: simpan hasil dev pass langsung tanpa blend (safety = LR rendah saja)
  - Winner: pakai mesin accumulate/sync internal trainer (_set_sync_gradients);
    kita pakai loop akumulasi eksplisit sederhana — grad DDP sync tiap
    micro-batch (sedikit lebih lambat, nol private API)
  - Winner: AdamW; kita SGD+Nesterov (tanpa state adaptif — nudge lebih flat)

Dipanggil dari train_instruct.py setelah trainer.train() dan submission
terbaik sudah tersimpan.
"""

import datetime
import gc
import os
import shutil
from typing import Callable, Optional

import torch
from transformers.trainer_utils import is_main_process

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

_MIN_REMAINING_SECS = 120    # batalkan jika sisa waktu < 2 menit
_DEV_LR_RATE        = 0.10   # 10% dari LR training
_MAX_OPT_UPDATES    = 25     # cap update — perturbasi terbatas
_BLEND_ALPHA        = 0.30   # bobot hasil dev pass dalam interpolasi akhir


def _remaining_secs(end_time: str) -> float:
    try:
        end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        return max(0.0, (end - datetime.datetime.now()).total_seconds())
    except Exception:
        return 0.0


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


_WEIGHT_EXTS  = frozenset({".safetensors", ".bin"})
_WEIGHT_NAMES = frozenset({"model.safetensors.index.json"})


def _is_weight_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext in _WEIGHT_EXTS or filename in _WEIGHT_NAMES


def _save_weights(unwrapped_model, submission_dir: str, log: Callable) -> None:
    """Perbarui bobot di submission_dir in-place dengan backup tempdir.

    File non-bobot (config.json, tokenizer, loss.txt) tidak pernah disentuh.
    """
    import tempfile

    if not submission_dir or not os.path.isdir(submission_dir):
        log(f"[final_dev] submission_dir tidak ada ({submission_dir}), skip simpan")
        return

    parent = os.path.dirname(submission_dir.rstrip("/")) or "."
    backup_dir = tempfile.mkdtemp(prefix="_devtrain_bak_", dir=parent)

    try:
        for fn in os.listdir(submission_dir):
            src = os.path.join(submission_dir, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(backup_dir, fn))

        for fn in list(os.listdir(submission_dir)):
            if _is_weight_file(fn):
                os.remove(os.path.join(submission_dir, fn))

        unwrapped_model.save_pretrained(submission_dir, safe_serialization=True)
        log("[final_dev] bobot submission diperbarui (blended, in-place)")
        shutil.rmtree(backup_dir, ignore_errors=True)

    except Exception as e:
        log(f"[final_dev] gagal simpan ({e}), rollback dari backup")
        try:
            for fn in os.listdir(backup_dir):
                dst = os.path.join(submission_dir, fn)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(backup_dir, fn), dst)
        except Exception as re:
            log(f"[final_dev] rollback juga gagal: {re}")
        shutil.rmtree(backup_dir, ignore_errors=True)


def run_final_dev_train(
    trainer,
    *,
    submission_dir: str,
    end_time: str,
    base_lr: float,
    lr_rate: float = _DEV_LR_RATE,
    max_grad_norm: float = 1.0,
    local_rank: int = 0,
    log: Optional[Callable] = None,
) -> None:
    """Dev pass terbatas + blend, lalu simpan tanpa eval gate.

    Args:
        trainer        : HF Trainer yang sudah selesai .train().
        submission_dir : Path submission (checkpoint terbaik sudah ada).
        end_time       : Batas waktu ('YYYY-MM-DD HH:MM:SS').
        base_lr        : LR training utama.
        lr_rate        : Faktor LR dev pass (default 10%).
        max_grad_norm  : Gradient clipping norm.
    """
    if log is None:
        log = lambda m: print(m, flush=True)

    secs = _remaining_secs(end_time)
    if secs < _MIN_REMAINING_SECS:
        log(f"[final_dev] dilewati: sisa {secs:.0f}s < {_MIN_REMAINING_SECS}s")
        return

    import inspect

    ddp_model  = getattr(trainer, "model_wrapped", None) or trainer.model
    unwrapped  = _unwrap(ddp_model)
    accelerator = getattr(trainer, "accelerator", None)
    grad_accum = max(1, int(trainer.args.gradient_accumulation_steps))
    dev_lr = base_lr * lr_rate

    _ts_sig    = inspect.signature(trainer.training_step)
    _ts_kwargs = {"num_items_in_batch": None} if "num_items_in_batch" in _ts_sig.parameters else {}

    trainable = [p for p in unwrapped.parameters() if p.requires_grad]
    if not trainable:
        log("[final_dev] tidak ada parameter trainable, skip")
        return

    # ── Snapshot pre-dev untuk blending nanti ────────────────────────────────
    with torch.no_grad():
        pre_state = {
            n: p.data.detach().cpu().clone()
            for n, p in unwrapped.named_parameters() if p.requires_grad
        }

    # Bersihkan optimizer lama → bebaskan VRAM
    try:
        trainer.optimizer = None
    except Exception:
        pass
    unwrapped.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dev_opt = torch.optim.SGD(
        trainable, lr=dev_lr, momentum=0.85, nesterov=True, weight_decay=0.0,
    )

    dev_loader = trainer.get_eval_dataloader()

    # Lockstep DDP: semua rank memproses jumlah micro-step yang sama.
    # Cap micro-step = MAX_OPT_UPDATES × grad_accum (perturbasi terbatas).
    n_micro = min(len(dev_loader), _MAX_OPT_UPDATES * grad_accum)
    if torch.distributed.is_initialized():
        _t = torch.tensor([n_micro], device=next(unwrapped.parameters()).device)
        torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.MIN)
        n_micro = int(_t.item())

    if n_micro == 0:
        log("[final_dev] dev loader kosong, skip")
        return

    log(
        f"[final_dev] mulai: lr={dev_lr:.2e} ({lr_rate:.0%}×{base_lr:.2e}), "
        f"{n_micro} micro-step, grad_accum={grad_accum}, "
        f"maks {n_micro // grad_accum} update, blend α={_BLEND_ALPHA}"
    )

    ddp_model.train()
    n_updates = 0
    micro_in_accum = 0
    for step, batch in enumerate(dev_loader):
        if step >= n_micro:
            break
        if _remaining_secs(end_time) < 60:
            log(f"[final_dev] waktu hampir habis, berhenti di micro-step {step}")
            break

        # Akumulasi eksplisit: grad menumpuk selama grad_accum micro-batch,
        # optimizer step hanya di boundary → effective batch = training utama.
        # (DDP tetap sync grad tiap micro-batch — sedikit overhead, nol
        # private API, deterministik antar rank.)
        trainer.training_step(ddp_model, batch, **_ts_kwargs)
        micro_in_accum += 1

        if micro_in_accum >= grad_accum:
            if max_grad_norm and max_grad_norm > 0:
                if accelerator is not None:
                    accelerator.clip_grad_norm_(ddp_model.parameters(), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            dev_opt.step()
            ddp_model.zero_grad(set_to_none=True)
            micro_in_accum = 0
            n_updates += 1
            if n_updates >= _MAX_OPT_UPDATES:
                break

    # Sisa grad parsial (belum mencapai boundary) dibuang — bukan di-step,
    # agar semua update punya effective batch penuh yang sama.
    ddp_model.zero_grad(set_to_none=True)
    log(f"[final_dev] selesai: {n_updates} optimizer update")

    del dev_opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if n_updates == 0:
        del pre_state
        log("[final_dev] tidak ada update, bobot tidak berubah — skip simpan")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    # ── Interpolasi bobot: final = (1-α)·pre + α·post ────────────────────────
    # Keamanan struktural: model tersimpan maksimal α dari jarak ke hasil dev
    # pass. Tidak ada eval gate — eval di data yang baru dilatih menyesatkan.
    with torch.no_grad():
        for n, p in unwrapped.named_parameters():
            if p.requires_grad and n in pre_state:
                pre = pre_state[n].to(p.device, dtype=torch.float32)
                post = p.data.float()
                p.data.copy_(
                    ((1.0 - _BLEND_ALPHA) * pre + _BLEND_ALPHA * post).to(p.dtype)
                )
    del pre_state
    gc.collect()
    log(f"[final_dev] blend selesai (α={_BLEND_ALPHA})")

    if is_main_process(local_rank):
        _save_weights(unwrapped, submission_dir, log)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
