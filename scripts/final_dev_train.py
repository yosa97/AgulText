"""
final_dev_train.py — Satu epoch terakhir dengan LR minimum pada data dev.

Mengapa aman:
  Validator mengevaluasi model pada TEST SET yang sepenuhnya terpisah dari
  dataset kita. Dev set hanya kita gunakan untuk memilih checkpoint terbaik
  selama training. Setelah checkpoint dipilih, dev set sudah tidak "terpakai"
  lagi → kita bisa pakai untuk satu epoch terakhir dengan LR kecil untuk
  memaksimalkan pemanfaatan data.

Perbedaan dari winner (dev_pass.py):
  - Menggunakan optimizer SGD + Nesterov momentum (winner: AdamW)
    → LR flat tanpa adaptive per-param state, cocok untuk fine-tuning kecil
  - Menghitung effective_lr dari min_lr_rate × current_lr
    (winner menggunakan args.learning_rate × multiplier langsung)
  - Guard waktu: cek sisa waktu sebelum mulai, batalkan jika < 2 menit tersisa
  - Save pattern: backup-in-place + restore jika gagal (winner: staging+rename)

DDP-safe (mengikuti pola winner):
  - Gunakan trainer.model_wrapped (bukan trainer.model) — menghindari
    pembuatan DDP hook kedua yang menyebabkan error saat backward.
  - Gunakan accelerator.gradient_state._set_sync_gradients(do_sync) +
    accelerator.no_sync(model=ddp_model) — pola resmi HF Accelerate untuk
    mengontrol kapan gradient di-sync antar rank.
  - Tutup dengan torch.distributed.barrier() agar semua rank sinkron.

Dipanggil dari train_instruct.py setelah trainer.train() selesai dan
checkpoint terbaik sudah di submission_dir.
"""

import gc
import os
import shutil
import datetime
from typing import Callable, Optional

import torch
from transformers.trainer_utils import is_main_process

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

_MIN_REMAINING_SECS = 120   # Batalkan jika sisa waktu < 2 menit
_DEFAULT_LR_RATE    = 0.05  # 5% dari LR training (sangat konservatif)


def _remaining_secs(end_time: str) -> float:
    """Hitung detik tersisa hingga end_time (format: 'YYYY-MM-DD HH:MM:SS')."""
    try:
        end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        delta = (end - datetime.datetime.now()).total_seconds()
        return max(0.0, delta)
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
    """Perbarui bobot di submission_dir secara in-place dengan backup tempdir.

    Strategi berbeda dari staging-rename: kita salin seluruh submission ke
    temp dir sistem, hapus weight lama dari submission, tulis bobot baru
    langsung ke submission_dir, lalu hapus temp backup jika sukses.
    File non-bobot (config.json, tokenizer, loss.txt) tidak pernah disentuh.
    """
    import tempfile

    if not submission_dir or not os.path.isdir(submission_dir):
        log(f"[final_dev] submission_dir tidak ada ({submission_dir}), skip simpan")
        return

    # Buat temp dir di direktori parent submission agar satu filesystem (rename cepat)
    parent = os.path.dirname(submission_dir.rstrip("/")) or "."
    backup_dir = tempfile.mkdtemp(prefix="_devtrain_bak_", dir=parent)

    try:
        # 1. Salin semua file existing ke backup (bukan copy tree — hindari nested dir)
        for fn in os.listdir(submission_dir):
            src = os.path.join(submission_dir, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(backup_dir, fn))

        # 2. Hapus hanya file bobot lama dari submission_dir
        for fn in list(os.listdir(submission_dir)):
            if _is_weight_file(fn):
                os.remove(os.path.join(submission_dir, fn))

        # 3. Tulis bobot baru langsung ke submission_dir
        unwrapped_model.save_pretrained(submission_dir, safe_serialization=True)

        log("[final_dev] bobot submission diperbarui (in-place, non-bobot dipertahankan)")

        # 4. Sukses — hapus backup
        shutil.rmtree(backup_dir, ignore_errors=True)

    except Exception as e:
        log(f"[final_dev] gagal simpan ({e}), rollback dari backup")
        # Rollback: kembalikan file yang ada di backup tapi tidak di submission
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
    lr_rate: float = _DEFAULT_LR_RATE,
    max_grad_norm: float = 1.0,
    local_rank: int = 0,
    log: Optional[Callable] = None,
) -> None:
    """Satu epoch terakhir menggunakan dev set sebagai data training.

    Args:
        trainer        : HF Trainer yang sudah selesai .train().
        submission_dir : Path ke submission directory (checkpoint terbaik sudah ada).
        end_time       : Batas waktu tournament ('YYYY-MM-DD HH:MM:SS').
        base_lr        : Learning rate saat training utama.
        lr_rate        : Faktor LR untuk dev pass (default 5% dari base_lr).
        max_grad_norm  : Gradient clipping norm.
        local_rank     : Rank proses saat ini.
        log            : Fungsi logging. Default: print.
    """
    if log is None:
        log = lambda m: print(m, flush=True)

    # Guard: cek sisa waktu
    secs = _remaining_secs(end_time)
    if secs < _MIN_REMAINING_SECS:
        log(
            f"[final_dev] dilewati: sisa waktu {secs:.0f}s "
            f"< minimum {_MIN_REMAINING_SECS}s"
        )
        return

    dev_lr = base_lr * lr_rate
    log(
        f"[final_dev] mulai dev-pass: lr={dev_lr:.2e} "
        f"(={lr_rate:.0%} × {base_lr:.2e}), sisa={secs:.0f}s"
    )

    import inspect

    # Gunakan model_wrapped (sudah DDP-wrapped oleh trainer) — bukan trainer.model
    # yang belum wrapped. Ini menghindari pembuatan set DDP reducer hooks kedua
    # saat kita memanggil training_step, yang menyebabkan "DDP communication hook
    # error" saat backward. Mengikuti pola winner yang terbukti aman.
    ddp_model = getattr(trainer, "model_wrapped", None) or trainer.model
    unwrapped  = _unwrap(ddp_model)
    accelerator = getattr(trainer, "accelerator", None)

    # Pre-build kwargs untuk training_step (cek sekali saja, bukan per iterasi)
    _ts_sig    = inspect.signature(trainer.training_step)
    _ts_kwargs = {"num_items_in_batch": None} if "num_items_in_batch" in _ts_sig.parameters else {}

    trainable = [p for p in unwrapped.parameters() if p.requires_grad]
    if not trainable:
        log("[final_dev] tidak ada parameter trainable, skip")
        return

    # Bersihkan optimizer lama untuk bebaskan VRAM
    try:
        trainer.optimizer = None
    except Exception:
        pass
    unwrapped.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optimizer dev-pass: SGD dengan Nesterov momentum.
    # Pilihan berbeda dari AdamW yang dipakai saat training utama:
    # - Tidak ada adaptive per-param LR (Adam's m/v state) → update lebih stabil
    # - Nesterov look-ahead: gradien dihitung di posisi "terlihat ke depan"
    #   → konvergensi lebih halus untuk fine-tuning akhir dengan LR sangat kecil
    # - LR flat (tidak decay per step) sesuai dengan tujuan "nudge" kecil
    dev_opt = torch.optim.SGD(
        trainable,
        lr=dev_lr,
        momentum=0.85,
        nesterov=True,
        weight_decay=0.0,
    )

    # Dev loader dari trainer — pakai eval dataloader (batch kecil, no shuffle)
    dev_loader = trainer.get_eval_dataloader()

    # Lockstep DDP: semua rank proses jumlah step yang sama
    n_steps = len(dev_loader)
    if torch.distributed.is_initialized():
        _t = torch.tensor([n_steps], device=next(unwrapped.parameters()).device)
        torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.MIN)
        n_steps = int(_t.item())

    if n_steps == 0:
        log("[final_dev] dev loader kosong, skip")
        return

    ddp_model.train()
    n_opt_steps = 0
    for step, batch in enumerate(dev_loader):
        if step >= n_steps:
            break

        # Guard waktu dalam loop
        if _remaining_secs(end_time) < 60:
            log(f"[final_dev] waktu hampir habis, berhenti di step {step}")
            break

        # Dev pass tidak memakai gradient accumulation: setiap micro-batch langsung
        # update parameter. Alasannya:
        #   1. Dev set kecil (satu epoch) — efisiensi accumulation tidak signifikan
        #   2. Menghindari pemakaian _set_sync_gradients (private API) yang berpotensi
        #      menyebabkan masalah versi bila HF Accelerate di-update
        #   3. DDP selalu sync tiap step — lebih aman, tidak ada edge case no_sync
        # Gradient SELALU di-sync → tidak perlu no_sync / accumulation logic.
        trainer.training_step(ddp_model, batch, **_ts_kwargs)

        if max_grad_norm and max_grad_norm > 0:
            if accelerator is not None:
                accelerator.clip_grad_norm_(ddp_model.parameters(), max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        dev_opt.step()
        ddp_model.zero_grad(set_to_none=True)
        n_opt_steps += 1

    log(
        f"[final_dev] selesai: {n_steps} micro-step, {n_opt_steps} opt-step, "
        f"lr={dev_lr:.2e}"
    )

    # Bersihkan optimizer
    del dev_opt
    unwrapped.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Simpan hasil ke submission (rank-0 saja)
    if is_main_process(local_rank):
        _save_weights(unwrapped, submission_dir, log)

    # Barrier agar semua rank selesai sebelum lanjut
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
