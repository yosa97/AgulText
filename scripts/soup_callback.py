"""
soup_callback.py — Checkpoint pool + uniform weight averaging di akhir training.

Konsep: "Model Soup" (Wortsman et al. 2022) — rata-rata bobot beberapa checkpoint
menghasilkan model yang lebih general daripada checkpoint terbaik tunggal.
Rata-rata bekerja karena checkpoint-checkpoint fine-tuning dari titik awal yang sama
cenderung berada di loss basin yang sama → rata-rata tetap di basin tersebut.

Implementasi kita berbeda dari winner:
- Winner: greedy soup (iteratif, eval setiap kandidat baru)
- Kita: uniform averaging (rata-ratakan semua yang di pool sekaligus, satu eval)
  lebih cepat, lebih sederhana, tidak butuh banyak evaluasi tambahan
- Pool dikelola dengan bisect (dua list paralel: kunci loss + data entry),
  bukan list-of-dicts + sort seperti pada umumnya
- RAM check: _ram_headroom_gb() mengembalikan headroom tersisa langsung (float),
  bukan available RAM mentah — sudah memperhitungkan kebutuhan float32 avg
- Overfitting detection: stop training lebih awal jika eval memburuk N kali
- DDP-safe via broadcast, overfitting detection → early stop

Dipanggil dari train_instruct.py sebagai callback tambahan di samping
CustomEvalSaveCallback.
"""

import bisect
import gc
import os
import shutil
from typing import Optional

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import is_main_process

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

# Overfitting: eval_loss > best * (1 + threshold) selama patience kali berturut
_OVERFIT_THRESHOLD = 0.06    # 6% lebih buruk dari best
_OVERFIT_PATIENCE  = 3       # 3 eval berturut-turut
_POOL_MAX          = 6       # maksimum checkpoint di pool
_MIN_HEADROOM_GB   = 3.0     # buffer minimum (GB) setelah kebutuhan snapshot


def _ram_headroom_gb(snap_gb: float) -> float:
    """Headroom RAM tersisa setelah dikurangi estimasi kebutuhan snapshot + soup avg.

    Soup avg di on_train_end membutuhkan float32 accumulator ≈ 2× snap_gb
    (bfloat16 → float32 upcast). Pendekatan konservatif:
        headroom = free_ram - (3 × snap_gb) - MIN_HEADROOM_GB
    Nilai negatif → tidak aman menambah snapshot.
    Nilai nan → RAM tidak bisa dibaca (fallback ke mode konservatif di caller).
    """
    free_gb: Optional[float] = None
    try:
        import psutil
        # .free = benar-benar kosong; lebih konservatif dari .available
        free_gb = psutil.virtual_memory().free / 1e9
    except Exception:
        pass

    if free_gb is None:
        try:
            with open("/proc/meminfo") as fh:
                for ln in fh:
                    if ln.startswith("MemAvailable:"):
                        # MemAvailable dalam kB → bagi 1024² untuk GB
                        free_gb = int(ln.split()[1]) / (1024 ** 2)
                        break
        except Exception:
            pass

    if free_gb is None:
        return float("nan")   # tidak bisa diukur → caller pakai fallback

    # reserved: 1× snap sekarang + 2× snap untuk float32 avg + buffer statis
    reserved = 3.0 * snap_gb + _MIN_HEADROOM_GB
    return free_gb - reserved


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


class ModelSoupCallback(TrainerCallback):
    """Kumpulkan snapshot bobot trainable terbaik, rata-ratakan di akhir training.

    Pool dikelola dengan dua list paralel yang dijaga terurut ascending
    menggunakan bisect: `_pool_keys` (loss float) dan `_pool_data` (dict step+state).
    Ini berbeda dari pendekatan list-of-dicts + sort karena:
    - Insertion O(n) tapi tidak butuh full-sort setiap kali
    - Worst item selalu di indeks [-1] → eviction O(1) pop dari belakang
    - Iterasi untuk rata-rata langsung tanpa lambda key

    Hanya rank-0 yang menyimpan snapshot (hemat RAM). Setelah training selesai,
    rata-rata dihitung di rank-0, broadcast ke semua rank, lalu model dieval.
    Jika rata-rata lebih baik dari best single → submission diupdate.

    Overfitting detection: jika eval_loss memburuk _OVERFIT_THRESHOLD selama
    _OVERFIT_PATIENCE eval berturut, training dihentikan lebih awal.
    """

    def __init__(
        self,
        submission_dir: str,
        pool_max: int = _POOL_MAX,
        overfit_threshold: float = _OVERFIT_THRESHOLD,
        overfit_patience: int = _OVERFIT_PATIENCE,
    ):
        self.submission_dir = submission_dir
        self.pool_max = pool_max
        self.overfit_threshold = overfit_threshold
        self.overfit_patience = overfit_patience

        self.best_loss: float = float("inf")
        self.best_step: int = -1

        # Pool: dua list paralel dijaga ascending by loss via bisect.insort
        # _pool_keys[i] = loss float (key untuk bisect)
        # _pool_data[i] = {"step": int, "state": dict[str, Tensor]}
        self._pool_keys: list[float] = []
        self._pool_data: list[dict]  = []

        self._snap_gb: Optional[float] = None

        self.overfit_counter: int = 0
        self._evaluating: bool = False
        self.trainer = None

    # ── Snapshot helpers ──────────────────────────────────────────────────────

    def _snapshot_gb(self, model) -> float:
        if self._snap_gb is None:
            total = sum(
                p.numel() * p.element_size()
                for p in _unwrap(model).parameters() if p.requires_grad
            )
            self._snap_gb = total / 1e9
        return self._snap_gb

    def _can_add_snapshot(self, model) -> bool:
        snap    = self._snapshot_gb(model)
        headroom = _ram_headroom_gb(snap)
        if headroom != headroom:   # NaN → RAM tidak terbaca, konservatif
            return len(self._pool_keys) < 2
        return headroom >= 0.0

    @torch.no_grad()
    def _take_snapshot(self, model) -> dict[str, torch.Tensor]:
        return {
            n: p.data.cpu().clone()
            for n, p in _unwrap(model).named_parameters()
            if p.requires_grad
        }

    def _pool_size(self) -> int:
        return len(self._pool_keys)

    def _update_pool(self, model, loss: float, step: int) -> None:
        """Masukkan snapshot ke pool menggunakan bisect untuk insert terurut.

        Strategi:
        - Pool belum penuh dan RAM cukup → insert di posisi bisect
        - Pool penuh dan loss baru lebih baik dari terburuk (indeks [-1]) →
          evict terburuk (pop belakang), baru insert
        - Selain itu → tidak masuk
        """
        if loss != loss or loss == float("inf"):
            return

        snap_gb  = self._snapshot_gb(model)
        headroom = _ram_headroom_gb(snap_gb)
        head_str = f"{headroom:.1f}GB" if headroom == headroom else "?"

        pool_full = self._pool_size() >= self.pool_max

        if not pool_full and self._can_add_snapshot(model):
            snap = self._take_snapshot(model)
            pos  = bisect.bisect_left(self._pool_keys, loss)
            self._pool_keys.insert(pos, loss)
            self._pool_data.insert(pos, {"step": step, "state": snap})
            print(
                f"[soup] pool +1 step={step} loss={loss:.4f} "
                f"({self._pool_size()}/{self.pool_max}, "
                f"snap~{snap_gb:.2f}GB, headroom={head_str})",
                flush=True,
            )
        elif pool_full and loss < self._pool_keys[-1]:
            # Evict worst (index -1) sebelum alokasi snapshot baru
            evicted = self._pool_data.pop()
            self._pool_keys.pop()
            evicted["state"] = None
            del evicted
            gc.collect()

            if self._can_add_snapshot(model):
                snap = self._take_snapshot(model)
                pos  = bisect.bisect_left(self._pool_keys, loss)
                self._pool_keys.insert(pos, loss)
                self._pool_data.insert(pos, {"step": step, "state": snap})
                print(
                    f"[soup] pool swap step={step} loss={loss:.4f} "
                    f"(snap~{snap_gb:.2f}GB, headroom={head_str})",
                    flush=True,
                )

    # ── DDP sync ─────────────────────────────────────────────────────────────

    def _sync_params(self, model):
        """Broadcast semua trainable params dari rank-0 ke semua rank."""
        if torch.distributed.is_initialized():
            for p in _unwrap(model).parameters():
                if p.requires_grad:
                    torch.distributed.broadcast(p.data, src=0)

    def _sync_scalar(self, model, val: float) -> float:
        if not torch.distributed.is_initialized():
            return val
        dev = next(model.parameters()).device
        t = torch.tensor([val], device=dev)
        torch.distributed.broadcast(t, src=0)
        return float(t.item())

    # ── Averaging ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _apply_uniform_avg(self, model) -> bool:
        """Rata-ratakan semua snapshot di pool, load ke model, sync semua rank."""
        if self._pool_size() < 2:
            return False
        is_main = is_main_process(LOCAL_RANK)
        if is_main:
            n = self._pool_size()
            avg: dict[str, torch.Tensor] = {}
            # Iterasi langsung ke _pool_data (sudah terurut ascending by loss)
            for name in self._pool_data[0]["state"]:
                acc = self._pool_data[0]["state"][name].float().clone()
                for entry in self._pool_data[1:]:
                    if name in entry["state"]:
                        acc.add_(entry["state"][name].float())
                avg[name] = acc.div_(n)

            unwrapped = _unwrap(model)
            for name, p in unwrapped.named_parameters():
                if name in avg and p.requires_grad:
                    p.data.copy_(avg[name].to(p.dtype).to(p.device))

            del avg
            gc.collect()
            print(f"[soup] uniform avg n={n} snapshot → model updated", flush=True)

        self._sync_params(model)
        return True

    # ── Submission update ─────────────────────────────────────────────────────

    def _save_to_submission(self, model, loss: float) -> None:
        """Simpan bobot model ke submission_dir (rank-0 saja).

        Menggunakan pola in-place + tempfile backup (sama dengan final_dev_train):
        1. Backup semua file ke temp dir
        2. Hapus weight files lama dari submission_dir
        3. Tulis bobot baru langsung ke submission_dir
        4. Hapus backup jika sukses; rollback jika gagal
        """
        if not is_main_process(LOCAL_RANK):
            return
        if not self.submission_dir or not os.path.isdir(self.submission_dir):
            return

        import tempfile

        _W_EXTS  = frozenset({".safetensors", ".bin"})
        _W_NAMES = frozenset({"model.safetensors.index.json"})

        def _is_weight(fn: str) -> bool:
            _, ext = os.path.splitext(fn)
            return ext in _W_EXTS or fn in _W_NAMES

        parent     = os.path.dirname(self.submission_dir.rstrip("/")) or "."
        backup_dir = tempfile.mkdtemp(prefix="_soup_bak_", dir=parent)

        try:
            for fn in os.listdir(self.submission_dir):
                src = os.path.join(self.submission_dir, fn)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(backup_dir, fn))

            for fn in list(os.listdir(self.submission_dir)):
                if _is_weight(fn):
                    os.remove(os.path.join(self.submission_dir, fn))

            _unwrap(model).save_pretrained(self.submission_dir, safe_serialization=True)

            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"soup_avg,{loss:.6f}")

            shutil.rmtree(backup_dir, ignore_errors=True)
            print(f"[soup] submission diperbarui (loss={loss:.4f})", flush=True)

        except Exception as e:
            print(f"[soup] gagal update submission: {e}", flush=True)
            try:
                for fn in os.listdir(backup_dir):
                    dst = os.path.join(self.submission_dir, fn)
                    if not os.path.exists(dst):
                        shutil.copy2(os.path.join(backup_dir, fn), dst)
            except Exception:
                pass
            shutil.rmtree(backup_dir, ignore_errors=True)

    # ── TrainerCallback hooks ─────────────────────────────────────────────────

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kw):
        if model is None:
            return
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        snap_gb  = self._snapshot_gb(model)
        headroom = _ram_headroom_gb(snap_gb)
        head_str = f"{headroom:.1f}GB" if headroom == headroom else "tidak bisa diukur"
        print(
            f"[soup] siap: pool_max={self.pool_max} (bisect), "
            f"overfit_threshold={self.overfit_threshold:.0%}, "
            f"patience={self.overfit_patience}, "
            f"n_trainable={n_trainable/1e6:.1f}M "
            f"(~{snap_gb:.2f}GB/snapshot, headroom awal={head_str})",
            flush=True,
        )

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        metrics=None,
        **kw,
    ):
        if self._evaluating or model is None or metrics is None:
            return

        loss = metrics.get("eval_loss")
        if loss is None or loss != loss:
            return

        is_main     = is_main_process(LOCAL_RANK)
        is_new_best = loss < self.best_loss

        if is_new_best:
            self.best_loss = loss
            self.best_step = state.global_step
            self.overfit_counter = 0

        # Sinkronisasi best_loss lintas rank
        self.best_loss = self._sync_scalar(model, self.best_loss)

        if is_main:
            self._update_pool(model, loss, state.global_step)

        # Deteksi overfitting
        if not is_new_best and loss > self.best_loss * (1 + self.overfit_threshold):
            self.overfit_counter += 1
            print(
                f"[soup] overfit signal #{self.overfit_counter}/{self.overfit_patience}: "
                f"loss={loss:.4f} > best={self.best_loss:.4f} "
                f"(+{(loss / self.best_loss - 1) * 100:.1f}%)",
                flush=True,
            )
            if self.overfit_counter >= self.overfit_patience:
                print("[soup] overfit dikonfirmasi, training dihentikan lebih awal", flush=True)
                control.should_training_stop = True
        else:
            self.overfit_counter = 0

    def on_train_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kw,
    ):
        if model is None or self.trainer is None:
            return

        if self._pool_size() < 2:
            print(
                f"[soup] hanya {self._pool_size()} snapshot di pool, skip averaging",
                flush=True,
            )
            return

        print(
            f"[soup] mulai uniform averaging dari {self._pool_size()} snapshot",
            flush=True,
        )

        # Simpan bobot saat ini untuk rollback jika avg lebih buruk
        is_main = is_main_process(LOCAL_RANK)
        current_state = None
        if is_main:
            current_state = {
                n: p.data.cpu().clone()
                for n, p in _unwrap(model).named_parameters() if p.requires_grad
            }

        applied = self._apply_uniform_avg(model)
        if not applied:
            return

        # Eval rata-rata — re-entry guard via self._evaluating
        self._evaluating = True
        try:
            avg_metrics = self.trainer.evaluate()
            avg_loss = avg_metrics.get("eval_loss", float("inf"))
        except Exception as e:
            print(f"[soup] eval gagal setelah averaging: {e}", flush=True)
            avg_loss = float("inf")
        finally:
            self._evaluating = False

        avg_loss = self._sync_scalar(model, avg_loss)

        if avg_loss < self.best_loss - 1e-4:
            print(
                f"[soup] rata-rata LEBIH BAIK: {avg_loss:.4f} < best={self.best_loss:.4f} "
                f"(delta {self.best_loss - avg_loss:.4f})",
                flush=True,
            )
            self.best_loss = avg_loss
            self._save_to_submission(model, avg_loss)
        else:
            print(
                f"[soup] rata-rata tidak lebih baik ({avg_loss:.4f} >= {self.best_loss:.4f}), "
                f"rollback ke checkpoint terbaik",
                flush=True,
            )
            if is_main and current_state is not None:
                for n, p in _unwrap(model).named_parameters():
                    if n in current_state and p.requires_grad:
                        p.data.copy_(current_state[n].to(p.device))
            self._sync_params(model)

        # Cleanup pool → bebaskan RAM
        for entry in self._pool_data:
            entry["state"] = None
        self._pool_data.clear()
        self._pool_keys.clear()
        if is_main and current_state is not None:
            del current_state
        gc.collect()

        print(
            f"[soup] selesai: best_loss={self.best_loss:.4f} @ step={self.best_step}",
            flush=True,
        )
