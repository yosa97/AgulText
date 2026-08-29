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
# Threshold 15% dan patience 5: cukup longgar agar training kecil tidak terhenti
# terlalu cepat, tapi masih sensitif terhadap overfitting yang nyata.
_OVERFIT_THRESHOLD = 0.08    # 8% lebih buruk dari best — lebih sensitif dari 15%
_OVERFIT_PATIENCE  = 4       # 4 eval berturut-turut — antara 3 (winner) dan 5 (lama)
_POOL_MAX          = 6       # maksimum checkpoint di pool
_MIN_HEADROOM_GB   = 3.0     # buffer minimum (GB) setelah kebutuhan snapshot

# Rollback saat overfit terkonfirmasi (pendekatan kita, berbeda dari winner):
# - Konfirmasi = counter >= patience DAN slope regresi linear eval-loss POSITIF
#   (winner: murni hitung berturut-turut di atas threshold)
# - Restore dari pool soup yang SUDAH ada (winner: simpan best snapshot terpisah)
# - Satu kali rollback saja (winner: 2), setelah itu konfirmasi kedua → stop
_ROLLBACK_LR_FACTOR = 0.6    # LR dipotong ke 60% (winner: 50%)
_NEFTUNE_BUMP       = 4.0    # alpha ditambah +4 (winner: tangga 5→10→15)
_SLOPE_WINDOW       = 5      # jumlah eval terakhir untuk regresi slope


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

        # Riwayat eval loss untuk uji slope + status rollback
        self._loss_history: list[float] = []
        self._rollback_done: bool = False

        # ── EMA bobot sepanjang lintasan training ────────────────────────────
        # Kandidat submission ketiga (selain best-single dan pool-avg):
        # rata-rata bergerak eksponensial dari bobot, di-update tiap N step.
        # Berbeda mekanisme dari pool-avg (yang merata-rata K titik eval
        # terbaik) — EMA menghaluskan SEPANJANG lintasan, sering unggul 1-3%
        # pada akhir jadwal decay. Matikan via EMA=0.
        self._ema_enabled = os.environ.get("EMA", "1") != "0"
        self._ema_every = max(1, int(os.environ.get("EMA_EVERY") or 10))
        self._ema_decay = float(os.environ.get("EMA_DECAY") or 0.99)
        self._ema_state: Optional[dict] = None

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
        """Rata-ratakan semua snapshot di pool, load ke model, sync semua rank.

        DDP-safe: broadcast jumlah snapshot dari rank-0 ke semua rank sebelum
        memutuskan apakah lanjut atau skip.  Non-main rank punya pool kosong
        (local _pool_size()=0), tapi keputusan harus berdasarkan nilai rank-0
        agar semua rank agree dan tidak ada split-return yang menyebabkan
        deadlock pada _sync_params / collective berikutnya.
        """
        is_main = is_main_process(LOCAL_RANK)

        # Broadcast pool size dari rank-0 agar semua rank agree apakah lanjut.
        # Non-main punya n=0 secara lokal, kita ganti dengan nilai rank-0.
        n = self._pool_size()    # rank-0: jumlah nyata; non-main: 0
        if torch.distributed.is_initialized():
            dev = next(_unwrap(model).parameters()).device
            t = torch.tensor([n], dtype=torch.long, device=dev)
            torch.distributed.broadcast(t, src=0)
            n = int(t.item())    # semua rank pakai nilai rank-0

        if n < 2:
            return False         # semua rank return bersama — tidak ada deadlock

        if is_main:
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

        self._sync_params(model)   # broadcast rank-0 weights ke semua rank
        return True

    # ── Overfit rollback ─────────────────────────────────────────────────────

    def _recent_slope(self) -> float:
        """Slope regresi linear dari _SLOPE_WINDOW eval loss terakhir.

        Least-squares sederhana pada (index, loss). Slope > 0 = loss sedang
        naik secara tren, bukan sekadar satu-dua eval yang jelek.
        """
        ys = self._loss_history[-_SLOPE_WINDOW:]
        n = len(ys)
        if n < 3:
            return 0.0
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom

    def _cut_lr(self, factor: float) -> None:
        """Potong LR optimizer + base_lrs scheduler dengan faktor.

        Scheduler cosine menghitung LR tiap step dari base_lrs, jadi hanya
        memotong param_groups tidak cukup — base_lrs juga harus dipotong.
        Dijalankan di SEMUA rank (tiap rank punya optimizer sendiri).
        """
        tr = self.trainer
        if tr is None:
            return
        try:
            if tr.optimizer is not None:
                for g in tr.optimizer.param_groups:
                    g["lr"] = g.get("lr", 0.0) * factor
            sched = getattr(tr, "lr_scheduler", None)
            if sched is not None and hasattr(sched, "base_lrs"):
                sched.base_lrs = [b * factor for b in sched.base_lrs]
        except Exception as e:
            print(f"[soup] gagal potong LR: {e}", flush=True)

    def _bump_neftune(self, model) -> None:
        """Naikkan NEFTune alpha (best-effort — skip jika NEFTune tidak aktif).

        HF Trainer menyimpan alpha di trainer.neftune_noise_alpha DAN di atribut
        module embedding yang di-hook; keduanya di-update.
        """
        tr = self.trainer
        try:
            cur = getattr(tr, "neftune_noise_alpha", None)
            if cur is None:
                return
            new = float(cur) + _NEFTUNE_BUMP
            tr.neftune_noise_alpha = new
            emb = _unwrap(model).get_input_embeddings()
            if hasattr(emb, "neftune_noise_alpha"):
                emb.neftune_noise_alpha = new
            print(f"[soup] NEFTune alpha {cur} → {new}", flush=True)
        except Exception as e:
            print(f"[soup] gagal bump NEFTune: {e}", flush=True)

    def _rollback_to_pool_best(self, model) -> None:
        """Restore bobot dari entry terbaik pool (indeks 0, terurut ascending).

        Rank-0 menyalin dari pool → broadcast ke semua rank via _sync_params.
        Semua rank HARUS memanggil fungsi ini bersama (collective).
        """
        if is_main_process(LOCAL_RANK) and self._pool_size() > 0:
            best_entry = self._pool_data[0]
            with torch.no_grad():
                for n, p in _unwrap(model).named_parameters():
                    if p.requires_grad and n in best_entry["state"]:
                        p.data.copy_(best_entry["state"][n].to(p.device, dtype=p.dtype))
            print(
                f"[soup] rollback ke pool best (step={best_entry['step']}, "
                f"loss={self._pool_keys[0]:.4f})",
                flush=True,
            )
        self._sync_params(model)

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

    # ── EMA machinery ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _ema_update(self, model) -> None:
        """Update EMA CPU-float32 dari bobot trainable (rank-0 saja)."""
        um = _unwrap(model)
        if self._ema_state is None:
            self._ema_state = {
                n: p.data.detach().float().cpu().clone()
                for n, p in um.named_parameters() if p.requires_grad
            }
            return
        d = self._ema_decay
        for n, p in um.named_parameters():
            if p.requires_grad and n in self._ema_state:
                self._ema_state[n].mul_(d).add_(
                    p.data.detach().float().cpu(), alpha=1.0 - d
                )

    def _try_ema_candidate(self, model) -> None:
        """Evaluasi bobot EMA; simpan ke submission jika mengalahkan best.

        DDP-safe: keputusan ada/tidaknya EMA di-broadcast dari rank-0,
        semua rank ikut evaluate() dan sync bobot.
        """
        is_main = is_main_process(LOCAL_RANK)
        _has = 1 if (self._ema_enabled and self._ema_state is not None) else 0
        if torch.distributed.is_initialized():
            dev = next(_unwrap(model).parameters()).device
            t = torch.tensor([_has], dtype=torch.long, device=dev)
            torch.distributed.broadcast(t, src=0)
            _has = int(t.item())
        if not _has or self.trainer is None:
            return

        prev_state = None
        if is_main:
            um = _unwrap(model)
            prev_state = {
                n: p.data.cpu().clone()
                for n, p in um.named_parameters() if p.requires_grad
            }
            with torch.no_grad():
                for n, p in um.named_parameters():
                    if p.requires_grad and n in self._ema_state:
                        p.data.copy_(self._ema_state[n].to(p.dtype).to(p.device))
        self._sync_params(model)

        self._evaluating = True
        try:
            _m = self.trainer.evaluate()
            ema_loss = _m.get("eval_loss", float("inf"))
        except Exception as _e:
            if is_main:
                print(f"[soup] eval EMA gagal: {_e}", flush=True)
            ema_loss = float("inf")
        finally:
            self._evaluating = False
        ema_loss = self._sync_scalar(model, ema_loss)

        if ema_loss < self.best_loss - 1e-4:
            if is_main:
                print(
                    f"[soup] EMA LEBIH BAIK: {ema_loss:.4f} < {self.best_loss:.4f} "
                    f"— menyimpan ke submission",
                    flush=True,
                )
            self.best_loss = ema_loss
            self._save_to_submission(model, ema_loss)
        else:
            if is_main:
                print(
                    f"[soup] EMA tidak lebih baik ({ema_loss:.4f} >= {self.best_loss:.4f})",
                    flush=True,
                )
                if prev_state is not None:
                    with torch.no_grad():
                        for n, p in _unwrap(model).named_parameters():
                            if p.requires_grad and n in prev_state:
                                p.data.copy_(prev_state[n].to(p.device))
            self._sync_params(model)

        if is_main:
            self._ema_state = None
        del prev_state
        gc.collect()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kw):
        if not self._ema_enabled or model is None:
            return
        if state.global_step % self._ema_every != 0:
            return
        if is_main_process(LOCAL_RANK):
            self._ema_update(model)

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

        self._loss_history.append(loss)

        # Deteksi overfitting: threshold + counter (seperti sebelumnya) DAN
        # slope regresi positif (loss benar-benar tren naik, bukan noise).
        if not is_new_best and loss > self.best_loss * (1 + self.overfit_threshold):
            self.overfit_counter += 1
            slope = self._recent_slope()
            print(
                f"[soup] overfit signal #{self.overfit_counter}/{self.overfit_patience}: "
                f"loss={loss:.4f} > best={self.best_loss:.4f} "
                f"(+{(loss / self.best_loss - 1) * 100:.1f}%, slope={slope:+.4f})",
                flush=True,
            )
            if self.overfit_counter >= self.overfit_patience and slope > 0:
                if not self._rollback_done:
                    # Kesempatan kedua: restore best dari pool, LR dipotong,
                    # NEFTune dinaikkan → lanjut training dengan regularisasi
                    # lebih kuat, alih-alih membuang sisa waktu.
                    print("[soup] overfit dikonfirmasi → ROLLBACK + potong LR", flush=True)
                    self._rollback_to_pool_best(model)
                    self._cut_lr(_ROLLBACK_LR_FACTOR)
                    self._bump_neftune(model)
                    self._rollback_done = True
                    self.overfit_counter = 0
                    self._loss_history.clear()
                else:
                    print(
                        "[soup] overfit kedua setelah rollback, "
                        "training dihentikan lebih awal",
                        flush=True,
                    )
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

        is_main = is_main_process(LOCAL_RANK)

        # ── DDP deadlock fix ──────────────────────────────────────────────────
        # Hanya rank-0 yang menyimpan snapshot (_update_pool dibatasi is_main).
        # Non-main rank punya pool kosong. Jika kita biarkan mereka return lebih
        # awal, mereka tidak akan ikut dalam broadcast/evaluate yang dipanggil
        # rank-0 → deadlock.
        #
        # Solusi: broadcast n_snap dari rank-0 ke semua rank SEBELUM decision.
        # Semua rank pakai nilai rank-0 untuk memutuskan apakah lanjut atau tidak.
        n_snap = self._pool_size()   # rank-0: jumlah snapshot; non-main: 0
        if torch.distributed.is_initialized():
            dev = next(_unwrap(model).parameters()).device
            t = torch.tensor([n_snap], dtype=torch.long, device=dev)
            torch.distributed.broadcast(t, src=0)
            n_snap = int(t.item())   # semua rank tahu jumlah snapshot rank-0

        if n_snap < 2:
            if is_main:
                print(
                    f"[soup] hanya {self._pool_size()} snapshot di pool, skip averaging",
                    flush=True,
                )
            # EMA tetap dievaluasi meski pool kosong
            self._try_ema_candidate(model)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        if is_main:
            print(
                f"[soup] mulai uniform averaging dari {self._pool_size()} snapshot",
                flush=True,
            )

        # Simpan bobot saat ini untuk rollback jika avg lebih buruk
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
            if is_main:
                print(f"[soup] eval gagal setelah averaging: {e}", flush=True)
            avg_loss = float("inf")
        finally:
            self._evaluating = False

        avg_loss = self._sync_scalar(model, avg_loss)

        if avg_loss < self.best_loss - 1e-4:
            if is_main:
                print(
                    f"[soup] rata-rata LEBIH BAIK: {avg_loss:.4f} < best={self.best_loss:.4f} "
                    f"(delta {self.best_loss - avg_loss:.4f})",
                    flush=True,
                )
            self.best_loss = avg_loss
            self._save_to_submission(model, avg_loss)
        else:
            if is_main:
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

        # Kandidat ketiga: EMA (setelah keputusan pool-avg selesai)
        self._try_ema_candidate(model)

        # Cleanup pool → bebaskan RAM
        for entry in self._pool_data:
            entry["state"] = None
        self._pool_data.clear()
        self._pool_keys.clear()
        if is_main and current_state is not None:
            del current_state
        gc.collect()

        if is_main:
            print(
                f"[soup] selesai: best_loss={self.best_loss:.4f} @ step={self.best_step}",
                flush=True,
            )

        # Barrier akhir — semua rank harus selesai sebelum trainer lanjut.
        # Mengikuti pola winner yang selalu menutup on_train_end dengan barrier.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
