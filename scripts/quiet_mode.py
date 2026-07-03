"""
AgulText SN56 — Log suppression gate.

Import sebagai baris PERTAMA di setiap entrypoint (text_trainer.py,
train_instruct.py, tokenize_instruct.py, dst.) SEBELUM import library berat:

    import quiet_mode  # noqa: F401,E402

Mengapa harus pertama: library seperti transformers dan datasets membaca
env vars verbosity pada saat import — kalau kita set setelah import sudah
terlambat.

Mode default: QUIET (aktif).
- Semua print() ke stdout → no-op (strategi training tidak bocor ke log publik)
- Library log level dipaksa ke ERROR
- tqdm progress bar dimatikan
- print(..., file=sys.stderr) dan traceback tetap tampil

Untuk debug lokal, nonaktifkan dengan:
    AGULTEXT_VERBOSE=1 bash examples/run_instruct.sh
atau
    AGULTEXT_QUIET=0 bash examples/run_instruct.sh

Modul ini tidak boleh raise exception apapun — semua dibungkus try/except
agar tidak merusak import entrypoint.
"""

import builtins
import functools
import logging
import os
import sys

_real_print = builtins.print


def _silent_print(*args, **kwargs):
    """Versi diam dari print() — hanya teruskan ke stderr, buang stdout."""
    if kwargs.get("file") is sys.stderr:
        return _real_print(*args, **kwargs)


# Agar modul yang introspect print (misalnya numba) tidak error karena
# __module__ atau __name__ berbeda, kita impersonasikan builtins.print.
try:
    functools.update_wrapper(_silent_print, _real_print)
except Exception:
    pass


def _quiet_active() -> bool:
    """True jika mode quiet aktif (default)."""
    _env_verbose = os.environ.get("AGULTEXT_VERBOSE", "").strip().lower()
    _env_quiet = os.environ.get("AGULTEXT_QUIET", "").strip().lower()
    if _env_verbose in ("1", "true", "yes", "on"):
        return False
    if _env_quiet in ("0", "false", "no", "off"):
        return False
    return True


def _apply_library_silence() -> None:
    """Set env vars sebelum library diimport + matikan logging handlers."""
    _defaults = {
        "TQDM_DISABLE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "DATASETS_VERBOSITY": "error",
        "DATASETS_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_LOGGING_LEVEL": "ERROR",
        "BITSANDBYTES_NOWELCOME": "1",
        "ACCELERATE_LOG_LEVEL": "error",
    }
    for _k, _v in _defaults.items():
        os.environ.setdefault(_k, _v)

    logging.disable(logging.WARNING)
    logging.getLogger().setLevel(logging.ERROR)
    for _name in (
        "transformers",
        "datasets",
        "accelerate",
        "deepspeed",
        "vllm",
        "axolotl",
        "torch",
        "huggingface_hub",
        "filelock",
        "urllib3",
    ):
        try:
            logging.getLogger(_name).setLevel(logging.ERROR)
        except Exception:
            pass


def _apply_print_silence() -> None:
    """Ganti builtins.print dengan versi diam."""
    builtins.print = _silent_print


def _activate() -> None:
    if not _quiet_active():
        return
    try:
        _apply_library_silence()
    except Exception:
        pass
    try:
        _apply_print_silence()
    except Exception:
        pass


_activate()
