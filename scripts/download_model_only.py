"""Standalone model-download script for AgulText tournament miners.

Downloads a HuggingFace model to the shared cache directory.  Prefers
single-shard safetensors files for speed; falls back to full snapshot.
Logs elapsed time and total bytes written for diagnostics.
"""

import os
import time

from huggingface_hub import snapshot_download
from trainer_downloader import is_safetensors_available, download_from_huggingface
import train_cst as cst
import typer

os.environ["HF_HOME"] = "/workspace/hf_cached/"
os.environ["TMPDIR"] = "/workspace/tmp"


def _get_dir_size_bytes(path: str) -> int:
    """Recursively sum byte sizes of all files under *path*."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            fp = os.path.join(dirpath, fname)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def download_base_model(repo_id: str, save_root: str) -> str:
    """Download *repo_id* into *save_root* and return the local model path."""
    model_name = repo_id.replace("/", "--")
    save_path = os.path.join(save_root, model_name)
    print(f"[download_model_only] target: {save_path}", flush=True)

    if os.path.exists(save_path):
        size_mb = _get_dir_size_bytes(save_path) / 1024 ** 2
        print(
            f"[download_model_only] {repo_id} already cached ({size_mb:.0f} MB). Skipping.",
            flush=True,
        )
        return save_path

    t0 = time.time()
    has_safetensors, safetensors_path = is_safetensors_available(repo_id)
    if has_safetensors and safetensors_path:
        result = download_from_huggingface(repo_id, safetensors_path, save_path)
    else:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=save_path,
            local_dir_use_symlinks=False,
        )
        result = save_path

    elapsed = time.time() - t0
    size_mb = _get_dir_size_bytes(result) / 1024 ** 2
    print(
        f"[download_model_only] done in {elapsed:.1f}s — {size_mb:.0f} MB at {result}",
        flush=True,
    )
    return result


def main(repo_id: str):
    model_dir = cst.CACHE_MODELS_DIR
    os.makedirs(model_dir, exist_ok=True)
    download_base_model(repo_id, model_dir)


if __name__ == "__main__":
    typer.run(main)
