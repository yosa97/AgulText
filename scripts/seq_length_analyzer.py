"""
Sequence length analyzer for adaptive max_length selection.

Reads tokenized dataset files produced by tokenize_instruct.py and computes
the optimal max_length for training based on the ACTUAL data distribution.

Why this matters:
  - Default max_length = 2048 is wasteful for short-context tasks (e.g., wiki_qa
    averages ~100 tokens) → padding dominates → fewer effective training steps.
  - Adaptive max_length can be 4–8× smaller for short tasks, enabling:
      (a) larger effective batch sizes
      (b) more gradient update steps per unit time
      (c) avoidance of unnecessary VRAM usage, preventing OOM
  - For long-context tasks, adaptive max_length can exceed 2048 if the model
    supports it and the data warrants it.

Rounding to multiples of 64 aligns with CUDA memory layout for efficiency.
"""
import json
import os
from typing import Optional


def _percentile(sorted_vals: list, frac: float) -> int:
    """Return percentile value from a pre-sorted list (frac in [0.0, 1.0])."""
    if not sorted_vals:
        return 0
    idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * frac)))
    return sorted_vals[idx]


def read_token_lengths(tokenized_path: str) -> list:
    """
    Extract sequence lengths from a tokenized JSON file.
    Handles both {'input_ids': [...]} and {'tokens': [...]} record formats.
    """
    if not os.path.exists(tokenized_path):
        return []
    lengths: list = []
    try:
        with open(tokenized_path, "r") as f:
            records = json.load(f)
        for rec in records:
            ids = rec.get("input_ids") or rec.get("tokens") or []
            if ids:
                lengths.append(len(ids))
    except Exception as exc:
        print(f"[seq_analyzer] Error reading {tokenized_path}: {exc}", flush=True)
    return lengths


def compute_adaptive_max_length(
    task_id: str,
    datasets_dir: str = "datasets",
    default_max: int = 2048,
    packing: bool = True,
    model_max_positions: Optional[int] = None,
) -> int:
    """
    Compute optimal max_length from the tokenized training dataset.

    Strategy:
      Packing ON  → use p90 of sequence lengths.
        With packing, most sequences are packed together; shorter target means
        tighter bins and less padding waste.
      Packing OFF → use p95.
        Without packing, we need more headroom to avoid truncating tail examples.

    The computed value is:
      - Rounded UP to nearest multiple of 64 (CUDA memory alignment)
      - Lower-bounded at 256 (minimum sensible context)
      - Upper-bounded at model_max_positions (hardware/architecture limit)
      - Allowed to go up to 4× the original default (for long-context tasks)
      - Never shrunk below default // 4 (avoid extreme reduction)

    Args:
        task_id           : Used to locate tokenized files as
                            {datasets_dir}/train_tokenized_{task_id}.json
        datasets_dir      : Directory containing tokenized dataset files
        default_max       : Fallback value when analysis is unavailable
        packing           : Whether sequence packing is enabled in training
        model_max_positions: Model's max_position_embeddings (hard cap)

    Returns:
        Recommended max_length integer value
    """
    tok_path = os.path.join(datasets_dir, f"train_tokenized_{task_id}.json")
    lengths = read_token_lengths(tok_path)

    if not lengths:
        print(
            f"[seq_analyzer] No tokenized data found at {tok_path}, "
            f"using default max_length={default_max}",
            flush=True,
        )
        return default_max

    lengths.sort()
    n = len(lengths)
    p50 = _percentile(lengths, 0.50)
    p90 = _percentile(lengths, 0.90)
    p95 = _percentile(lengths, 0.95)
    p99 = _percentile(lengths, 0.99)
    mean_len = sum(lengths) // n

    # Select target percentile based on packing strategy
    raw_target = p90 if packing else p95

    # Round up to nearest 64
    aligned = ((raw_target + 63) // 64) * 64

    # Apply bounds
    aligned = max(256, aligned)                     # minimum sensible context

    if model_max_positions and model_max_positions > 0:
        aligned = min(aligned, model_max_positions) # respect model hard limit

    aligned = min(aligned, default_max * 4)         # cap at 4× default
    aligned = max(aligned, default_max // 4)        # don't shrink below 25% default

    print(
        f"[seq_analyzer] dataset(n={n}): "
        f"mean={mean_len} p50={p50} p90={p90} p95={p95} p99={p99} max={lengths[-1]} "
        f"→ max_length={aligned} (packing={'on' if packing else 'off'})",
        flush=True,
    )
    return aligned
