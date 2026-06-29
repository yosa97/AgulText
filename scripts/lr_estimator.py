"""
Adaptive learning rate estimator for SN56 text mining tasks.

Estimates a principled starting LR from model weight statistics,
replacing hash-based lookup tables with physics-based calculation
that adapts to any model/dataset combination automatically.

Core algorithm (Adam optimizer theory):
  For Adam, the effective relative weight update ≈ lr × (g / sqrt(v)) / w_rms
  At initialization: v ≈ g² → relative_update ≈ lr / w_rms
  We target: relative_update = η (task-specific, empirically tuned)
  Therefore: lr_base = η × w_rms

  Corrections applied for:
  - Model scale  : larger models → smaller relative updates → scale 1/sqrt(N/N_ref)
  - Batch size   : larger batches → less gradient noise → can increase LR
  - Time pressure: tight budget → slightly higher LR to converge faster

Ensemble mode: when lrs/instruct.json has an entry for this model, we compute
the geometric mean of the stats-based estimate and the historical lookup value.
This blends empirical historical performance with model-specific physics.
"""
import glob
import hashlib
import json
import math
import os
from typing import Optional


# Task-specific target relative weight update (η) and LR safety bounds.
# Derived independently from SN56 tournament observation and Adam theory.
_TASK_PROFILE: dict = {
    "InstructTextTask": {"eta": 2.5e-3, "lo": 5e-7, "hi": 8e-4},
    "ChatTask":         {"eta": 2.5e-3, "lo": 5e-7, "hi": 8e-4},
    "DpoTask":          {"eta": 5e-4,   "lo": 2e-7, "hi": 2e-5},
    "GrpoTask":         {"eta": 4e-4,   "lo": 3e-7, "hi": 1e-5},
}

# Reference parameter count for scale correction (1 B)
_REF_PARAMS = 1_000_000_000
# Reference effective batch for batch correction
_REF_BATCH = 64

# Projection weight name suffixes common across transformer families
_PROJ_SUFFIXES = (
    ".q_proj.weight", ".k_proj.weight", ".v_proj.weight",
    ".o_proj.weight", ".gate_proj.weight", ".up_proj.weight",
    ".down_proj.weight", ".c_attn.weight", ".c_proj.weight",
    ".dense.weight", ".dense_h_to_4h.weight", ".dense_4h_to_h.weight",
    ".wi_0.weight",  ".wi_1.weight",  ".wo.weight",   # T5/FLAN variants
)


def _median(values: list) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _sample_weight_rms(model_path: str, n_target: int = 8) -> Optional[float]:
    """
    Memory-map a few projection tensors from safetensors shards and return
    the median weight RMS.  Uses at most n_target tensors.

    Returns None if safetensors is unavailable or the model has no matching keys.
    """
    sf_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not sf_files:
        return None

    rms_values: list = []
    try:
        import numpy as np
        from safetensors import safe_open  # type: ignore[import]

        for sf_path in sf_files:
            # framework="pt" agar bfloat16 tensor terbaca sebagai PyTorch tensor
            # (numpy tidak mengenal bfloat16, sehingga framework="numpy" error)
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                all_keys = list(f.keys())
                proj_keys = [k for k in all_keys
                             if any(k.endswith(s) for s in _PROJ_SUFFIXES)]
                if not proj_keys:
                    continue
                stride = max(1, len(proj_keys) // n_target)
                for key in proj_keys[::stride][:n_target]:
                    t = f.get_tensor(key).float().numpy()   # bfloat16→float32 via torch
                    rms = float(np.sqrt(np.mean(t ** 2)))
                    if rms > 1e-9:
                        rms_values.append(rms)
                    if len(rms_values) >= n_target:
                        break
            if len(rms_values) >= n_target:
                break
    except Exception as exc:
        print(f"[lr_estimator] safetensors sampling error: {exc}", flush=True)
        return None

    return _median(rms_values) if rms_values else None


def _heuristic_weight_rms(param_count: int) -> float:
    """
    Fallback weight RMS based on standard initialization theory:
    typical transformer projection weights init with std ≈ 1/sqrt(d_model),
    giving weight_rms ∈ [0.010, 0.055] across common model sizes.
    """
    ref_rms = 0.028                                  # geometric centre of typical range
    scale = (1e9 / max(param_count, 1)) ** 0.10     # mild N-dependence
    return float(max(0.010, min(0.060, ref_rms * scale)))


def _lookup_lr(model_name: str, task_type: str) -> Optional[float]:
    """
    Try to retrieve a previously measured optimal LR from lrs/instruct.json.
    Returns None when the lookup file is absent or the model has no entry.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        task_map = {
            "InstructTextTask": "instruct",
            "ChatTask": "instruct",
            "DpoTask": "dpo",
            "GrpoTask": "grpo",
        }
        fname = task_map.get(task_type, "instruct")
        lrs_path = os.path.join(script_dir, "lrs", f"{fname}.json")
        if not os.path.exists(lrs_path):
            return None
        with open(lrs_path) as f:
            records = json.load(f)
        model_hash = hashlib.sha256(model_name.encode()).hexdigest()
        index = {rec["h"]: rec["lr"] for rec in records}
        return index.get(model_hash)
    except Exception:
        return None


def estimate_lr(
    model_path: str,
    model_name: str,
    task_type: str,
    param_count: int,
    effective_batch_size: int,
    hours_to_complete: float = 1.0,
    fallback_lr: Optional[float] = None,
) -> float:
    """
    Estimate an optimal starting learning rate for the given model and task.

    Combines two signals via geometric mean (ensemble):
      1. Stats-based estimate from model weight RMS (adaptive, model-agnostic)
      2. Historical lookup from lrs/instruct.json (specific, experience-based)

    When only one signal is available, that signal is used directly.

    Args:
        model_path          : Local path to the base model (for weight sampling)
        model_name          : HuggingFace model name (for lookup table)
        task_type           : "InstructTextTask" | "DpoTask" | etc.
        param_count         : Total model parameter count
        effective_batch_size: batch_size × grad_accum_steps × gpu_count
        hours_to_complete   : Training time budget in hours
        fallback_lr         : Used when all estimation methods fail

    Returns:
        Estimated starting LR (float)
    """
    profile = _TASK_PROFILE.get(task_type, _TASK_PROFILE["InstructTextTask"])

    try:
        # ── Step 1: sample weight RMS ──────────────────────────────────────
        w_rms = _sample_weight_rms(model_path)
        if w_rms is None:
            w_rms = _heuristic_weight_rms(param_count)
            print(f"[lr_estimator] Using heuristic w_rms={w_rms:.4f}", flush=True)
        else:
            print(f"[lr_estimator] Sampled w_rms={w_rms:.4f}", flush=True)

        # ── Step 2: base LR from target relative update ──────────────────
        lr_stats = profile["eta"] * w_rms

        # ── Step 3: model scale correction ───────────────────────────────
        scale_factor = math.sqrt(_REF_PARAMS / max(param_count, 1))

        # ── Step 4: batch size correction (linear scaling law, capped 2×) ──
        batch_factor = min(2.0, math.sqrt(
            max(effective_batch_size, _REF_BATCH) / _REF_BATCH
        ))

        # ── Step 5: time-pressure correction ─────────────────────────────
        if hours_to_complete < 0.5:
            time_factor = 1.20   # very tight: push LR for faster convergence
        elif hours_to_complete < 1.0:
            time_factor = 1.10
        elif hours_to_complete > 4.0:
            time_factor = 0.88   # long budget: lower LR for fine-grained descent
        else:
            time_factor = 1.00

        lr_stats = lr_stats * scale_factor * batch_factor * time_factor

        # ── Step 6: ensemble with historical lookup (geometric mean) ──────
        lr_lookup = _lookup_lr(model_name, task_type)
        if lr_lookup is not None and lr_lookup > 0:
            lr_final = math.sqrt(lr_stats * lr_lookup)
            print(
                f"[lr_estimator] ensemble: stats={lr_stats:.3e} "
                f"lookup={lr_lookup:.3e} → geo_mean={lr_final:.3e}",
                flush=True,
            )
        else:
            lr_final = lr_stats
            print(
                f"[lr_estimator] stats-only (no lookup): lr={lr_final:.3e}",
                flush=True,
            )

        # ── Step 7: safety clamp ─────────────────────────────────────────
        lr_final = max(profile["lo"], min(profile["hi"], lr_final))

        print(
            f"[lr_estimator] {task_type}: FINAL lr={lr_final:.3e} "
            f"(scale={scale_factor:.3f} batch={batch_factor:.3f} "
            f"time={time_factor:.3f} params={param_count/1e9:.2f}B)",
            flush=True,
        )
        return lr_final

    except Exception as exc:
        print(f"[lr_estimator] Estimation failed ({exc}), using fallback", flush=True)
        if fallback_lr is not None:
            return fallback_lr
        _safe = {
            "InstructTextTask": 5e-5, "ChatTask": 5e-5,
            "DpoTask": 5e-6, "GrpoTask": 1e-6,
        }
        return _safe.get(task_type, 5e-5)
