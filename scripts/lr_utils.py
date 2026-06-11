"""Learning-rate utilities for AgulText SN56 tournament miners.

Provides logarithmic LR grids used during the LK (learning-rate search) phase,
plus additional helpers for clamping and warmup scheduling.
"""
import math

# Hard bounds that prevent runaway or vanishingly small learning rates.
_LR_GLOBAL_MIN: float = 1e-8
_LR_GLOBAL_MAX: float = 5e-4


def _suggest_learning_rates(
    best_lr: float,
    n: int,
    log_range: float = 0.4,
) -> list[float]:
    """Return *n* log-spaced candidates centred on *best_lr* ± *log_range* decades."""
    if n < 0:
        raise ValueError("Number of tries (n) cannot be negative.")
    if n == 0:
        return []
    if n == 1:
        return [best_lr]

    lower_bound = best_lr / (10 ** log_range)
    upper_bound = best_lr * (10 ** log_range)

    log_lower = math.log10(lower_bound)
    log_upper = math.log10(upper_bound)

    log_spaced = [
        log_lower + i * (log_upper - log_lower) / (n - 1)
        for i in range(n)
    ]
    return sorted(10 ** v for v in log_spaced)


def suggest_learning_rates(
    best_lr: float,
    n: int,
    log_range: float = 0.2,
) -> list[float]:
    """Log-spaced LR grid; if *n* is even, the exact *best_lr* replaces one endpoint."""
    lrs = _suggest_learning_rates(best_lr, n, log_range)
    if n % 2 == 1:
        return lrs
    # For even grids: drop the leftmost value and insert best_lr in sorted order.
    lrs = lrs[1:] + [best_lr]
    return sorted(lrs)


def extend_learning_rates(
    lr: float,
    n: int,
    log_range: float = 0.2,
) -> list[float]:
    """Like _suggest_learning_rates but guarantees *lr* is the first element."""
    lrs = _suggest_learning_rates(lr, n, log_range)
    closest_idx = min(range(len(lrs)), key=lambda i: abs(lrs[i] - lr))
    lrs[closest_idx] = lr  # exact value, no floating-point drift
    if closest_idx != 0:
        lrs.insert(0, lrs.pop(closest_idx))
    return lrs


def clamp_lr(
    lr: float,
    lo: float = _LR_GLOBAL_MIN,
    hi: float = _LR_GLOBAL_MAX,
) -> float:
    """Hard-clamp *lr* to [*lo*, *hi*], printing a warning if clamped."""
    if lr < lo or lr > hi:
        clamped = max(lo, min(hi, lr))
        print(
            f"[lr_utils] clamp_lr: {lr:.3e} outside [{lo:.0e}, {hi:.0e}] → {clamped:.3e}",
            flush=True,
        )
        return clamped
    return lr


def warmup_lr_rampup(
    step: int,
    warmup_steps: int,
    peak_lr: float,
    start_lr: float = 0.0,
) -> float:
    """Return the linearly ramped LR for *step* during the warmup phase.

    After *warmup_steps* the function simply returns *peak_lr* unchanged — the
    scheduler in HuggingFace Trainer takes over for the decay phase.
    """
    if warmup_steps <= 0 or step >= warmup_steps:
        return peak_lr
    fraction = step / warmup_steps
    return start_lr + fraction * (peak_lr - start_lr)


def find_lr_bracket(
    candidates: list[float],
    scores: list[float],
) -> tuple[float, float]:
    """Given a list of (lr, score) pairs, return (lo, hi) brackets around the best score.

    Useful for narrowing down a second LK sweep after an initial grid.
    Returns (best_lr, best_lr) when only one candidate is provided.
    """
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length")
    if not candidates:
        raise ValueError("Empty candidates list")

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_lr = candidates[best_idx]

    lo = candidates[best_idx - 1] if best_idx > 0 else best_lr
    hi = candidates[best_idx + 1] if best_idx < len(candidates) - 1 else best_lr
    return lo, hi


def test():
    lr = 1e-5
    for n in [3, 4, 5, 6]:
        lrs = extend_learning_rates(lr, n, log_range=0.4)
        print(lrs)
        assert lrs[0] == lr
    print("clamp:", clamp_lr(1e-2))
    print("warmup:", warmup_lr_rampup(5, 10, 1e-4))
    print("bracket:", find_lr_bracket([1e-5, 3e-5, 1e-4], [0.1, 0.9, 0.5]))


if __name__ == "__main__":
    test()
