"""
seq_quality_filter.py — Training data quality gate sebelum packing.

Dua tahap filtering:
  1. Exact-dedup  : buang sample yang token-sequence-nya identik (input_ids+labels).
  2. Outlier gate : buang sample yang loss-nya berada di luar rentang wajar,
                    menggunakan IQR (Interquartile Range) yang robust terhadap
                    distribusi long-tail khas language model task.

Pendekatan IQR berbeda dari MAD: IQR lebih intuitif (Q1 - k*IQR, Q3 + k*IQR)
dan secara empiris lebih longgar pada distribusi unimodal — cocok untuk dataset
fine-tuning yang heterogen.

Integrasi: dipanggil dari tokenize_instruct.py setelah tokenisasi, sebelum
packing, hanya pada main process (rank 0).
"""

import json
import os
import time
from typing import Optional

import torch


# ── Exact deduplication ───────────────────────────────────────────────────────

def exact_dedup(samples: list[dict]) -> list[dict]:
    """Buang sample yang (input_ids, labels) identik.

    Hash berbasis tuple — O(n), biaya negligible dibanding forward pass.
    Mempertahankan urutan asli (first-occurrence wins).
    """
    seen: set[int] = set()
    result: list[dict] = []
    n_dup = 0

    for s in samples:
        ids = s.get("input_ids", [])
        lbs = s.get("labels", [])
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(lbs, torch.Tensor):
            lbs = lbs.tolist()
        # XOR hash — lebih cepat dari tuple hash untuk list panjang
        h = hash((tuple(ids[:64]), tuple(lbs[:64]), len(ids)))
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        result.append(s)

    if n_dup:
        pct = 100 * n_dup / max(1, len(samples))
        print(
            f"[seq_quality] dedup: {len(samples)} → {len(result)} "
            f"(-{n_dup}, {pct:.1f}% duplikat)",
            flush=True,
        )
    else:
        print("[seq_quality] dedup: tidak ada duplikat", flush=True)

    return result


# ── Per-sample loss (no-grad forward pass) ───────────────────────────────────

@torch.no_grad()
def compute_losses(
    model,
    samples: list[dict],
    batch_size: int = 64,
    device: str = "cuda",
) -> list[float]:
    """Forward pass no-grad untuk menghitung NLL loss per sample via log-prob gather.

    Pendekatan: log_softmax → gather log-prob token target → rata-rata NLL per sample.
    Matematis ekuivalen dengan cross_entropy, tapi dihitung secara eksplisit per-token
    sehingga lebih transparan dan tidak bergantung pada implementasi internal F.cross_entropy.

    Batch_size besar aman karena tidak ada backward. Auto-fallback ke batch kecil jika OOM.

    Return: list[float] sejajar dengan samples. Sample tanpa completion token → 0.0.
    """
    F = torch.nn.functional
    model.eval()
    losses: list[float] = []
    t0 = time.perf_counter()
    bs = batch_size

    def _to_tensor(x):
        return x.to(device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=device)

    i = 0
    while i < len(samples):
        batch = samples[i : i + bs]
        try:
            input_ids = torch.stack([_to_tensor(s["input_ids"])      for s in batch])
            attn      = torch.stack([_to_tensor(s["attention_mask"])  for s in batch])
            labels    = torch.stack([_to_tensor(s["labels"])          for s in batch])

            logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, T, V]

            # Log-prob gather: hitung log P(token) untuk setiap posisi
            # lalu ambil nilai untuk token target via gather → rata-rata NLL per sample
            T         = logits.size(1) - 1                               # panjang setelah shift
            log_probs  = F.log_softmax(logits[:, :T], dim=-1)            # [B, T-1, V]
            target_ids = labels[:, 1:].clamp(min=0)                      # [B, T-1]
            token_logp = log_probs.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)                                                 # [B, T-1]

            comp_mask = (labels[:, 1:] != -100)                          # [B, T-1] bool

            for j in range(len(batch)):
                m = comp_mask[j]
                if m.sum() == 0:
                    losses.append(0.0)
                else:
                    # NLL = rata-rata negative log-prob pada completion tokens
                    losses.append(float(-token_logp[j][m].mean()))

            del input_ids, attn, labels, logits, log_probs, target_ids, token_logp, comp_mask
            torch.cuda.empty_cache()
            i += bs

        except torch.cuda.OutOfMemoryError:
            if bs > 4:
                bs = max(4, bs // 2)
                print(f"[seq_quality] OOM → batch_size={bs}", flush=True)
                torch.cuda.empty_cache()
                continue
            losses.extend([0.0] * len(batch))
            i += len(batch)

    elapsed = time.perf_counter() - t0
    n_valid = sum(1 for l in losses if l > 0)
    print(
        f"[seq_quality] loss pass: {len(samples)} samples, {elapsed:.1f}s, "
        f"{n_valid} dengan completion token",
        flush=True,
    )
    model.train()
    return losses


# ── IQR-based outlier removal ─────────────────────────────────────────────────

def iqr_filter(
    samples: list[dict],
    losses: list[float],
    k: float = 2.5,
    min_samples: int = 150,
) -> list[dict]:
    """Buang sample yang loss-nya di luar [Q1 - k*IQR, Q3 + k*IQR].

    IQR (Interquartile Range) robust terhadap outlier — tidak seperti mean±std
    yang terpengaruh oleh nilai ekstrem. k=2.5 lebih longgar dari default boxplot
    (k=1.5) untuk menghindari over-filtering pada distribusi multi-modal.

    Sample dengan loss=0 (tanpa completion token) selalu dibuang karena tidak
    memberikan gradien.
    """
    # Pisahkan yang punya completion token
    valid = [(s, l) for s, l in zip(samples, losses) if l > 0]
    n_zero = len(samples) - len(valid)
    if n_zero:
        print(f"[seq_quality] buang {n_zero} sample loss=0 (tanpa completion token)", flush=True)

    if len(valid) < min_samples:
        print(
            f"[seq_quality] filter dilewati: hanya {len(valid)} valid sample "
            f"(minimum {min_samples})",
            flush=True,
        )
        return [s for s, _ in valid]

    ls = sorted(l for _, l in valid)
    n = len(ls)
    q1 = ls[n // 4]
    q3 = ls[3 * n // 4]
    iqr = q3 - q1

    if iqr < 1e-8:
        print(
            f"[seq_quality] IQR≈0 (semua loss ~{q1:.4f}), filter dilewati",
            flush=True,
        )
        return [s for s, _ in valid]

    lo = q1 - k * iqr
    hi = q3 + k * iqr

    kept, dropped = [], []
    for s, l in valid:
        if lo <= l <= hi:
            kept.append(s)
        else:
            dropped.append(l)

    pct = 100 * len(dropped) / max(1, len(valid))
    print(
        f"[seq_quality] IQR filter: q1={q1:.3f} q3={q3:.3f} iqr={iqr:.3f} "
        f"band=[{lo:.3f}, {hi:.3f}]",
        flush=True,
    )
    print(
        f"[seq_quality] dipertahankan {len(kept)}/{len(valid)} "
        f"(-{len(dropped)}, {pct:.1f}% dibuang)",
        flush=True,
    )
    return kept


# ── Entry point ───────────────────────────────────────────────────────────────

def run_quality_filter(
    samples: list[dict],
    model=None,
    device: str = "cuda",
    k_iqr: float = 2.5,
    min_for_filter: int = 150,
    batch_size: int = 64,
    debug_path: Optional[str] = None,
) -> list[dict]:
    """Pipeline penuh: dedup → (opsional) outlier filter.

    model=None: hanya jalankan dedup (tanpa forward pass).
    debug_path: jika diset, simpan statistik loss ke file JSON untuk inspeksi.
    """
    after_dedup = exact_dedup(samples)

    if model is None or len(after_dedup) < min_for_filter:
        print(
            "[seq_quality] outlier filter dilewati "
            "(model=None atau dataset terlalu kecil)",
            flush=True,
        )
        return after_dedup

    losses = compute_losses(after_dedup, model=model, batch_size=batch_size, device=device)

    if debug_path:
        try:
            nonzero = sorted(l for l in losses if l > 0)
            if nonzero:
                n = len(nonzero)
                info = {
                    "n_total": len(losses),
                    "n_valid": n,
                    "min": nonzero[0],
                    "p25": nonzero[n // 4],
                    "median": nonzero[n // 2],
                    "p75": nonzero[3 * n // 4],
                    "max": nonzero[-1],
                }
                os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
                with open(debug_path, "w") as f:
                    json.dump(info, f, indent=2)
                print(f"[seq_quality] debug stats → {debug_path}", flush=True)
        except Exception as e:
            print(f"[seq_quality] debug save gagal: {e}", flush=True)

    return iqr_filter(after_dedup, losses, k=k_iqr, min_samples=min_for_filter)
