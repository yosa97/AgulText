"""
stratified_split.py — Validation split yang merepresentasikan distribusi data.

Masalah dengan random split biasa:
  - Dev set bisa kebetulan berisi semua sample pendek → eval_loss tidak
    mencerminkan performa pada sample panjang
  - Dev set bisa berisi duplikat dari train set → eval terlalu optimis

Solusi:
  A. Length-stratified sampling: dev set diambil proporsional dari setiap
     quartile panjang sequence, sehingga distribusi panjang dev ≈ distribusi train.
  B. Label-hash dedup guard: sample yang token-sequence-nya identik dialokasikan
     ke sisi yang sama (train), agar eval tidak "melihat" data yang sudah
     di-training.

Implementasi ringan tanpa MinHash LSH — cukup untuk ukuran dataset
fine-tuning tournament (biasanya ratusan hingga puluhan ribu sample).

Dipanggil dari tokenize_instruct.py sebagai pengganti split random.
"""

import random
from collections import defaultdict
from typing import Callable, Optional


# ── Helpers ──────────────────────────────────────────────────────────────────

def _quartile_bin(value: float, q25: float, q50: float, q75: float) -> int:
    """Masukkan value ke bin quartile 0-3."""
    if value <= q25:
        return 0
    if value <= q50:
        return 1
    if value <= q75:
        return 2
    return 3


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    """Hitung Q25, Q50, Q75 dari list nilai."""
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 4], s[n // 2], s[3 * n // 4]


def _token_fingerprint(sample: dict) -> int:
    """Hash cepat untuk dedup check: ambil 32 token pertama input_ids + panjang."""
    ids = sample.get("input_ids", [])
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    prefix = tuple(ids[:32]) if len(ids) >= 32 else tuple(ids)
    return hash((prefix, len(ids)))


# ── Main function ─────────────────────────────────────────────────────────────

def length_stratified_split(
    samples: list[dict],
    dev_ratio: float = 0.1,
    max_dev: int = 800,
    min_dev: int = 50,
    seed: int = 42,
    length_fn: Optional[Callable[[dict], int]] = None,
) -> tuple[list[dict], list[dict]]:
    """Bagi dataset menjadi (dev, train) dengan stratifikasi panjang sequence.

    Args:
        samples     : Seluruh tokenized dataset.
        dev_ratio   : Proporsi target dev set (default 10%).
        max_dev     : Batas atas ukuran dev (hindari dev yang terlalu besar).
        min_dev     : Batas bawah ukuran dev.
        seed        : Seed random untuk reproduktibilitas.
        length_fn   : Fungsi panjang per sample. Default: len(input_ids).

    Returns:
        (dev_samples, train_samples)
    """
    rng = random.Random(seed)
    n = len(samples)

    if n < max(10, min_dev * 2):
        # Dataset terlalu kecil → tidak bisa split bermakna
        print(
            f"[stratified_split] dataset kecil ({n} samples), "
            f"pakai 10% random split",
            flush=True,
        )
        shuffled = list(range(n))
        rng.shuffle(shuffled)
        cut = max(1, min(min_dev, n // 10))
        dev_idx = set(shuffled[:cut])
        return (
            [samples[i] for i in range(n) if i in dev_idx],
            [samples[i] for i in range(n) if i not in dev_idx],
        )

    dev_target = max(min_dev, min(max_dev, int(n * dev_ratio)))

    # ── A. Hitung panjang per sample ──
    if length_fn is None:
        def length_fn(s):
            ids = s.get("input_ids", [])
            return len(ids) if not hasattr(ids, "__len__") else len(ids)

    lengths = [float(length_fn(s)) for s in samples]
    q25, q50, q75 = _quartiles(lengths)

    # ── B. Kelompokkan index per quartile bin ──
    bins: dict[int, list[int]] = defaultdict(list)
    for i, l in enumerate(lengths):
        b = _quartile_bin(l, q25, q50, q75)
        bins[b].append(i)

    # ── C. Finger-print dedup: tandai index yang ada duplikatnya ──
    fp_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        fp_to_indices[_token_fingerprint(s)].append(i)

    # Index yang punya duplikat → semua anggota grup ke sisi yang sama
    dup_group_of: dict[int, int] = {}  # index → representative index (terkecil)
    for fp, indices in fp_to_indices.items():
        if len(indices) > 1:
            rep = indices[0]
            for idx in indices:
                dup_group_of[idx] = rep

    # ── D. Sample proporsional dari setiap bin ──
    dev_idx: set[int] = set()
    allocated = 0

    for b in range(4):
        bin_indices = bins[b]
        if not bin_indices:
            continue
        rng.shuffle(bin_indices)
        quota = max(1, round(dev_target * len(bin_indices) / n))

        for idx in bin_indices:
            if allocated >= dev_target:
                break
            # Skip jika duplikat dari sample yang sudah masuk dev
            rep = dup_group_of.get(idx, idx)
            if rep in dev_idx:
                continue
            # Jika idx punya duplikat, masukkan seluruh grup ke dev
            # (bounded: jika grup terlalu besar, skip)
            group = fp_to_indices.get(_token_fingerprint(samples[idx]), [idx])
            if len(group) > max(5, dev_target // 10):
                continue
            dev_idx.add(rep)
            allocated += 1
            if allocated >= quota:
                break

    # Fill jika masih di bawah min_dev
    if allocated < min_dev:
        remaining = [i for i in range(n) if i not in dev_idx]
        rng.shuffle(remaining)
        for idx in remaining:
            if allocated >= min_dev:
                break
            dev_idx.add(idx)
            allocated += 1

    dev_samples = [samples[i] for i in range(n) if i in dev_idx]
    train_samples = [samples[i] for i in range(n) if i not in dev_idx]

    # ── Diagnostics ──
    dev_lens = sorted(lengths[i] for i in range(n) if i in dev_idx)
    train_lens = sorted(lengths[i] for i in range(n) if i not in dev_idx)

    def _summary(ls):
        if not ls:
            return "kosong"
        n_ = len(ls)
        return (
            f"min={ls[0]:.0f} p25={ls[n_//4]:.0f} "
            f"p50={ls[n_//2]:.0f} p75={ls[3*n_//4]:.0f} max={ls[-1]:.0f}"
        )

    n_dup_in_dev = sum(
        1 for i in dev_idx
        if dup_group_of.get(i, i) != i or len(fp_to_indices.get(
            _token_fingerprint(samples[i]), [i]
        )) > 1
    )

    print(
        f"[stratified_split] dev={len(dev_samples)} train={len(train_samples)} "
        f"(target={dev_target}, n_dup_in_dev={n_dup_in_dev})",
        flush=True,
    )
    print(f"[stratified_split] panjang dev  : {_summary(dev_lens)}", flush=True)
    print(f"[stratified_split] panjang train: {_summary(train_lens)}", flush=True)

    return dev_samples, train_samples
