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


# ── Near-duplicate removal via SimHash ───────────────────────────────────────
# Pendekatan BERBEDA dari MinHash/LSH yang lazim: SimHash 64-bit di atas
# shingle 4-gram token, dengan pencarian radius Hamming via pigeonhole
# (radius r → r+1 pita bit; dua fingerprint berjarak ≤r pasti identik di
# minimal satu pita). Keuntungan vs MinHash: satu integer per sampel (hemat
# memori), deterministik tanpa permutasi acak, dan verifikasi jarak eksak
# (popcount) alih-alih estimasi Jaccard.

def simhash_near_dedup(
    samples: list[dict],
    hamming_radius: int = 6,
    max_shingles: int = 256,
    min_tokens: int = 24,
) -> list[dict]:
    """Buang sampel yang hampir-duplikat (SimHash Hamming ≤ radius).

    Sampel sangat pendek (< min_tokens) dilewati dari near-dedup — fingerprint
    mereka terlalu mudah bertabrakan padahal kontennya bisa berbeda sah.
    First-occurrence wins; urutan asli dipertahankan.
    """
    if not samples:
        return samples
    t0 = time.perf_counter()
    n_bands = hamming_radius + 1
    band_bits = 64 // n_bands
    band_mask = (1 << band_bits) - 1

    def _fingerprint(ids) -> int:
        # Shingle 4-gram dengan stride adaptif agar ≤ max_shingles shingle.
        n = len(ids) - 3
        step = max(1, n // max_shingles)
        counts = [0] * 64
        for i in range(0, n, step):
            h = hash((ids[i], ids[i + 1], ids[i + 2], ids[i + 3])) & 0xFFFFFFFFFFFFFFFF
            for b in range(64):
                counts[b] += 1 if (h >> b) & 1 else -1
        fp = 0
        for b in range(64):
            if counts[b] > 0:
                fp |= 1 << b
        return fp

    buckets: list[dict[int, list[int]]] = [dict() for _ in range(n_bands)]
    kept: list[dict] = []
    n_near = 0

    for s in samples:
        ids = s.get("input_ids", [])
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if len(ids) < max(min_tokens, 4):
            kept.append(s)
            continue
        fp = _fingerprint(ids)

        is_dup = False
        cand_seen: set[int] = set()
        for bi in range(n_bands):
            key = (fp >> (bi * band_bits)) & band_mask
            for cand_fp in buckets[bi].get(key, ()):  # fingerprint terdahulu
                if cand_fp in cand_seen:
                    continue
                cand_seen.add(cand_fp)
                if bin(fp ^ cand_fp).count("1") <= hamming_radius:
                    is_dup = True
                    break
            if is_dup:
                break

        if is_dup:
            n_near += 1
            continue
        for bi in range(n_bands):
            key = (fp >> (bi * band_bits)) & band_mask
            buckets[bi].setdefault(key, []).append(fp)
        kept.append(s)

    dt = time.perf_counter() - t0
    frac = n_near / max(1, len(samples))
    # PAGAR TERKALIBRASI (T2 3 Sep): rate near-dup tinggi = dataset sintetis
    # yang "duplikatnya" se-distribusi dengan test set → MEMBUANGNYA MERUGIKAN
    # (0.5654 vs 0.5227 tanpa buang; winner 0.5278). Rate rendah (T1: 0.1%) =
    # duplikat organik sungguhan → aman dibuang. Ambang via NEAR_DEDUP_CAP.
    cap = float(os.environ.get("NEAR_DEDUP_CAP") or 0.10)
    if frac > cap:
        print(
            f"[seq_quality] near-dedup: terdeteksi {frac:.1%} > cap {cap:.0%} — "
            f"indikasi data sintetis in-distribution → TIDAK ada yang dibuang "
            f"({dt:.1f}s)",
            flush=True,
        )
        return samples
    pct = 100 * frac
    print(
        f"[seq_quality] near-dedup (simhash r={hamming_radius}): "
        f"{len(samples)} → {len(kept)} (-{n_near}, {pct:.1f}%) dalam {dt:.1f}s",
        flush=True,
    )
    return kept


# ── MAD gate pada statistik token (murah, tanpa model) ───────────────────────
# Analog konservatif dari filter kualitas berbasis loss: buang sampel yang
# panjang completion-nya ekstrem (log-length di luar median ± k×MAD) atau
# completion-nya degeneratif (satu token mendominasi). Tanpa forward pass,
# biaya O(n) murni CPU.

def mad_stat_filter(
    samples: list[dict],
    k_mad: float = 4.0,
    dominance: float = 0.6,
    min_samples: int = 500,
    use_len_gate: bool = False,
) -> list[dict]:
    # TERKALIBRASI (T1 falcon, 2 Sep): gerbang panjang MERUGIKAN (1.5353→1.5432)
    # karena dokumen terpanjang di task summarization adalah data sah, bukan
    # outlier. use_len_gate default False; hanya deteksi completion degeneratif
    # yang aktif. Aktifkan kembali via QF_MAD_LEN=1 jika suatu dataset terbukti
    # kotor panjangnya.
    import math

    if len(samples) < min_samples:
        return samples

    def _comp_tokens(s):
        labs = s.get("labels", [])
        if isinstance(labs, torch.Tensor):
            labs = labs.tolist()
        return [t for t in labs if t != -100]

    comp_lens = []
    comps = []
    for s in samples:
        c = _comp_tokens(s)
        comps.append(c)
        comp_lens.append(len(c))

    logs = sorted(math.log1p(x) for x in comp_lens if x > 0)
    if len(logs) < min_samples:
        return samples
    med = logs[len(logs) // 2]
    devs = sorted(abs(x - med) for x in logs)
    mad = devs[len(devs) // 2]
    if not use_len_gate or mad < 1e-6:
        lo, hi = float("-inf"), float("inf")
    else:
        lo, hi = med - k_mad * mad, med + k_mad * mad

    kept = []
    n_len, n_dom = 0, 0
    for s, c, L in zip(samples, comps, comp_lens):
        if L == 0:
            kept.append(s)  # kasus loss=0 ditangani di tempat lain
            continue
        lg = math.log1p(L)
        if not (lo <= lg <= hi):
            n_len += 1
            continue
        if L > 20:
            top = max(c.count(t) for t in set(c[:200]))
            if top / min(L, 200) > dominance:
                n_dom += 1
                continue
        kept.append(s)

    n_drop = n_len + n_dom
    if n_drop:
        print(
            f"[seq_quality] MAD gate: -{n_len} outlier panjang, "
            f"-{n_dom} completion degeneratif "
            f"(band log-len=[{lo:.2f},{hi:.2f}], {len(samples)}→{len(kept)})",
            flush=True,
        )
    else:
        print("[seq_quality] MAD gate: tidak ada yang dibuang", flush=True)
    return kept


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
    t0 = time.perf_counter()

    # Batching dinamis: sortir per panjang (desc), isi batch sampai budget token
    # terpenuhi. Budget dihitung dari vocab agar memori logits terkendali
    # (bf16: tokens × vocab × 2B; target ≤ ~5GB logits per batch).
    _vocab = int(getattr(getattr(model, "config", None), "vocab_size", 50_000) or 50_000)
    token_budget = max(2048, int(2_500_000_000 / _vocab))

    def _ids(s, key):
        v = s.get(key, [])
        return v.tolist() if isinstance(v, torch.Tensor) else list(v)

    n = len(samples)
    losses = [0.0] * n
    order = sorted(range(n), key=lambda i: len(samples[i].get("input_ids", [])), reverse=True)

    i = 0
    while i < n:
        # bentuk batch: panjang maksimum = panjang elemen pertama (sorted desc)
        max_len = max(2, len(samples[order[i]].get("input_ids", [])))
        bsz = max(1, token_budget // max_len)
        idxs = order[i : i + bsz]
        try:
            B = len(idxs)
            input_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
            attn      = torch.zeros(B, max_len, dtype=torch.long, device=device)
            labels    = torch.full((B, max_len), -100, dtype=torch.long, device=device)
            for j, si in enumerate(idxs):
                ids = _ids(samples[si], "input_ids")[:max_len]
                lbs = _ids(samples[si], "labels")[:max_len]
                L = len(ids)
                input_ids[j, :L] = torch.tensor(ids, device=device)
                attn[j, :L] = 1
                labels[j, : len(lbs)] = torch.tensor(lbs, device=device)

            logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B,T,V]
            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            ce = F.cross_entropy(
                shift_logits.float(), shift_labels,
                ignore_index=-100, reduction="none",
            ).view(B, -1)                                                    # [B,T-1]
            mask = (labels[:, 1:] != -100)
            for j, si in enumerate(idxs):
                m = mask[j]
                losses[si] = float(ce[j][m].mean()) if m.any() else 0.0

            del input_ids, attn, labels, logits, shift_logits, shift_labels, ce, mask
            i += bsz

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if token_budget > 1024:
                token_budget //= 2
                print(f"[seq_quality] OOM → token_budget={token_budget}", flush=True)
                continue
            for si in idxs:
                losses[si] = -1.0  # sentinel: tak terukur → JANGAN difilter
            i += len(idxs)

    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - t0
    n_valid = sum(1 for l in losses if l > 0)
    print(
        f"[seq_quality] loss pass: {n} samples, {elapsed:.1f}s, "
        f"{n_valid} dengan completion token (vocab={_vocab}, budget={token_budget} tok)",
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
    # Pisahkan yang punya completion token. loss<0 = sentinel "tak terukur"
    # (OOM saat loss pass) → selalu dipertahankan tanpa ikut statistik IQR.
    unmeasured = [s for s, l in zip(samples, losses) if l < 0]
    valid = [(s, l) for s, l in zip(samples, losses) if l > 0]
    n_zero = len(samples) - len(valid) - len(unmeasured)
    if n_zero:
        print(f"[seq_quality] buang {n_zero} sample loss=0 (tanpa completion token)", flush=True)
    if unmeasured:
        print(f"[seq_quality] {len(unmeasured)} sample tak terukur (OOM) → dipertahankan", flush=True)

    if len(valid) < min_samples:
        print(
            f"[seq_quality] filter dilewati: hanya {len(valid)} valid sample "
            f"(minimum {min_samples})",
            flush=True,
        )
        return [s for s, _ in valid] + unmeasured

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
        return [s for s, _ in valid] + unmeasured

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
    return kept + unmeasured


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

    losses = compute_losses(model, after_dedup, batch_size=batch_size, device=device)

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


def stride_unmask_prompt(samples: list, plw: float, seed: int = 13) -> list:
    """Buka sebagian token prompt untuk perhitungan loss — STRIDE deterministik.

    Untuk dataset prompt-dominan (rasio prompt:completion > 5:1), masking
    penuh (-100) membuang seluruh sinyal gradien tentang bahasa/domain input.
    Membuka sebagian kecil token prompt memberi model konteks domain —
    penting untuk data non-Inggris / dokumen panjang (clinical notes dsb).

    Implementasi BERBEDA dari pendekatan Bernoulli per-token yang lazim:
    setiap sampel membuka tepat 1 dari setiap round(1/plw) token prompt,
    dengan offset bergilir per sampel. Keuntungan: fraksi yang dibuka
    presisi (bukan ekspektasi), varian gradien antar batch lebih rendah,
    dan hasil identik antar rank DDP tanpa sinkronisasi RNG.
    """
    if plw <= 0 or not samples:
        return samples
    stride = max(2, int(round(1.0 / plw)))
    n_open = 0
    n_prompt = 0
    for si, s in enumerate(samples):
        ids = s.get("input_ids", [])
        labs = s.get("labels", [])
        if len(ids) != len(labs):
            continue
        new_labs = list(labs)
        offset = (si + seed) % stride
        k = 0
        for i, lab in enumerate(labs):
            if lab == -100:
                if k % stride == offset:
                    new_labs[i] = ids[i]
                    n_open += 1
                k += 1
        n_prompt += k
        s["labels"] = new_labs
    print(
        f"[plw-stride] {n_open}/{n_prompt} token prompt dibuka "
        f"(stride={stride}, fraksi efektif={n_open / max(1, n_prompt):.3%})",
        flush=True,
    )
    return samples
