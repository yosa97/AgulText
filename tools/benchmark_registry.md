# Registry Benchmark — Task Tournament Historis

Daftar task nyata untuk kalibrasi lintas-arsitektur. Model winner publik di HF
(`gradients-io-tournaments/...`) — eval mereka sekaligus memvalidasi evaluator lokal.

Alur per task:
1. Ambil `training_data` URL segar: `curl https://api.gradients.io/... /tasks/{task_id}`
   (atau dari dashboard — presigned URL kedaluwarsa 7 hari, tapi API menerbitkan ulang)
2. Train: `DATASET_URL='<url>' MODEL=<base> HOURS=<jam> HF_REPO=yosa722/bench-<slug> bash examples/run_instruct.sh`
3. Eval:  `python -m ops.validator_ops.run_evaluation --task_id <id> --models yosa722/bench-<slug> <repo_winner> --gpu_ids 0`

## Tournament 2026-08-24 (tourn_e758aac2d861c378)

| Task | Base model | Sifat | Winner (loss) | Skor kita saat itu | Repo winner |
|---|---|---|---|---|---|
| `28c9f3bb-81f7-47b6-9468-4918b98904ed` | unsloth/Llama-3.2-1B | Turki, 50k, pendek | 1.7681 | 2.2216 → kalibrasi 1.8506 | `...20260824-318ef829-...-5C7vE26G` |
| `3a6658eb-128a-4d7d-9efc-a3eee139b985` | unsloth/Qwen2-1.5B-Instruct | clinical notes, DOKUMEN PANJANG (p50≈2000 tok) | 0.1111 | 0.2541 → kalibrasi 0.1182 | `...20260824-f820e6ca-...-5FW2Eaae` |
| `10b5cdcc-7cb9-4e33-ac17-79876ddd62f3` | Qwen/Qwen3-4B-Base | Bengali, 4B | 0.5966 | 0.6747 | `...20260824-f039c9ca-...-5GpcTKW7` |

## Tournament 2026-08-17 (tourn_03a3ba3f5bb25c4a)

| Task | Base model | Sifat | Winner (loss) | Catatan |
|---|---|---|---|---|
| `05266233-5bcc-4430-b657-f244d5b30d88` | microsoft/phi-1_5 | MathFusionQA | **0.4484 = KITA (menang)** | benchmark regresi: jangan sampai memburuk |
| `45f9a179-1e8e-4754-b2af-67888f94cdc9` | LiquidAI/LFM2.5-2.6B | newsqa, arsitektur baru | 5.3101 (nyaris base) | kita kini bisa train penuh — peluang unggul besar |

## Cakupan arsitektur

Llama (1B) · Qwen2 (1.5B) · Qwen3 (4B) · Phi (1.4B) · LFM2 (2.6B) — 5 keluarga.
Tambahkan task baru setiap Senin selesai (JSON hasil → baris baru di sini).

## Temuan terkalibrasi (jangan diulang eksperimennya)

- 2026-08-28: LR safety 2.0→1.4, blend 0.5→0.7 → 1.8563→1.8506 (T1). Dibakukan.
- 2026-08-28: pre-tokenize p99 (data panjang tidak dibuang) → 0.2541→0.1182 (T2). Efek terbesar sejauh ini.
- Kandidat berikut: Prompt Loss Weight (T1/T2 prompt-dominan), MAD outlier filter, near-dedup split.

## Tournament 31 Agu 2026 (tourn_add1dc83b8fd58b0) — Group Stage, rank 10/11
| Task | Model | Dataset | Jam | Loss kita | Rank | Winner | Kluster tengah |
|---|---|---|---|---|---|---|---|
| T1 d99ce208 | tiiuae/falcon-rw-1b | lighteval/summarization | 1.25 | 3.0088 | 10/11 | 1.4721 | 2.2–2.6 |
| T2 bc347f11 | Qwen/Qwen2.5-3B-Instruct | hivaze/ru-AAQG-QA-QG | 1.5 | 0.5883 | 10/11 | 0.5277 | 0.545–0.575 |
| T3 597dc05e | bigscience/bloomz-560m | AlekseyKorshuk/evol-codealpaca-v1-dpo | 1.25 | 1.4304 | 6/11 | 1.3348 | 1.41–1.49 |
Catatan: 0 DNF (pertama kalinya). T1 anomali (di bawah kluster) — hipotesis: overhead pipeline memakan waktu di task pendek + falcon FA-off lambat. T3 mid-pack = jalur sehat.
URL train/test data: lihat JSON task (valid s/d ~7 Sep).

### Temuan kalibrasi T1 (2 Sep): repro lokal dev-loss 1.577 ≈ papan atas (1.47–1.55)
Akar skor 3.0088 di tournament: falcon-rw = eager attention (FA-off) + E1 mematikan
grad-ckpt → memori atensi kuadratik meledak → OOM kaskade bs12→6→3, waktu/attempt habis,
tersubmit model level-base. Fix: E1 kini syarat attn != eager (train_instruct.py).
Dengan bs3 + 1 epoch penuh: best 1.5771 @ step1000, soup & EMA benar menolak, final_dev jalan.

### Verifikasi fix eager (2 Sep, eval resmi test set T1)
yosa722/bench-t1-falcon = **1.5353** → setara rank 3/11 (winner 1.4721, rank2 1.5187,
rank3 1.5479; skor kita saat tournament 3.0088). Fix E1-eager terbukti mengubah
rank 10 → rank 3 di T1. Sisa gap ke winner 4.3% = wilayah kualitas data (MAD/near-dedup).
Evaluator lokal mereproduksi angka winner 4 desimal → harness valid.

### Fitur baru (2 Sep): near-dedup SimHash + MAD gate (distinct dari winner)
- simhash_near_dedup: 64-bit SimHash atas shingle 4-gram token, pencarian radius
  Hamming r=6 via pigeonhole 7 pita — BUKAN MinHash/LSH (keluarga metode berbeda).
- mad_stat_filter: median±4×MAD pada log-panjang-completion + deteksi completion
  degeneratif (1 token >60%). Tanpa forward pass.
- Kill-switch: NEAR_DEDUP=0 / QF_MAD=0; radius: NEAR_DEDUP_R.
- Uji sintetis: 44/50 near-dup pendek tertangkap, 0 false positive pada 3000 unik.
- Benchmark: T1 falcon 1.5353 (tanpa filter) → target ≤1.50 dengan filter.
- 2026-09-02 (qf): T1 falcon — filter penuh 1.5432 vs baseline 1.5353 → gerbang
  panjang MAD merugikan (buang 330 dok terpanjang yang sah). Dibakukan: len-gate
  default OFF (QF_MAD_LEN=1 utk opt-in); near-dedup + deteksi degeneratif tetap ON
  (netral di data bersih, potensi besar di dataset kotor).
- 2026-09-03 (T2 qwen3b): repro dgn near-dedup → 0.5695 vs Senin 0.5883 (−3.2%,
  rank 10→~7). Dataset ru-AAQG: 4% exact-dup + 28.2% NEAR-DUP (16k sampel) —
  bukti pertama near-dedup SimHash bekerja di data kotor. Winner 0.5277 (gap 7.9%):
  duo teratas (0.5277/0.5281) terpisah jauh dari kluster — punya trik ekstra di
  task QA-generatif; kandidat penyelidikan minggu depan.
- 2026-09-03 (implementasi): (a) FINAL_DEV_RESERVE=240s di epoch planner —
  final_dev tak boleh lagi tergusur; (b) QF_LOSS=1 filter loss berbasis model
  (compute_losses ditulis ulang: padding + batch dinamis budget token by vocab,
  sentinel -1 utk OOM; IQR k=2.5 via QF_LOSS_K). Default QF_LOSS MASIH 0 —
  kalibrasi dulu di benchmark T2 (0.5695), menang → flip default sebelum Senin.
- 2026-09-03 (T2): FINAL_DEV_RESERVE tervalidasi sendiri: 0.5695 → 0.5654 (−0.7%)
  — QF_LOSS ternyata TIDAK sampai container (passthrough env kurang; diperbaiki
  di run_instruct.sh). Perbaikan murni dari final_dev yang kini selalu jalan.
  QF_LOSS belum teruji → run bench-t2-qwen3b-qfloss2. Baseline baru T2: 0.5654.
- 2026-09-03 (KECELAKAAN BERHARGA): env kosong ("") → int("")/float("") meledak →
  run qfloss2 tanpa filter/split/epoch-plan → 0.5311 (!) vs 0.5654 pipeline penuh.
  Resep tak sengaja: data penuh 59k + LR KONSTAN tanpa decay + soup avg (menang
  0.5827<0.6292). Hipotesis: ini trik duo teratas (pola WSD/SWA). Semua env
  di-harden (`or` default); saklar NO_DECAY=1 ditambahkan utk uji sengaja.
  Baseline T2 terbaik: 0.5311 (konfigurasi belum resmi — perlu reproduksi sengaja).

### MATRIKS T2 LENGKAP (4 Sep) — MENGALAHKAN WINNER
| dedup | decay | loss |
|---|---|---|
| on  | on  | 0.5654 |
| on  | off | 0.5720 |
| off | on  | **0.5227** ← menang vs winner 0.5278 |
| off | off | **0.5225** ← menang vs winner 0.5278 |
Kesimpulan: (1) near-dedup di data sintetis MERUGIKAN — "near-dup" = data sah
se-distribusi test. (2) decay vs no-decay = noise; decay+epoch-plan tetap default.
DIBAKUKAN: NEAR_DEDUP_CAP=0.10 — deteksi >10% → tidak membuang apa pun (T2),
≤10% → buang normal (T1). NO_DECAY tetap env eksperimen, bukan default.
QF_LOSS masih belum teruji murni (run qfloss2 ternyata kecelakaan env).
