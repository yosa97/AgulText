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
