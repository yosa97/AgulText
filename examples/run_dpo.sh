#!/bin/bash
# =============================================================================
#  AgulText — Test DPO Training + Upload ke HuggingFace
#  Usage: bash examples/run_dpo.sh
#
#  Jalankan dari root repo:
#    git clone https://github.com/yosa97/AgulText
#    cd AgulText
#    bash examples/run_dpo.sh
#
#  Dengan upload HF otomatis:
#    HF_TOKEN=hf_xxx HF_REPO=yosa97/test-dpo bash examples/run_dpo.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TASK_ID="test_dpo_$(date +%s)"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-}"
HOURS="${HOURS:-0.1}"
REPO_NAME="dpo-test-output"
IMAGE_NAME="agultext:latest"
CACHE_DIR="/ephemeral/agultext_cache"

echo "╔══════════════════════════════════════════╗"
echo "  AgulText — DPO Training Test"
echo "  Model  : $MODEL"
echo "  Task ID: $TASK_ID"
echo "╚══════════════════════════════════════════╝"

mkdir -p "$CACHE_DIR/models" "$CACHE_DIR/datasets" "$CACHE_DIR/wandb_logs"
mkdir -p "$CACHE_DIR/checkpoints"

DATASET_PATH="$CACHE_DIR/datasets/${TASK_ID}_train_data.json"

echo ">>> Mengunduh dataset DPO dari Stanford Alpaca (~500 preference pairs panjang)..."
python3 << PYEOF
import json, urllib.request, sys

URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
DATASET_PATH = "$DATASET_PATH"
N_TARGET = 500
# Filter output >= 500 karakter agar chosen response punya konten substansial
MIN_OUT_CHARS = 500

# Rejected responses: sengaja vague/tidak membantu untuk sinyal DPO yang jelas
BAD_RESPONSES = [
    "Saya tidak yakin dengan hal itu.",
    "Ini terlalu kompleks untuk dijelaskan singkat.",
    "Coba cari informasinya di internet.",
    "Saya tidak memiliki informasi yang cukup.",
    "Jawabannya tergantung situasi masing-masing.",
    "Tidak bisa memberikan jawaban yang pasti.",
    "Pertanyaan ini sebaiknya ditanyakan ke ahlinya.",
    "Saya kurang memahami topik ini.",
    "Susah dijelaskan dalam kata-kata sederhana.",
    "Mungkin ada, mungkin tidak, tergantung kasusnya.",
]

try:
    req = urllib.request.Request(URL, headers={"User-Agent": "python/3"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = json.loads(r.read().decode("utf-8"))

    samples = []
    for i, item in enumerate(raw):
        instr = item.get("instruction", "").strip()
        inp   = item.get("input", "").strip()
        out   = item.get("output", "").strip()
        if not instr or not out or len(out) < MIN_OUT_CHARS:
            continue
        prompt   = f"{instr}\n\n{inp}" if inp else instr
        rejected = BAD_RESPONSES[i % len(BAD_RESPONSES)]
        samples.append({"prompt": prompt, "chosen": out, "rejected": rejected})
        if len(samples) >= N_TARGET:
            break

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    print(f"Download berhasil: {len(samples)} preference pairs dari Stanford Alpaca")
    sys.exit(0)

except Exception as e:
    print(f"Download gagal: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
DOWNLOAD_ALPACA=$?

if [ $DOWNLOAD_ALPACA -ne 0 ]; then
    echo ">>> PERINGATAN: Download gagal. Menggunakan dataset inline (15 entries, tanpa repetisi)."
    cat > "$DATASET_PATH" << 'EOF'
[
  {"prompt": "Apa ibu kota Indonesia?", "chosen": "Ibu kota Indonesia saat ini adalah Nusantara di Kalimantan Timur, menggantikan Jakarta.", "rejected": "Saya tidak tahu ibu kota Indonesia."},
  {"prompt": "Jelaskan machine learning.", "chosen": "Machine learning adalah cabang AI yang memungkinkan sistem belajar dari data secara otomatis.", "rejected": "Machine learning itu susah dipahami."},
  {"prompt": "Berapa 15 dikali 7?", "chosen": "15 dikali 7 adalah 105.", "rejected": "Sekitar seratus lebih."},
  {"prompt": "Apa itu neural network?", "chosen": "Neural network adalah model komputasi berlapis terinspirasi otak manusia untuk memproses informasi kompleks.", "rejected": "Neural network adalah jaringan internet."},
  {"prompt": "Apa itu overfitting?", "chosen": "Overfitting terjadi ketika model terlalu sesuai data training sehingga buruk saat diprediksi pada data baru.", "rejected": "Overfitting adalah kesalahan dalam coding."},
  {"prompt": "Apa itu LoRA?", "chosen": "LoRA adalah teknik fine-tuning efisien dengan matriks berdimensi rendah tanpa mengubah bobot asli model.", "rejected": "LoRA adalah singkatan yang tidak jelas."},
  {"prompt": "Apa itu transformer?", "chosen": "Transformer adalah arsitektur deep learning berbasis self-attention yang menjadi fondasi model seperti GPT.", "rejected": "Transformer adalah alat elektronik pengubah tegangan."},
  {"prompt": "Apa fungsi dropout?", "chosen": "Dropout menonaktifkan neuron secara acak saat training untuk mencegah overfitting dan meningkatkan generalisasi.", "rejected": "Dropout menghapus data dari dataset training."},
  {"prompt": "Apa itu gradient descent?", "chosen": "Gradient descent mengupdate parameter model ke arah negatif gradien untuk meminimalkan fungsi loss.", "rejected": "Gradient descent adalah cara menghitung error saja."},
  {"prompt": "Bedanya supervised vs unsupervised learning?", "chosen": "Supervised learning pakai data berlabel, unsupervised learning mencari pola dari data tanpa label.", "rejected": "Keduanya hampir sama, tidak ada perbedaan signifikan."},
  {"prompt": "Apa itu attention mechanism?", "chosen": "Attention mechanism memungkinkan model fokus pada token relevan dengan menghitung bobot kepentingan tiap token.", "rejected": "Attention mechanism adalah cara model memperhatikan input."},
  {"prompt": "Apa itu fine-tuning?", "chosen": "Fine-tuning adalah proses melatih ulang model pre-trained dengan dataset spesifik untuk tugas tertentu.", "rejected": "Fine-tuning adalah cara membuat model baru dari awal."},
  {"prompt": "Apa itu batch normalization?", "chosen": "Batch normalization menormalisasi output setiap layer saat training untuk mempercepat konvergensi.", "rejected": "Batch normalization adalah cara mengurangi batch size."},
  {"prompt": "Apa itu residual connection?", "chosen": "Residual connection menghubungkan input layer langsung ke outputnya untuk mencegah vanishing gradient.", "rejected": "Residual connection adalah koneksi yang sisa."},
  {"prompt": "Apa itu tokenizer?", "chosen": "Tokenizer mengubah teks mentah menjadi token yang dapat diproses model bahasa, misalnya kata atau sub-kata.", "rejected": "Tokenizer adalah program yang membaca file teks."}
]
EOF
fi

DATASET_N=$(python3 -c "import json; d=json.load(open('$DATASET_PATH')); print(len(d))")
echo ">>> Dataset siap: $DATASET_N entries unik (tidak ada repetisi)"

DATASET_TYPE='{"field_prompt":"prompt","field_chosen":"chosen","field_rejected":"rejected","prompt_format":"{prompt}","chosen_format":"{chosen}","rejected_format":"{rejected}"}'

if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo ">>> Building Docker image (pertama kali ~20-30 menit)..."
    docker build -f dockerfiles/standalone-text-trainer.dockerfile -t "$IMAGE_NAME" .
fi

MODEL_DIR_NAME="$(echo $MODEL | tr '/' '--')"

rm -rf "$CACHE_DIR/internal_datasets"
rm -rf "$CACHE_DIR/soutputs"
rm -rf "$CACHE_DIR/wandb"
mkdir -p "$CACHE_DIR/internal_datasets"
mkdir -p "$CACHE_DIR/soutputs"
mkdir -p "$CACHE_DIR/wandb"

FULL_LOG="$CACHE_DIR/dpo_run_${TASK_ID}.log"

echo ">>> Menjalankan container training DPO... (log: $FULL_LOG)"
echo ""

set +e
docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v "$CACHE_DIR:/cache" \
    -v "$CACHE_DIR/checkpoints:/app/checkpoints" \
    -v "$REPO_ROOT/scripts:/workspace/scripts" \
    -v "$CACHE_DIR/internal_datasets:/workspace/scripts/datasets" \
    -v "$CACHE_DIR/soutputs:/workspace/scripts/soutputs" \
    -v "$CACHE_DIR/wandb:/workspace/scripts/wandb" \
    -e WANDB_MODE=offline \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e TASK_ID="$TASK_ID" \
    -e MODEL="$MODEL" \
    -e MODEL_DIR_NAME="$MODEL_DIR_NAME" \
    -e DATASET_TYPE="$DATASET_TYPE" \
    -e HOURS="$HOURS" \
    -e REPO_NAME="$REPO_NAME" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e HF_REPO="$HF_REPO" \
    --entrypoint bash \
    "$IMAGE_NAME" \
    -c '
        redis-server --daemonize yes && sleep 2

        cd /workspace/scripts
        python download_model_only.py "$MODEL"

        mkdir -p /workspace/input_data
        cp "/cache/datasets/${TASK_ID}_train_data.json" /workspace/input_data/

        python -m text_trainer \
            --task-id "$TASK_ID" \
            --model "$MODEL" \
            --dataset "/cache/datasets/${TASK_ID}_train_data.json" \
            --dataset-type "$DATASET_TYPE" \
            --task-type DpoTask \
            --file-format json \
            --hours-to-complete "$HOURS" \
            --expected-repo-name "$REPO_NAME"

        if [ -n "$HF_REPO" ] && [ -n "$HF_TOKEN" ]; then
            echo ">>> Mengupload ke HuggingFace: $HF_REPO ..."
            python -c "
from huggingface_hub import HfApi, create_repo
api = HfApi(token=\"$HF_TOKEN\")
create_repo(\"$HF_REPO\", token=\"$HF_TOKEN\", private=False, exist_ok=True)
api.upload_folder(
    folder_path=\"/app/checkpoints/$TASK_ID/$REPO_NAME\",
    repo_id=\"$HF_REPO\",
    token=\"$HF_TOKEN\",
    ignore_patterns=[\"*.log\", \"optimizer.pt\", \"rng_state*.pth\"],
)
print(\"Upload selesai: https://huggingface.co/$HF_REPO\")
"
        fi
    ' 2>&1 | tee "$FULL_LOG"
DOCKER_EXIT="${PIPESTATUS[0]}"
set -e

SUBMIT_DIR="$CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  VERIFIKASI HASIL TEST DPO"
echo "════════════════════════════════════════════════════════"

PASS=0
FAIL=0

_check() {
    local label="$1"
    local result="$2"
    local note="$3"
    if [ "$result" = "ok" ]; then
        echo "  ✓  $label"
        PASS=$((PASS+1))
    else
        echo "  ✗  $label  ← $note"
        FAIL=$((FAIL+1))
    fi
}

# 1. Container selesai tanpa crash
[ "$DOCKER_EXIT" -eq 0 ] \
    && _check "Container selesai normal (exit 0)" "ok" "" \
    || _check "Container selesai normal (exit 0)" "fail" "exit=$DOCKER_EXIT"

# 2. Training masuk success path
grep -q "Training successfully done" "$FULL_LOG" \
    && _check "Training success path" "ok" "" \
    || _check "Training success path" "fail" "tidak ada 'Training successfully done'"

# 3. loss.txt ada di submission
[ -f "$SUBMIT_DIR/loss.txt" ] \
    && _check "loss.txt ada (eval loss tercatat)" "ok" "" \
    || _check "loss.txt ada (eval loss tercatat)" "fail" "tidak ada loss.txt — training gagal sebelum eval"

# 4. Submission dir berisi model
FILE_COUNT=$(ls "$SUBMIT_DIR" 2>/dev/null | wc -l || echo 0)
[ "$FILE_COUNT" -ge 2 ] \
    && _check "Submission dir berisi model ($FILE_COUNT files)" "ok" "" \
    || _check "Submission dir berisi model" "fail" "hanya $FILE_COUNT file di $SUBMIT_DIR"

# 5. config.json ada
[ -f "$SUBMIT_DIR/config.json" ] \
    && _check "config.json ada di submission" "ok" "" \
    || _check "config.json ada di submission" "fail" "tidak ada config.json"

# 6. ModelSoupCallback aktif (soup terintegrasi di train_dpo.py)
grep -q "\[soup\] siap:" "$FULL_LOG" \
    && _check "ModelSoupCallback (DPO) aktif" "ok" "" \
    || _check "ModelSoupCallback (DPO) aktif" "fail" "tidak ada '[soup] siap:' — soup_callback tidak ter-load"

echo ""
echo "  Total: $PASS lulus, $FAIL gagal"
echo "════════════════════════════════════════════════════════"

echo ""
echo "File output:"
echo "  Log lengkap   : $FULL_LOG"
echo "  Submission    : $SUBMIT_DIR"

# Tampilkan eval loss dari loss.txt
LOSS_FILE="$SUBMIT_DIR/loss.txt"
if [ -f "$LOSS_FILE" ]; then
    LOSS_CONTENT=$(cat "$LOSS_FILE")
    EVAL_STEP=$(echo "$LOSS_CONTENT" | cut -d',' -f1)
    EVAL_LOSS=$(echo "$LOSS_CONTENT" | cut -d',' -f2)
    echo "  Eval loss     : $EVAL_LOSS  (best checkpoint: step $EVAL_STEP)"
    echo "  ↳ Untuk DPO: makin negatif = reward makin tinggi = makin bagus"
else
    echo "  Eval loss     : tidak tersedia (loss.txt belum ditulis)"
fi
echo ""

# Tampilkan hasil soup averaging DPO
SOUP_END=$(grep -E "\[soup\] (rata-rata (LEBIH BAIK|tidak lebih baik)|hanya [0-9]+ snapshot)" "$FULL_LOG" | tail -1 || true)
if [ -n "$SOUP_END" ]; then
    echo "Soup averaging DPO: $SOUP_END"
    echo ""
fi

FILE_COUNT=$(ls "$SUBMIT_DIR" 2>/dev/null | wc -l || echo 0)
if [ -n "$HF_REPO" ] && grep -q "Upload selesai:" "$FULL_LOG" 2>/dev/null; then
    echo "✓ Model terupload → https://huggingface.co/$HF_REPO"
elif [ -n "$HF_REPO" ] && [ "$DOCKER_EXIT" -eq 0 ]; then
    echo "✓ Training selesai → $SUBMIT_DIR"
    echo "  (Upload HF: tidak ada log upload, periksa HF_TOKEN jika diperlukan)"
elif [ -z "$HF_REPO" ] && [ "$FILE_COUNT" -ge 2 ]; then
    echo "✓ Training selesai → $SUBMIT_DIR"
    echo "TIP: Upload ke HF: HF_TOKEN=hf_xxx HF_REPO=yosa97/nama-repo bash examples/run_dpo.sh"
fi

# Kembalikan exit code berdasarkan hasil verifikasi
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
