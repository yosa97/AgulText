#!/bin/bash
# =============================================================================
#  AgulText — Test GRPO Training + Upload ke HuggingFace
#  Usage: bash examples/run_grpo.sh
#
#  Jalankan dari root repo:
#    git clone https://github.com/yosa97/AgulText
#    cd AgulText
#    bash examples/run_grpo.sh
#
#  Dengan upload HF otomatis:
#    HF_TOKEN=hf_xxx HF_REPO=yosa97/test-grpo bash examples/run_grpo.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TASK_ID="test_grpo_$(date +%s)"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-}"
HOURS="${HOURS:-0.1}"
REPO_NAME="grpo-test-output"
IMAGE_NAME="agultext:latest"
CACHE_DIR="/ephemeral/agultext_cache"

echo "╔══════════════════════════════════════════╗"
echo "  AgulText — GRPO Training Test"
echo "  Model  : $MODEL"
echo "  Task ID: $TASK_ID"
echo "╚══════════════════════════════════════════╝"

mkdir -p "$CACHE_DIR/models" "$CACHE_DIR/datasets" "$CACHE_DIR/wandb_logs"
mkdir -p "$CACHE_DIR/checkpoints"

DATASET_PATH="$CACHE_DIR/datasets/${TASK_ID}_train_data.json"
# 80 math/reasoning prompts unik — tanpa repetisi.
# Reward function mendeteksi angka di output, jadi cukup gunakan soal numerik beragam.
cat > "$DATASET_PATH" << 'EOF'
[
  {"prompt": "Berapa 15 dikali 7? Jawab dengan angka saja."},
  {"prompt": "Berapa 8 ditambah 13? Jawab dengan angka saja."},
  {"prompt": "Berapa 100 dibagi 4? Jawab dengan angka saja."},
  {"prompt": "Berapa 9 pangkat 2? Jawab dengan angka saja."},
  {"prompt": "Berapa 50 dikurangi 17? Jawab dengan angka saja."},
  {"prompt": "Berapa 6 dikali 8? Jawab dengan angka saja."},
  {"prompt": "Berapa 144 dibagi 12? Jawab dengan angka saja."},
  {"prompt": "Berapa 7 pangkat 2? Jawab dengan angka saja."},
  {"prompt": "Berapa 25 ditambah 38? Jawab dengan angka saja."},
  {"prompt": "Berapa 1000 dibagi 8? Jawab dengan angka saja."},
  {"prompt": "Berapa 13 dikali 6? Jawab dengan angka saja."},
  {"prompt": "Berapa 200 dikurangi 75? Jawab dengan angka saja."},
  {"prompt": "Berapa 3 pangkat 3? Jawab dengan angka saja."},
  {"prompt": "Berapa 360 dibagi 9? Jawab dengan angka saja."},
  {"prompt": "Berapa 17 ditambah 44? Jawab dengan angka saja."},
  {"prompt": "Berapa 5 pangkat 4? Jawab dengan angka saja."},
  {"prompt": "Berapa 256 dibagi 16? Jawab dengan angka saja."},
  {"prompt": "Berapa 99 dikurangi 43? Jawab dengan angka saja."},
  {"prompt": "Berapa 11 dikali 11? Jawab dengan angka saja."},
  {"prompt": "Berapa 450 ditambah 275? Jawab dengan angka saja."},
  {"prompt": "Berapa akar kuadrat dari 144? Jawab dengan angka saja."},
  {"prompt": "Berapa akar kuadrat dari 225? Jawab dengan angka saja."},
  {"prompt": "Berapa akar kuadrat dari 64? Jawab dengan angka saja."},
  {"prompt": "Berapa akar kuadrat dari 400? Jawab dengan angka saja."},
  {"prompt": "Berapa 2 pangkat 10? Jawab dengan angka saja."},
  {"prompt": "Jika ada 5 lusin pensil, berapa total pensilnya? Jawab dengan angka saja."},
  {"prompt": "Jika harga buku Rp 15.000 dan diskon 20%, berapa harga setelah diskon? Jawab dengan angka saja."},
  {"prompt": "Sebuah persegi panjang panjangnya 12 cm dan lebarnya 8 cm. Berapa luasnya dalam cm persegi? Jawab dengan angka saja."},
  {"prompt": "Jika kereta berjalan 80 km/jam selama 3 jam, berapa jarak yang ditempuh? Jawab dengan angka saja."},
  {"prompt": "Toko menjual 150 barang per hari. Berapa barang terjual dalam 2 minggu? Jawab dengan angka saja."},
  {"prompt": "Berapa 15% dari 200? Jawab dengan angka saja."},
  {"prompt": "Berapa 25% dari 480? Jawab dengan angka saja."},
  {"prompt": "Berapa 10% dari 750? Jawab dengan angka saja."},
  {"prompt": "Berapa 30% dari 90? Jawab dengan angka saja."},
  {"prompt": "Berapa 50% dari 346? Jawab dengan angka saja."},
  {"prompt": "Jika ada 3 kotak berisi 24 apel, berapa total apel? Jawab dengan angka saja."},
  {"prompt": "Sebuah segitiga memiliki alas 10 cm dan tinggi 6 cm. Berapa luasnya? Jawab dengan angka saja."},
  {"prompt": "Berapa keliling persegi dengan sisi 7 cm? Jawab dengan angka saja."},
  {"prompt": "Jika 1 kg beras harganya Rp 12.000, berapa harga 5 kg? Jawab dengan angka saja."},
  {"prompt": "Berapa menit dalam 4 jam? Jawab dengan angka saja."},
  {"prompt": "Berapa detik dalam 1 jam? Jawab dengan angka saja."},
  {"prompt": "Berapa cm dalam 3 meter? Jawab dengan angka saja."},
  {"prompt": "Berapa mm dalam 5 cm? Jawab dengan angka saja."},
  {"prompt": "Jika suhu 20 derajat Celsius, berapa Fahrenheitnya? (F = C x 1.8 + 32). Jawab dengan angka saja."},
  {"prompt": "Berapa 72 dibagi 8 dikali 3? Jawab dengan angka saja."},
  {"prompt": "Berapa (15 + 25) dikali 2? Jawab dengan angka saja."},
  {"prompt": "Berapa 100 dikurangi 37 ditambah 12? Jawab dengan angka saja."},
  {"prompt": "Berapa (8 pangkat 2) dikurangi (6 pangkat 2)? Jawab dengan angka saja."},
  {"prompt": "Berapa bilangan prima ke-10? Jawab dengan angka saja."},
  {"prompt": "Berapa FPB dari 24 dan 36? Jawab dengan angka saja."},
  {"prompt": "Berapa KPK dari 4 dan 6? Jawab dengan angka saja."},
  {"prompt": "Berapa KPK dari 3, 4, dan 6? Jawab dengan angka saja."},
  {"prompt": "Jika sebuah lingkaran berjari-jari 7 cm, berapa kelilingnya? (pi = 22/7). Jawab dengan angka saja."},
  {"prompt": "Jika sebuah lingkaran berjari-jari 14 cm, berapa luasnya? (pi = 22/7). Jawab dengan angka saja."},
  {"prompt": "Berapa hasil dari 1000 dikurangi 1? Jawab dengan angka saja."},
  {"prompt": "Berapa 17 dikali 17? Jawab dengan angka saja."},
  {"prompt": "Berapa 999 ditambah 1? Jawab dengan angka saja."},
  {"prompt": "Berapa 500 dibagi 4? Jawab dengan angka saja."},
  {"prompt": "Sebuah persegi memiliki luas 81 cm persegi. Berapa panjang sisinya? Jawab dengan angka saja."},
  {"prompt": "Jika sebuah kotak memiliki panjang 5, lebar 4, dan tinggi 3, berapa volumenya? Jawab dengan angka saja."},
  {"prompt": "Berapa pangkat 2 dari 15? Jawab dengan angka saja."},
  {"prompt": "Berapa 1000 dibagi 25? Jawab dengan angka saja."},
  {"prompt": "Berapa 4 pangkat 5? Jawab dengan angka saja."},
  {"prompt": "Jika seorang berlari 5 km per hari selama seminggu, berapa total km? Jawab dengan angka saja."},
  {"prompt": "Berapa sisa dari 100 dibagi 7? Jawab dengan angka saja."},
  {"prompt": "Berapa sisa dari 50 dibagi 3? Jawab dengan angka saja."},
  {"prompt": "Berapa 123 ditambah 456 ditambah 789? Jawab dengan angka saja."},
  {"prompt": "Berapa 2000 dikurangi 1337? Jawab dengan angka saja."},
  {"prompt": "Jika harga TV Rp 5.000.000 dan PPN 11%, berapa total harga? Jawab dengan angka saja."},
  {"prompt": "Berapa 3/4 dari 200? Jawab dengan angka saja."},
  {"prompt": "Berapa 2/5 dari 150? Jawab dengan angka saja."},
  {"prompt": "Sebuah mobil mengisi 40 liter bensin dengan harga Rp 10.000/liter. Berapa total bayar? Jawab dengan angka saja."},
  {"prompt": "Berapa 19 dikali 19? Jawab dengan angka saja."},
  {"prompt": "Jika ada 7 baris kursi dengan 12 kursi per baris, berapa total kursi? Jawab dengan angka saja."},
  {"prompt": "Berapa 512 dibagi 8? Jawab dengan angka saja."},
  {"prompt": "Berapa 6 pangkat 3? Jawab dengan angka saja."},
  {"prompt": "Berapa hasil dari (100 dikali 5) ditambah (200 dikali 3)? Jawab dengan angka saja."}
]
EOF
DATASET_N=$(python3 -c "import json; d=json.load(open('$DATASET_PATH')); print(len(d))")
echo ">>> Dataset siap: $DATASET_N entries unik (tidak ada repetisi)"

# Reward function: beri nilai 1.0 jika jawaban mengandung angka
DATASET_TYPE='{"field_prompt":"prompt","reward_functions":[{"reward_func":"def reward_func(completions, **kwargs):\n    import re\n    return [1.0 if re.search(r\"\\\\d+\", c) else 0.0 for c in completions]","reward_weight":1.0,"func_hash":"test_grpo_hash","is_generic":false}]}'

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

echo ">>> Menjalankan container training GRPO..."
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
            --task-type GrpoTask \
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
    '

SUBMIT_DIR="$CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME"
DOCKER_EXIT=$?

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Ringkasan GRPO Training"
echo "════════════════════════════════════════════════════════"
echo ""
echo "File output:"
echo "  Submission    : $SUBMIT_DIR"

# Tampilkan eval loss dari loss.txt (ditulis trainer setelah evaluasi terbaik)
LOSS_FILE="$SUBMIT_DIR/loss.txt"
if [ -f "$LOSS_FILE" ]; then
    LOSS_CONTENT=$(cat "$LOSS_FILE")
    EVAL_STEP=$(echo "$LOSS_CONTENT" | cut -d',' -f1)
    EVAL_LOSS=$(echo "$LOSS_CONTENT" | cut -d',' -f2)
    echo "  Eval loss     : $EVAL_LOSS  (best checkpoint: step $EVAL_STEP)"
    echo "  ↳ Untuk GRPO: nilai menunjukkan reward signal (lebih tinggi = lebih baik)"
else
    echo "  Eval loss     : tidak tersedia (loss.txt belum ditulis)"
fi
echo ""

FILE_COUNT=$(ls "$SUBMIT_DIR" 2>/dev/null | wc -l || echo 0)
if [ -n "$HF_REPO" ] && [ "$DOCKER_EXIT" -eq 0 ]; then
    echo "✓ Training selesai → $SUBMIT_DIR"
    echo "  (Periksa log di atas untuk status upload HF)"
elif [ -z "$HF_REPO" ] && [ "$FILE_COUNT" -ge 2 ]; then
    echo "✓ Training selesai → $SUBMIT_DIR"
    echo "TIP: Upload ke HF: HF_TOKEN=hf_xxx HF_REPO=yosa97/nama-repo bash examples/run_grpo.sh"
fi
