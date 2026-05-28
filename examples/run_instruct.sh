#!/bin/bash
# =============================================================================
#  AgulText — Test InstructText Training + Upload ke HuggingFace
#  Usage: bash examples/run_instruct.sh
#
#  Jalankan dari root repo:
#    git clone https://github.com/yosa97/AgulText
#    cd AgulText
#    bash examples/run_instruct.sh
#
#  Dengan upload HF otomatis:
#    HF_TOKEN=hf_xxx HF_REPO=yosa97/test-instruct bash examples/run_instruct.sh
# =============================================================================

set -e

# Selalu jalankan dari root repo, tidak peduli dari mana script ini dipanggil
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TASK_ID="test_instruct_$(date +%s)"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-}"
HOURS="${HOURS:-0.1}"
REPO_NAME="instruct-test-output"
IMAGE_NAME="agultext:latest"
CACHE_DIR="/ephemeral/agultext_cache"

echo "╔══════════════════════════════════════════╗"
echo "  AgulText — InstructText Training Test"
echo "  Model  : $MODEL"
echo "  Task ID: $TASK_ID"
echo "╚══════════════════════════════════════════╝"

# Buat direktori cache di host
mkdir -p "$CACHE_DIR/models" "$CACHE_DIR/datasets" "$CACHE_DIR/wandb_logs"
mkdir -p "$CACHE_DIR/checkpoints"

# Tulis dataset
DATASET_PATH="$CACHE_DIR/datasets/${TASK_ID}_train_data.json"
cat > "$DATASET_PATH" << 'EOF'
[
  {"instruction": "Apa ibu kota Indonesia?", "output": "Ibu kota Indonesia saat ini adalah Nusantara di Kalimantan Timur, menggantikan Jakarta."},
  {"instruction": "Jelaskan apa itu machine learning.", "output": "Machine learning adalah cabang AI yang memungkinkan sistem belajar dari data secara otomatis tanpa diprogram secara eksplisit."},
  {"instruction": "Berapa hasil 15 dikali 7?", "output": "Hasil 15 dikali 7 adalah 105."},
  {"instruction": "Apa perbedaan Python dan JavaScript?", "output": "Python digunakan untuk data science dan backend, JavaScript untuk pengembangan web frontend dan backend (Node.js)."},
  {"instruction": "Tulis kode Python untuk mencetak angka 1 sampai 10.", "output": "for i in range(1, 11):\n    print(i)"},
  {"instruction": "Apa itu neural network?", "output": "Neural network adalah model komputasi berlapis yang terinspirasi dari jaringan saraf otak manusia."},
  {"instruction": "Sebutkan 3 planet terbesar di tata surya.", "output": "Tiga planet terbesar di tata surya adalah Jupiter, Saturnus, dan Uranus."},
  {"instruction": "Jelaskan konsep overfitting.", "output": "Overfitting terjadi ketika model terlalu sesuai data training sehingga kehilangan kemampuan generalisasi pada data baru."},
  {"instruction": "Apa itu gradient descent?", "output": "Gradient descent adalah algoritma optimasi yang mengupdate parameter model ke arah negatif gradien untuk meminimalkan fungsi loss."},
  {"instruction": "Apa itu LoRA?", "output": "LoRA (Low-Rank Adaptation) adalah teknik fine-tuning efisien dengan matriks berdimensi rendah tanpa mengubah bobot asli model."},
  {"instruction": "Apa itu transformer dalam deep learning?", "output": "Transformer adalah arsitektur deep learning berbasis self-attention yang menjadi fondasi model bahasa modern seperti GPT dan BERT."},
  {"instruction": "Apa itu fine-tuning LLM?", "output": "Fine-tuning adalah proses melatih ulang model pre-trained dengan dataset spesifik untuk mengadaptasinya pada tugas tertentu."},
  {"instruction": "Apa fungsi dropout dalam neural network?", "output": "Dropout menonaktifkan neuron secara acak saat training untuk mencegah overfitting dan meningkatkan generalisasi."},
  {"instruction": "Apa itu attention mechanism?", "output": "Attention mechanism memungkinkan model fokus pada token relevan saat menghasilkan output dengan menghitung bobot kepentingan tiap token."},
  {"instruction": "Bedanya supervised vs unsupervised learning?", "output": "Supervised learning pakai data berlabel, unsupervised learning mencari pola dari data tanpa label."},
  {"instruction": "Apa itu batch normalization?", "output": "Batch normalization menormalisasi output setiap layer saat training untuk mempercepat konvergensi dan meningkatkan stabilitas."},
  {"instruction": "Apa itu residual connection?", "output": "Residual connection menghubungkan input layer langsung ke outputnya untuk mencegah vanishing gradient pada jaringan yang dalam."},
  {"instruction": "Apa bedanya CPU dan GPU untuk training?", "output": "GPU jauh lebih cepat karena ribuan corenya memproses operasi matriks secara paralel, berbeda dengan CPU yang hanya beberapa core."},
  {"instruction": "Apa itu tokenizer dalam NLP?", "output": "Tokenizer mengubah teks mentah menjadi token yang dapat diproses model bahasa, misalnya kata atau sub-kata."},
  {"instruction": "Sebutkan framework machine learning populer.", "output": "Framework ML populer antara lain PyTorch, TensorFlow, JAX, scikit-learn, dan Keras."}
]
EOF

python3 -c "
import json
with open('$DATASET_PATH') as f:
    data = json.load(f)
expanded = (data * 15)[:250]
with open('$DATASET_PATH', 'w') as f:
    json.dump(expanded, f, ensure_ascii=False)
print(f'Dataset diperluas ke {len(expanded)} samples')
"

DATASET_TYPE='{"field_instruction":"instruction","field_output":"output","no_input_format":"{instruction}","format":"{instruction}"}'

# Build image jika belum ada
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo ">>> Building Docker image (pertama kali ~20-30 menit)..."
    docker build -f dockerfiles/standalone-text-trainer.dockerfile -t "$IMAGE_NAME" .
fi

MODEL_DIR_NAME="$(echo $MODEL | tr '/' '--')"

# Buat direktori internal agar log tokenisasi bisa dibaca dari host
rm -rf "$CACHE_DIR/internal_datasets"
rm -rf "$CACHE_DIR/soutputs"
rm -rf "$CACHE_DIR/wandb"
mkdir -p "$CACHE_DIR/internal_datasets"
mkdir -p "$CACHE_DIR/soutputs"
mkdir -p "$CACHE_DIR/wandb"

echo ">>> Menjalankan container training..."
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

        # Download model menggunakan script resmi G.O.D
        cd /workspace/scripts
        python download_model_only.py "$MODEL"

        mkdir -p /workspace/input_data
        cp "/cache/datasets/${TASK_ID}_train_data.json" /workspace/input_data/

        python -m text_trainer \
            --task-id "$TASK_ID" \
            --model "$MODEL" \
            --dataset "/cache/datasets/${TASK_ID}_train_data.json" \
            --dataset-type "$DATASET_TYPE" \
            --task-type InstructTextTask \
            --file-format json \
            --hours-to-complete "$HOURS" \
            --expected-repo-name "$REPO_NAME"

        # Upload ke HF dari dalam container jika HF_REPO di-set
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

echo ""
echo "✓ Training selesai → $CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME"

if [ -n "$HF_REPO" ]; then
    echo "✓ Model terupload → https://huggingface.co/$HF_REPO"
else
    echo ""
    echo "TIP: Untuk upload ke HF, jalankan:"
    echo "  HF_TOKEN=hf_xxx HF_REPO=yosa97/nama-repo bash examples/run_instruct.sh"
    echo "  atau upload manual:"
    echo "  bash examples/upload_to_hf.sh $CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME yosa97/nama-repo"
fi
