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

set -e

TASK_ID="test_dpo_$(date +%s)"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-}"
HOURS="${HOURS:-0.5}"
REPO_NAME="dpo-test-output"
IMAGE_NAME="agultext:latest"
CACHE_DIR="/tmp/agultext_cache"

echo "╔══════════════════════════════════════════╗"
echo "  AgulText — DPO Training Test"
echo "  Model  : $MODEL"
echo "  Task ID: $TASK_ID"
echo "╚══════════════════════════════════════════╝"

mkdir -p "$CACHE_DIR/models" "$CACHE_DIR/datasets" "$CACHE_DIR/wandb_logs"
mkdir -p "$CACHE_DIR/checkpoints"

DATASET_PATH="$CACHE_DIR/datasets/${TASK_ID}_train_data.json"
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

echo ">>> Dataset DPO ditulis: $DATASET_PATH (15 contoh)"

DATASET_TYPE='{"field_prompt":"prompt","field_chosen":"chosen","field_rejected":"rejected","prompt_format":"{prompt}","chosen_format":"{chosen}","rejected_format":"{rejected}"}'

if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo ">>> Building Docker image (pertama kali ~20-30 menit)..."
    docker build -f dockerfiles/standalone-text-trainer.dockerfile -t "$IMAGE_NAME" .
fi

echo ">>> Menjalankan container training DPO..."
docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v "$CACHE_DIR:/cache" \
    -v "$CACHE_DIR/checkpoints:/app/checkpoints" \
    -e WANDB_MODE=offline \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    "$IMAGE_NAME" \
    bash -c "
        redis-server --daemonize yes && sleep 2
        MODEL_DIR='/cache/models/$(echo $MODEL | tr '/' '--')'
        if [ ! -d \"\$MODEL_DIR\" ]; then
            python -c \"from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='\$MODEL_DIR', ignore_patterns=['*.gguf']); print('Model downloaded.')\"
        fi
        cd /workspace/scripts
        python -m text_trainer \
            --task-id '$TASK_ID' \
            --model '$MODEL' \
            --dataset '/cache/datasets/${TASK_ID}_train_data.json' \
            --dataset-type '$DATASET_TYPE' \
            --task-type DpoTask \
            --file-format json \
            --hours-to-complete $HOURS \
            --expected-repo-name '$REPO_NAME'
    "

echo ""
echo "✓ Training selesai → $CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME"

if [ -n "$HF_REPO" ]; then
    echo ""
    echo ">>> Mengupload ke HuggingFace: $HF_REPO ..."
    HF_TOKEN="$HF_TOKEN" bash examples/upload_to_hf.sh \
        "$CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME" \
        "$HF_REPO"
else
    echo ""
    echo "TIP: Upload ke HF:"
    echo "  bash examples/upload_to_hf.sh $CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME yosa97/nama-repo"
fi
