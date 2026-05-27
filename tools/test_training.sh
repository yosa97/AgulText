#!/bin/bash
# =============================================================================
#  AgulText — GPU Training Test
#  Usage  : bash tools/test_training.sh [instruct|dpo|grpo]
#  Default: instruct
#
#  Jalankan dari root repo setelah git clone:
#    git clone https://github.com/yosa97/AgulText
#    cd AgulText
#    bash tools/test_training.sh instruct
# =============================================================================

set -e

TASK_TYPE="${1:-instruct}"
TASK_ID="test_$(date +%s)"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
HOURS=0.5
REPO_NAME="test-output"
IMAGE_NAME="agultext-test:latest"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "  AgulText GPU Training Test"
echo "  Task type : $TASK_TYPE"
echo "  Model     : $MODEL"
echo "  Task ID   : $TASK_ID"
echo "╚══════════════════════════════════════════╝"
echo ""

# Validasi task type
if [[ "$TASK_TYPE" != "instruct" && "$TASK_TYPE" != "dpo" && "$TASK_TYPE" != "grpo" ]]; then
    echo "ERROR: Task type tidak valid. Pilih: instruct | dpo | grpo"
    exit 1
fi

# Cek Docker dan GPU tersedia
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker tidak ditemukan. Install dulu:"
    echo "  curl -fsSL https://get.docker.com | sh"
    exit 1
fi

if ! docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then
    echo "WARNING: nvidia runtime tidak terdeteksi. Pastikan nvidia-container-toolkit terinstall."
    echo "  Lanjut dengan --gpus all anyway..."
fi

# Build Docker image jika belum ada
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo ">>> Building Docker image (ini akan lama pertama kali ~10-30 menit)..."
    docker build -f dockerfiles/standalone-text-trainer.dockerfile -t "$IMAGE_NAME" .
    echo ">>> Image built: $IMAGE_NAME"
else
    echo ">>> Docker image sudah ada: $IMAGE_NAME"
fi

# Buat direktori cache & checkpoint di host
mkdir -p /tmp/agultext_test/cache/models
mkdir -p /tmp/agultext_test/cache/datasets
mkdir -p /tmp/agultext_test/cache/wandb_logs
mkdir -p /tmp/agultext_test/checkpoints

DATASET_PATH="/cache/datasets/${TASK_ID}_train_data.json"

# ── Generate dataset sesuai task type ─────────────────────────────────────────
if [ "$TASK_TYPE" = "instruct" ]; then
    TASK_TYPE_ARG="InstructTextTask"
    DATASET_TYPE='{"field_instruction":"instruction","field_output":"output","no_input_format":"{instruction}","format":"{instruction}"}'
    DATASET_JSON='[
{"instruction":"Apa ibu kota Indonesia?","output":"Ibu kota Indonesia saat ini adalah Nusantara di Kalimantan Timur, menggantikan Jakarta."},
{"instruction":"Jelaskan apa itu machine learning.","output":"Machine learning adalah cabang AI yang memungkinkan sistem belajar dari data secara otomatis tanpa diprogram secara eksplisit."},
{"instruction":"Berapa hasil 15 dikali 7?","output":"Hasil 15 dikali 7 adalah 105."},
{"instruction":"Apa perbedaan Python dan JavaScript?","output":"Python digunakan untuk data science dan backend, JavaScript untuk pengembangan web frontend dan backend (Node.js)."},
{"instruction":"Tulis kode Python untuk mencetak 1 sampai 10.","output":"for i in range(1, 11):\n    print(i)"},
{"instruction":"Apa itu neural network?","output":"Neural network adalah model komputasi berlapis yang terinspirasi dari jaringan saraf otak manusia untuk memproses informasi kompleks."},
{"instruction":"Sebutkan 3 planet terbesar di tata surya.","output":"Tiga planet terbesar di tata surya adalah Jupiter, Saturnus, dan Uranus."},
{"instruction":"Jelaskan konsep overfitting.","output":"Overfitting terjadi ketika model terlalu menyesuaikan data training sehingga kehilangan kemampuan generalisasi pada data baru."},
{"instruction":"Apa itu gradient descent?","output":"Gradient descent adalah algoritma optimasi yang mengupdate parameter model ke arah negatif gradien untuk meminimalkan fungsi loss."},
{"instruction":"Bagaimana cara membuat list di Python?","output":"List di Python dibuat dengan kurung siku: my_list = [1, 2, 3] atau list() untuk list kosong."},
{"instruction":"Apa itu transformer dalam deep learning?","output":"Transformer adalah arsitektur deep learning berbasis self-attention yang menjadi fondasi model bahasa modern seperti GPT dan BERT."},
{"instruction":"Apa itu fine-tuning LLM?","output":"Fine-tuning adalah proses melatih ulang model pre-trained dengan dataset spesifik untuk mengadaptasinya pada tugas tertentu."},
{"instruction":"Apa itu LoRA?","output":"LoRA (Low-Rank Adaptation) adalah teknik fine-tuning efisien yang menambahkan matriks berdimensi rendah tanpa mengubah bobot asli model."},
{"instruction":"Apa fungsi dropout?","output":"Dropout menonaktifkan neuron secara acak saat training untuk mencegah overfitting dan meningkatkan generalisasi."},
{"instruction":"Apa itu attention mechanism?","output":"Attention mechanism memungkinkan model fokus pada token yang relevan saat menghasilkan output dengan menghitung bobot kepentingan setiap token."},
{"instruction":"Bedanya supervised vs unsupervised learning?","output":"Supervised learning pakai data berlabel, unsupervised learning mencari pola dari data tanpa label."},
{"instruction":"Apa itu batch normalization?","output":"Batch normalization menormalisasi output setiap layer selama training untuk mempercepat konvergensi dan meningkatkan stabilitas."},
{"instruction":"Apa itu tokenizer dalam NLP?","output":"Tokenizer mengubah teks mentah menjadi token yang dapat diproses model bahasa, misalnya kata atau sub-kata."},
{"instruction":"Apa bedanya CPU dan GPU untuk training?","output":"GPU jauh lebih cepat untuk training karena ribuan core-nya memproses operasi matriks secara paralel, berbeda dengan CPU yang hanya beberapa core."},
{"instruction":"Apa itu residual connection?","output":"Residual connection atau skip connection menghubungkan input layer langsung ke output-nya untuk mencegah vanishing gradient pada jaringan yang dalam."}
]'

elif [ "$TASK_TYPE" = "dpo" ]; then
    TASK_TYPE_ARG="DpoTask"
    DATASET_TYPE='{"field_prompt":"prompt","field_chosen":"chosen","field_rejected":"rejected","prompt_format":"{prompt}","chosen_format":"{chosen}","rejected_format":"{rejected}"}'
    DATASET_JSON='[
{"prompt":"Apa ibu kota Indonesia?","chosen":"Ibu kota Indonesia saat ini adalah Nusantara di Kalimantan Timur, menggantikan Jakarta.","rejected":"Saya tidak tahu ibu kota Indonesia."},
{"prompt":"Jelaskan machine learning.","chosen":"Machine learning adalah cabang AI yang memungkinkan sistem belajar dari data secara otomatis.","rejected":"Machine learning itu susah dipahami."},
{"prompt":"Berapa 15 dikali 7?","chosen":"15 dikali 7 adalah 105.","rejected":"Sekitar seratus lebih."},
{"prompt":"Apa itu neural network?","chosen":"Neural network adalah model komputasi berlapis terinspirasi otak manusia untuk memproses informasi kompleks.","rejected":"Neural network adalah jaringan internet."},
{"prompt":"Apa itu overfitting?","chosen":"Overfitting terjadi ketika model terlalu sesuai data training sehingga buruk saat diprediksi pada data baru.","rejected":"Overfitting adalah kesalahan dalam coding."},
{"prompt":"Apa itu LoRA?","chosen":"LoRA adalah teknik fine-tuning efisien dengan matriks berdimensi rendah tanpa mengubah bobot asli model.","rejected":"LoRA adalah singkatan yang tidak jelas."},
{"prompt":"Apa itu transformer?","chosen":"Transformer adalah arsitektur deep learning berbasis self-attention yang menjadi fondasi model seperti GPT.","rejected":"Transformer adalah alat elektronik pengubah tegangan."},
{"prompt":"Apa fungsi dropout?","chosen":"Dropout menonaktifkan neuron secara acak saat training untuk mencegah overfitting dan meningkatkan generalisasi.","rejected":"Dropout menghapus data dari dataset training."},
{"prompt":"Apa itu gradient descent?","chosen":"Gradient descent mengupdate parameter model ke arah negatif gradien untuk meminimalkan fungsi loss.","rejected":"Gradient descent adalah cara menghitung error saja."},
{"prompt":"Bedanya supervised vs unsupervised learning?","chosen":"Supervised learning pakai data berlabel, unsupervised learning mencari pola dari data tanpa label.","rejected":"Keduanya hampir sama, tidak ada perbedaan signifikan."}
]'

elif [ "$TASK_TYPE" = "grpo" ]; then
    TASK_TYPE_ARG="GrpoTask"
    DATASET_TYPE='{"field_prompt":"prompt","reward_functions":[{"reward_func":"def reward_func(completions, **kwargs):\n    import re\n    rewards = []\n    for c in completions:\n        nums = re.findall(r\"\\\\d+\", c)\n        rewards.append(1.0 if nums else 0.0)\n    return rewards","reward_weight":1.0,"func_hash":"test_hash","is_generic":false}]}'
    DATASET_JSON='[
{"prompt":"Berapa 15 dikali 7? Jawab dengan angka saja."},
{"prompt":"Berapa 8 ditambah 13? Jawab dengan angka saja."},
{"prompt":"Berapa 100 dibagi 4? Jawab dengan angka saja."},
{"prompt":"Berapa 9 pangkat 2? Jawab dengan angka saja."},
{"prompt":"Berapa 50 dikurangi 17? Jawab dengan angka saja."},
{"prompt":"Berapa 6 dikali 8? Jawab dengan angka saja."},
{"prompt":"Berapa 144 dibagi 12? Jawab dengan angka saja."},
{"prompt":"Berapa 7 pangkat 2? Jawab dengan angka saja."},
{"prompt":"Berapa 25 ditambah 38? Jawab dengan angka saja."},
{"prompt":"Berapa 1000 dibagi 8? Jawab dengan angka saja."}
]'
fi

# Tulis dataset ke host cache, akan di-mount ke container
echo "$DATASET_JSON" > "/tmp/agultext_test/cache/datasets/${TASK_ID}_train_data.json"
echo ">>> Dataset ditulis: /tmp/agultext_test/cache/datasets/${TASK_ID}_train_data.json"

# ── Jalankan container ─────────────────────────────────────────────────────────
echo ""
echo ">>> Menjalankan Docker container..."
echo ">>> Task type : $TASK_TYPE_ARG"
echo ">>> Logs akan muncul di bawah ini..."
echo "───────────────────────────────────────────"

docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v /tmp/agultext_test/cache:/cache \
    -v /tmp/agultext_test/checkpoints:/app/checkpoints \
    -e WANDB_MODE=offline \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    "$IMAGE_NAME" \
    bash -c "
        set -e
        redis-server --daemonize yes && sleep 2

        # Download model jika belum ada
        MODEL_DIR='/cache/models/$(echo $MODEL | tr '/' '--')'
        if [ ! -d \"\$MODEL_DIR\" ]; then
            echo '>>> Downloading model $MODEL ...'
            python -c \"
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='\$MODEL_DIR', ignore_patterns=['*.gguf'])
print('Model downloaded.')
\"
        else
            echo '>>> Model sudah ada di '\$MODEL_DIR
        fi

        cd /workspace/scripts
        python -m text_trainer \
            --task-id '$TASK_ID' \
            --model '$MODEL' \
            --dataset '$DATASET_PATH' \
            --dataset-type '$DATASET_TYPE' \
            --task-type '$TASK_TYPE_ARG' \
            --file-format json \
            --hours-to-complete $HOURS \
            --expected-repo-name '$REPO_NAME'
    "

echo ""
echo "╔══════════════════════════════════════════╗"
echo "  Training SELESAI!"
echo "  Output: /tmp/agultext_test/checkpoints/$TASK_ID/$REPO_NAME"
echo "╚══════════════════════════════════════════╝"
