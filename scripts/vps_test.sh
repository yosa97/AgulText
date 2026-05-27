#!/bin/bash
# Test script untuk AgulText di VPS dengan GPU
# Usage: bash run_test.sh [instruct|dpo|grpo]

set -e

TASK_TYPE=${1:-instruct}
TASK_ID="test_$(date +%s)"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"   # model kecil ~500MB, ganti jika mau model lain
HOURS=0.5
REPO_NAME="test-repo"

echo "=============================="
echo "  AgulText Training Test"
echo "  Task type : $TASK_TYPE"
echo "  Model     : $MODEL"
echo "  Task ID   : $TASK_ID"
echo "=============================="

# ── 1. Buat direktori yang dibutuhkan ──────────────────────────────────────────
mkdir -p /cache/models /cache/datasets /cache/wandb_logs
mkdir -p /app/checkpoints
mkdir -p /workspace/axolotl/{data,data_prepared,configs,outputs,src,input_data}
mkdir -p /workspace/scripts/datasets

# ── 2. Download model ke /cache/models ────────────────────────────────────────
MODEL_DIR="/cache/models/$(echo $MODEL | tr '/' '--')"
if [ ! -d "$MODEL_DIR" ]; then
    echo ">>> Downloading model $MODEL ..."
    HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='$MODEL_DIR', ignore_patterns=['*.gguf'])
print('Model downloaded to $MODEL_DIR')
"
else
    echo ">>> Model sudah ada di $MODEL_DIR"
fi

# ── 3. Buat dataset test ───────────────────────────────────────────────────────
DATASET_PATH="/cache/datasets/${TASK_ID}_train_data.json"

if [ "$TASK_TYPE" = "instruct" ]; then
    echo ">>> Membuat dataset instruct..."
    python -c "
import json
data = [
    {'instruction': 'Apa ibu kota Indonesia?', 'output': 'Ibu kota Indonesia adalah Jakarta (secara historis), namun ibu kota baru Indonesia adalah Nusantara di Kalimantan Timur.'},
    {'instruction': 'Jelaskan apa itu machine learning.', 'output': 'Machine learning adalah cabang kecerdasan buatan yang memungkinkan sistem belajar dari data tanpa diprogram secara eksplisit.'},
    {'instruction': 'Berapa hasil 15 dikali 7?', 'output': '15 dikali 7 adalah 105.'},
    {'instruction': 'Apa perbedaan antara Python dan JavaScript?', 'output': 'Python biasanya digunakan untuk data science dan backend, sedangkan JavaScript utamanya digunakan untuk pengembangan web frontend dan backend (Node.js).'},
    {'instruction': 'Tulis kode Python untuk mencetak angka 1 sampai 10.', 'output': 'for i in range(1, 11):\n    print(i)'},
    {'instruction': 'Apa itu neural network?', 'output': 'Neural network adalah model komputasi yang terinspirasi dari jaringan saraf biologis, terdiri dari lapisan-lapisan neuron buatan yang saling terhubung.'},
    {'instruction': 'Sebutkan 3 planet terbesar di tata surya.', 'output': 'Tiga planet terbesar di tata surya adalah Jupiter, Saturnus, dan Uranus.'},
    {'instruction': 'Jelaskan konsep overfitting dalam machine learning.', 'output': 'Overfitting terjadi ketika model terlalu menyesuaikan diri dengan data pelatihan sehingga kehilangan kemampuan generalisasi pada data baru.'},
    {'instruction': 'Apa itu gradient descent?', 'output': 'Gradient descent adalah algoritma optimasi yang digunakan untuk meminimalkan fungsi loss dengan mengupdate parameter model ke arah negatif gradien.'},
    {'instruction': 'Bagaimana cara membuat list di Python?', 'output': 'Di Python, list dibuat dengan tanda kurung siku: my_list = [1, 2, 3] atau list() untuk list kosong.'},
    {'instruction': 'Apa itu transformer dalam deep learning?', 'output': 'Transformer adalah arsitektur deep learning berbasis mekanisme attention yang sangat efektif untuk pemrosesan sekuens, menjadi fondasi model bahasa modern seperti GPT dan BERT.'},
    {'instruction': 'Jelaskan perbedaan supervised dan unsupervised learning.', 'output': 'Supervised learning menggunakan data berlabel untuk melatih model, sedangkan unsupervised learning mencari pola dalam data tanpa label.'},
    {'instruction': 'Apa fungsi dari dropout dalam neural network?', 'output': 'Dropout adalah teknik regularisasi yang secara acak menonaktifkan sebagian neuron selama pelatihan untuk mencegah overfitting.'},
    {'instruction': 'Sebutkan framework machine learning yang populer.', 'output': 'Framework machine learning populer antara lain PyTorch, TensorFlow, JAX, scikit-learn, dan Keras.'},
    {'instruction': 'Apa itu fine-tuning dalam konteks LLM?', 'output': 'Fine-tuning adalah proses melatih ulang model bahasa yang sudah pre-trained dengan dataset spesifik untuk mengadaptasinya pada tugas tertentu.'},
    {'instruction': 'Jelaskan apa itu tokenizer dalam NLP.', 'output': 'Tokenizer adalah komponen yang mengubah teks mentah menjadi token (unit terkecil) yang dapat diproses oleh model bahasa.'},
    {'instruction': 'Apa perbedaan antara CPU dan GPU untuk training?', 'output': 'GPU jauh lebih cepat untuk training karena memiliki ribuan core yang dapat memproses operasi matriks secara paralel, sementara CPU hanya memiliki beberapa core.'},
    {'instruction': 'Apa itu LoRA dalam fine-tuning?', 'output': 'LoRA (Low-Rank Adaptation) adalah teknik fine-tuning efisien yang menambahkan matriks berdimensi rendah ke layer model tanpa mengubah bobot asli, sehingga hemat memori.'},
    {'instruction': 'Bagaimana cara mencegah vanishing gradient?', 'output': 'Vanishing gradient dapat dicegah dengan teknik seperti batch normalization, residual connections, aktivasi ReLU, dan inisialisasi bobot yang tepat.'},
    {'instruction': 'Apa itu attention mechanism?', 'output': 'Attention mechanism memungkinkan model untuk fokus pada bagian-bagian input yang relevan saat menghasilkan output, dengan menghitung bobot kepentingan setiap token.'},
]
with open('$DATASET_PATH', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f'Dataset dibuat: $DATASET_PATH ({len(data)} contoh)')
"
    DATASET_TYPE='{"field_instruction":"instruction","field_output":"output","no_input_format":"{instruction}","format":"{instruction}"}'
    TASK_TYPE_ARG="InstructTextTask"

elif [ "$TASK_TYPE" = "dpo" ]; then
    echo ">>> Membuat dataset DPO..."
    python -c "
import json
data = [
    {'prompt': 'Apa ibu kota Indonesia?', 'chosen': 'Ibu kota Indonesia adalah Jakarta, namun ibu kota baru adalah Nusantara di Kalimantan Timur.', 'rejected': 'Saya tidak tahu.'},
    {'prompt': 'Jelaskan machine learning.', 'chosen': 'Machine learning adalah cabang AI yang memungkinkan sistem belajar dari data secara otomatis.', 'rejected': 'Machine learning itu susah.'},
    {'prompt': 'Berapa 15 x 7?', 'chosen': '15 dikali 7 adalah 105.', 'rejected': 'Sekitar 100.'},
    {'prompt': 'Apa itu neural network?', 'chosen': 'Neural network adalah model komputasi berlapis yang terinspirasi dari otak manusia untuk memproses informasi.', 'rejected': 'Neural network adalah jaringan komputer.'},
    {'prompt': 'Apa itu overfitting?', 'chosen': 'Overfitting terjadi ketika model terlalu sesuai dengan data training sehingga buruk pada data baru.', 'rejected': 'Overfitting adalah error yang terjadi saat training.'},
    {'prompt': 'Apa itu gradient descent?', 'chosen': 'Gradient descent adalah algoritma optimasi yang mengupdate parameter model ke arah negatif gradien untuk meminimalkan loss.', 'rejected': 'Gradient descent adalah cara menghitung gradien.'},
    {'prompt': 'Apa itu LoRA?', 'chosen': 'LoRA adalah teknik fine-tuning efisien yang menambahkan matriks berdimensi rendah tanpa mengubah bobot asli model.', 'rejected': 'LoRA adalah singkatan dari Low Rank.'},
    {'prompt': 'Apa itu transformer?', 'chosen': 'Transformer adalah arsitektur deep learning berbasis self-attention yang menjadi fondasi model bahasa modern seperti GPT.', 'rejected': 'Transformer adalah alat elektronik.'},
    {'prompt': 'Apa bedanya supervised dan unsupervised learning?', 'chosen': 'Supervised learning menggunakan data berlabel, unsupervised learning mencari pola dari data tanpa label.', 'rejected': 'Keduanya hampir sama.'},
    {'prompt': 'Apa fungsi dropout?', 'chosen': 'Dropout menonaktifkan neuron secara acak saat training untuk mencegah overfitting dan meningkatkan generalisasi model.', 'rejected': 'Dropout menghapus data dari dataset.'},
]
with open('$DATASET_PATH', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f'Dataset DPO dibuat: $DATASET_PATH ({len(data)} contoh)')
"
    DATASET_TYPE='{"field_prompt":"prompt","field_chosen":"chosen","field_rejected":"rejected","prompt_format":"{prompt}","chosen_format":"{chosen}","rejected_format":"{rejected}"}'
    TASK_TYPE_ARG="DpoTask"

elif [ "$TASK_TYPE" = "grpo" ]; then
    echo ">>> Membuat dataset GRPO..."
    python -c "
import json
data = [
    {'prompt': 'Berapa 15 dikali 7? Jawab dengan angka saja.'},
    {'prompt': 'Berapa 8 ditambah 13? Jawab dengan angka saja.'},
    {'prompt': 'Berapa 100 dibagi 4? Jawab dengan angka saja.'},
    {'prompt': 'Berapa 9 pangkat 2? Jawab dengan angka saja.'},
    {'prompt': 'Berapa 50 dikurangi 17? Jawab dengan angka saja.'},
    {'prompt': 'Berapa 6 dikali 8? Jawab dengan angka saja.'},
    {'prompt': 'Berapa 144 dibagi 12? Jawab dengan angka saja.'},
    {'prompt': 'Berapa 7 pangkat 2? Jawab dengan angka saja.'},
    {'prompt': 'Berapa 25 ditambah 38? Jawab dengan angka saja.'},
    {'prompt': 'Berapa 1000 dibagi 8? Jawab dengan angka saja.'},
]
with open('$DATASET_PATH', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f'Dataset GRPO dibuat: $DATASET_PATH ({len(data)} contoh)')
"
    REWARD_FUNC='def reward_func(completions, **kwargs):\n    import re\n    rewards = []\n    for c in completions:\n        nums = re.findall(r\"\\d+\", c)\n        rewards.append(1.0 if nums else 0.0)\n    return rewards'
    DATASET_TYPE="{\"field_prompt\":\"prompt\",\"reward_functions\":[{\"reward_func\":\"$(echo $REWARD_FUNC)\",\"reward_weight\":1.0}]}"
    TASK_TYPE_ARG="GrpoTask"
fi

# ── 4. Start Redis ─────────────────────────────────────────────────────────────
echo ">>> Starting Redis..."
redis-server --daemonize yes 2>/dev/null || true
sleep 2

# ── 5. Jalankan training ───────────────────────────────────────────────────────
echo ""
echo ">>> Menjalankan training $TASK_TYPE_ARG ..."
echo ">>> Task ID  : $TASK_ID"
echo ">>> Model    : $MODEL"
echo ">>> Dataset  : $DATASET_PATH"
echo ""

cd /workspace/scripts

python -m text_trainer \
    --task-id "$TASK_ID" \
    --model "$MODEL" \
    --dataset "$DATASET_PATH" \
    --dataset-type "$DATASET_TYPE" \
    --task-type "$TASK_TYPE_ARG" \
    --file-format json \
    --hours-to-complete "$HOURS" \
    --expected-repo-name "$REPO_NAME"

echo ""
echo "=============================="
echo "  Training selesai!"
echo "  Output: /app/checkpoints/$TASK_ID/$REPO_NAME"
echo "=============================="
