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
#
#  Fitur yang diuji dalam test ini:
#    - lr_estimator   : LR dihitung dari statistik bobot model (weight RMS sampling)
#    - seq_analyzer   : max_length adaptif dari distribusi panjang token di dataset
#    - Single-run     : training satu kali penuh, tanpa while-True LR-search loop
#    - OOM fallback   : pengurangan max_length saat batch_size=1 masih OOM
#    - config-patch   : menjaga arsitektur model di config.json submission
# =============================================================================

# Jangan pakai set -e — kita ingin tangkap dan laporan error secara eksplisit
set -uo pipefail
# Nonaktifkan exit-on-error sementara untuk bagian yang perlu tangkap exit code

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TASK_ID="test_instruct_$(date +%s)"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-}"
HOURS="${HOURS:-1.0}"          # 60 menit — cukup untuk tokenisasi axolotl + training (~5 mnt tokenisasi + 52 mnt training)
REPO_NAME="instruct-test-output"
IMAGE_NAME="agultext:latest"
CACHE_DIR="${CACHE_DIR:-/ephemeral/agultext_cache}"

echo "╔══════════════════════════════════════════════════════╗"
echo "  AgulText — InstructText Training Test"
echo "  Model    : $MODEL"
echo "  Task ID  : $TASK_ID"
echo "  Hours    : $HOURS"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Direktori cache ────────────────────────────────────────────────────────
mkdir -p "$CACHE_DIR/models" "$CACHE_DIR/datasets"
mkdir -p "$CACHE_DIR/checkpoints"

# ── Buat dataset test ──────────────────────────────────────────────────────
# Download Stanford Alpaca (52K instruksi nyata) untuk simulasi tournament realistis.
# Tidak ada repetisi — eval_loss mencerminkan kemampuan generalisasi, bukan memorisasi.
# Fallback ke dataset inline 20 entries jika tidak ada koneksi internet.
DATASET_PATH="$CACHE_DIR/datasets/${TASK_ID}_train_data.json"

echo ">>> Mengunduh dataset Stanford Alpaca (~500 entries panjang, mirip data tournament)..."
python3 << PYEOF
import json, urllib.request, sys

URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
DATASET_PATH = "$DATASET_PATH"
N_TARGET = 500
# Filter output >= 800 karakter (~200 token) agar mirip data tournament
# (tournament data rata-rata ~419 token per sample)
MIN_OUT_CHARS = 800

try:
    req = urllib.request.Request(URL, headers={"User-Agent": "python/3"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = json.loads(r.read().decode("utf-8"))

    samples = []
    for item in raw:
        instr = item.get("instruction", "").strip()
        inp   = item.get("input", "").strip()
        out   = item.get("output", "").strip()
        if not out or len(out) < MIN_OUT_CHARS:
            continue
        if inp:
            instr = f"{instr}\n\n{inp}"
        samples.append({"instruction": instr, "output": out})
        if len(samples) >= N_TARGET:
            break

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    print(f"Download berhasil: {len(samples)} samples panjang dari Stanford Alpaca (output >= {MIN_OUT_CHARS} chars)")
    sys.exit(0)

except Exception as e:
    print(f"Download gagal: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
DOWNLOAD_ALPACA=$?

if [ $DOWNLOAD_ALPACA -ne 0 ]; then
    echo ">>> PERINGATAN: Download gagal. Menggunakan dataset inline (20 entries, tanpa repetisi)."
    echo ">>> Eval loss mungkin tidak realistis karena dataset terlalu kecil."
    cat > "$DATASET_PATH" << 'EOF'
[
  {
    "instruction": "Jelaskan secara mendalam bagaimana arsitektur Transformer bekerja dalam deep learning.",
    "output": "Transformer adalah arsitektur deep learning revolusioner yang diperkenalkan oleh Google pada tahun 2017 dalam paper berjudul Attention is All You Need. Arsitektur ini sepenuhnya mengandalkan mekanisme attention dan menghilangkan ketergantungan pada struktur rekuren atau konvolusional yang sebelumnya mendominasi model NLP. Komponen utama Transformer terdiri dari encoder dan decoder. Encoder bertugas memproses input dan membangun representasi kontekstual, sementara decoder menghasilkan output berdasarkan representasi tersebut. Setiap blok encoder mengandung dua sub-lapisan utama: multi-head self-attention dan feed-forward network yang fully connected. Di antara sub-lapisan ini terdapat residual connection dan layer normalization untuk stabilitas pelatihan. Mekanisme self-attention bekerja dengan menghitung hubungan antara setiap pasangan token dalam urutan. Setiap token direpresentasikan sebagai tiga vektor yaitu Query, Key, dan Value. Skor attention dihitung dengan mengalikan Query dengan transpose Key, dibagi akar dimensi untuk normalisasi skala, kemudian melewati fungsi softmax. Hasilnya digunakan untuk menimbang vektor Value. Multi-head attention menjalankan proses ini secara paralel dengan beberapa head, memungkinkan model menangkap pola dari berbagai subspace representasi sekaligus. Karena Transformer memproses semua token secara paralel dan tidak berurutan, diperlukan positional encoding untuk menyuntikkan informasi posisi. Encoding ini biasanya menggunakan fungsi sinus dan kosinus dengan frekuensi berbeda untuk setiap dimensi. Feed-forward network dalam setiap lapisan terdiri dari dua transformasi linear dengan fungsi aktivasi non-linear seperti ReLU atau GELU di antaranya."
  },
  {
    "instruction": "Bagaimana algoritma backpropagation bekerja dalam melatih neural network?",
    "output": "Backpropagation adalah algoritma inti dalam pelatihan neural network yang memungkinkan model belajar dengan menyesuaikan bobotnya berdasarkan kesalahan prediksi. Algoritma ini merupakan aplikasi praktis dari aturan rantai diferensial dalam kalkulus, diterapkan pada komputasi grafik dari neural network. Proses pelatihan neural network terbagi dalam dua fase utama. Fase pertama adalah forward pass, di mana data input diproses melalui setiap lapisan jaringan dari input hingga output. Setiap lapisan menerapkan transformasi linear diikuti fungsi aktivasi non-linear. Hasil akhir dibandingkan dengan label target menggunakan fungsi loss seperti cross-entropy untuk klasifikasi atau mean squared error untuk regresi. Fase kedua adalah backward pass atau backpropagation. Di sini, gradien dari fungsi loss terhadap setiap parameter dihitung menggunakan aturan rantai. Gradien mengalir dari lapisan output ke lapisan input, sehingga disebut propagasi balik. Untuk setiap lapisan, kita menghitung turunan parsial loss terhadap bobot dan bias lapisan tersebut. Gradient descent kemudian menggunakan gradien ini untuk memperbarui parameter. Dalam stochastic gradient descent, pembaruan dilakukan setelah setiap sampel atau mini-batch kecil dari data. Formula pembaruannya adalah bobot baru sama dengan bobot lama dikurangi learning rate dikalikan gradien. Learning rate menentukan seberapa besar langkah pembaruan setiap iterasi. Terlalu besar menyebabkan osilasi dan divergensi, terlalu kecil mengakibatkan konvergensi lambat. Optimizers modern seperti Adam menggabungkan momentum dan adaptive learning rate untuk mengatasi tantangan ini."
  },
  {
    "instruction": "Jelaskan perbedaan antara supervised, unsupervised, dan reinforcement learning secara mendalam.",
    "output": "Machine learning dibagi menjadi tiga paradigma utama yang masing-masing memiliki karakteristik, kebutuhan data, dan aplikasi berbeda. Supervised learning adalah paradigma paling umum di mana model dilatih menggunakan dataset berlabel. Setiap contoh pelatihan terdiri dari input dan label target yang benar. Model belajar memetakan input ke output dengan meminimalkan perbedaan antara prediksinya dan label sebenarnya. Contoh aplikasinya meliputi klasifikasi email spam, deteksi objek dalam gambar, penerjemahan mesin, dan prediksi harga properti. Algoritma populer mencakup decision tree, support vector machine, random forest, dan neural network. Unsupervised learning bekerja tanpa label pada data pelatihan. Model harus menemukan struktur dan pola tersembunyi secara mandiri dalam data mentah. Clustering seperti K-means mengelompokkan data berdasarkan kemiripan. Dimensionality reduction seperti PCA dan autoencoder mengekstrak representasi lebih ringkas dari data berdimensi tinggi. Generative models seperti GAN dan VAE mempelajari distribusi data untuk menghasilkan sampel baru yang realistis. Aplikasinya meliputi segmentasi pelanggan, deteksi anomali, dan kompresi data. Reinforcement learning mengadopsi pendekatan berbeda dengan melatih agen melalui interaksi dengan lingkungan. Agen mengambil tindakan dan menerima reward atau penalti berdasarkan hasilnya. Tujuannya adalah memaksimalkan reward kumulatif jangka panjang. Tidak ada dataset statis yang diperlukan karena pengalaman dikumpulkan secara online. Aplikasi terkenal mencakup permainan seperti Go dan StarCraft, robotika, manajemen energi, dan sistem rekomendasi. Deep Q-learning dan Proximal Policy Optimization adalah algoritma reinforcement learning populer."
  },
  {
    "instruction": "Apa itu attention mechanism dan bagaimana kontribusinya dalam pemrosesan bahasa alami?",
    "output": "Attention mechanism adalah terobosan dalam deep learning yang memungkinkan model untuk secara selektif memfokuskan perhatian pada bagian tertentu dari input saat menghasilkan output. Konsep ini pertama kali diperkenalkan untuk meningkatkan performa model sequence-to-sequence dalam penerjemahan mesin, dan kemudian menjadi pondasi arsitektur Transformer. Sebelum attention, model sequence-to-sequence menggunakan encoder untuk memampatkan seluruh kalimat input menjadi satu vektor konteks tetap yang kemudian digunakan decoder. Pendekatan ini menciptakan bottleneck informasi terutama untuk kalimat panjang. Attention mengatasi masalah ini dengan memberi decoder akses ke semua hidden state encoder sekaligus. Mekanisme dasar attention bekerja dengan menghitung skor relevansi antara setiap hidden state decoder dengan semua hidden state encoder. Skor ini kemudian dinormalisasi menggunakan softmax menghasilkan bobot attention yang menjumlah menjadi satu. Weighted sum dari semua hidden state encoder menggunakan bobot ini menghasilkan vektor konteks dinamis yang berbeda untuk setiap langkah dekoding. Self-attention yang digunakan dalam Transformer menerapkan konsep yang sama tetapi di dalam satu urutan. Setiap elemen urutan memperhatikan semua elemen lain termasuk dirinya sendiri, memungkinkan model menangkap hubungan jarak jauh tanpa terbatas jarak sekuensial. Query, Key, dan Value adalah tiga representasi yang digunakan dalam self-attention. Query mewakili elemen yang sedang mencari informasi, Key mewakili elemen yang menyediakan informasi, dan Value adalah konten informasi yang akan diambil."
  },
  {
    "instruction": "Jelaskan berbagai teknik regularisasi yang digunakan untuk mencegah overfitting dalam deep learning.",
    "output": "Regularisasi adalah sekumpulan teknik yang digunakan untuk mencegah overfitting dalam model machine learning, memastikan model dapat melakukan generalisasi dengan baik pada data yang belum pernah dilihat sebelumnya. Overfitting terjadi ketika model terlalu kompleks dan menghafal pola spesifik dalam data pelatihan termasuk noise, sehingga performanya buruk pada data baru. Regularisasi L2 atau weight decay adalah teknik yang paling umum digunakan. Ia menambahkan penalti proporsional dengan kuadrat bobot model ke fungsi loss. Penalti ini mendorong bobot mendekati nol namun tidak sampai persis nol, menghasilkan model yang lebih smooth dan sederhana. Dalam konteks neural network, L2 regularisasi sering disebut weight decay karena efeknya sama dengan mengalikan bobot dengan faktor yang sedikit kurang dari satu setiap iterasi. Regularisasi L1 menambahkan penalti proporsional dengan nilai absolut bobot. Berbeda dengan L2, L1 cenderung menghasilkan solusi sparse di mana banyak bobot menjadi persis nol. Ini berguna untuk seleksi fitur implisit karena fitur yang tidak relevan mendapat bobot nol. Dropout adalah teknik regularisasi spesifik untuk neural network yang bekerja dengan secara acak menonaktifkan sebagian neuron selama pelatihan. Setiap neuron dinonaktifkan dengan probabilitas tertentu pada setiap forward pass. Ini memaksa jaringan untuk tidak bergantung pada fitur atau jalur tertentu dan mengembangkan representasi redundan dan lebih robust. Batch normalization menormalisasi aktivasi setiap lapisan untuk memiliki mean nol dan variansi satu dalam setiap mini-batch."
  },
  {
    "instruction": "Bagaimana cara kerja optimizer Adam dan mengapa ia lebih unggul dibandingkan gradient descent biasa?",
    "output": "Adam atau Adaptive Moment Estimation adalah optimizer yang paling banyak digunakan dalam pelatihan model deep learning saat ini. Diperkenalkan oleh Diederik Kingma dan Jimmy Ba pada 2014, Adam menggabungkan dua konsep kunci dari optimizer sebelumnya yaitu momentum dari SGD with Momentum dan adaptive learning rate dari RMSprop. Gradient descent standar menggunakan satu learning rate tetap untuk semua parameter. Ini sering tidak optimal karena parameter berbeda mungkin membutuhkan kecepatan pembaruan berbeda. Beberapa parameter mungkin memerlukan pembaruan besar sementara yang lain hanya perlu penyesuaian kecil. Adam mengatasi ini dengan mempertahankan dua momen bergerak untuk setiap parameter. Momen pertama adalah rata-rata bergerak eksponensial dari gradien, yang mirip dengan momentum. Momen ini memberikan arah rata-rata gradien dari waktu ke waktu dan membantu mengatasi gradien yang bising. Momen kedua adalah rata-rata bergerak eksponensial dari kuadrat gradien, mirip dengan RMSprop. Ini memberikan perkiraan variansi gradien dan digunakan untuk mengadaptasi learning rate per parameter. Parameter yang memiliki gradien bervariasi besar akan mendapat pembaruan yang lebih kecil, sementara parameter dengan gradien konsisten mendapat pembaruan yang lebih besar. Bias correction adalah fitur penting Adam yang sering diabaikan. Karena kedua momen diinisialisasi sebagai nol, estimasinya bias menuju nol pada tahap awal pelatihan terutama ketika hyperparameter decay tinggi. Adam mengoreksi ini dengan membagi setiap momen dengan faktor koreksi yang bergantung pada jumlah langkah."
  },
  {
    "instruction": "Jelaskan konsep transfer learning dan bagaimana pra-pelatihan digunakan dalam model bahasa modern.",
    "output": "Transfer learning adalah paradigma dalam machine learning di mana pengetahuan yang diperoleh dari satu tugas atau domain digunakan untuk meningkatkan pembelajaran pada tugas atau domain yang berbeda namun terkait. Pendekatan ini sangat penting dalam deep learning karena melatih model besar dari awal membutuhkan komputasi dan data yang sangat besar. Konsep dasar transfer learning berasal dari observasi bahwa fitur yang dipelajari model pada satu tugas sering berguna untuk tugas lain. Misalnya, model yang dilatih pada jutaan gambar untuk pengenalan objek telah mempelajari detektor tepi, tekstur, dan bentuk yang juga berguna untuk klasifikasi gambar medis. Dalam NLP, model yang dilatih pada teks besar telah mempelajari representasi semantik dan sintaktik bahasa yang berlaku universal. Pra-pelatihan adalah fase pertama di mana model besar dilatih pada dataset sangat besar menggunakan tugas generik. Untuk model bahasa, ini biasanya melibatkan prediksi kata berikutnya atau rekonstruksi teks yang disembunyikan. Model belajar representasi kaya dari bahasa selama fase ini tanpa memerlukan label manual. Fine-tuning adalah fase kedua di mana model yang telah dipra-latih diadaptasi untuk tugas spesifik menggunakan dataset yang biasanya jauh lebih kecil. Seluruh atau sebagian bobot model diperbarui menggunakan data berlabel tugas target. Learning rate biasanya diatur lebih kecil dari pelatihan awal untuk menghindari pelupaan katastropik di mana model kehilangan pengetahuan dari pra-pelatihan."
  },
  {
    "instruction": "Apa itu word embedding dan bagaimana cara kerjanya dalam merepresentasikan makna kata?",
    "output": "Word embedding adalah teknik representasi kata sebagai vektor bilangan real berdimensi rendah dalam ruang vektor kontinu. Representasi ini memungkinkan model machine learning untuk bekerja dengan teks secara matematis dan menangkap hubungan semantik dan sintaktik antara kata secara eksplisit dalam geometri vektor. Pendekatan tradisional seperti one-hot encoding merepresentasikan setiap kata sebagai vektor biner dengan dimensi sama dengan ukuran kosakata. Representasi ini sangat sparse, berdimensi sangat tinggi, dan tidak mengandung informasi semantik karena semua kata dianggap sama-sama berbeda satu sama lain. Word embedding mengatasi masalah ini dengan memetakan kata ke ruang berdimensi rendah di mana kata dengan makna serupa memiliki representasi yang berdekatan. Word2Vec adalah algoritma embedding seminal yang diperkenalkan Google pada 2013. Ia menggunakan dua arsitektur berbeda: Continuous Bag of Words yang memprediksi kata target dari kata-kata konteks di sekitarnya, dan Skip-gram yang melakukan sebaliknya yaitu memprediksi kata konteks dari kata target. Keduanya menggunakan jaringan neural sederhana yang dilatih pada data teks besar. Negative sampling adalah teknik efisiensi yang digunakan Word2Vec di mana selain contoh positif nyata, model juga dilatih untuk membedakan kata yang dipilih secara acak. Properti aritmatika adalah salah satu temuan paling menarik dari word embedding. Vektor representasi kata memiliki sifat di mana operasi vektor menghasilkan hasil yang masuk akal secara semantik."
  },
  {
    "instruction": "Jelaskan perbedaan mendasar antara CNN dan RNN serta kapan sebaiknya menggunakan masing-masing.",
    "output": "Convolutional Neural Network dan Recurrent Neural Network adalah dua arsitektur deep learning fundamental yang dirancang untuk jenis data berbeda dan memiliki kekuatan serta keterbatasan masing-masing. Pemahaman perbedaan keduanya penting untuk memilih arsitektur yang tepat untuk masalah tertentu. CNN dirancang khusus untuk data dengan struktur spasial seperti gambar. Komponen kunci CNN adalah lapisan konvolusional yang menggunakan filter kecil atau kernel untuk mendeteksi fitur lokal. Filter ini meluncur di seluruh input menerapkan operasi konvolusi di setiap posisi. Karena bobot filter dibagi di seluruh posisi spasial, CNN memiliki sifat ekuivarians translasi yaitu fitur yang sama dapat dideteksi di mana saja dalam gambar. Pooling layer mengurangi dimensi spasial sambil mempertahankan fitur yang paling penting. Hierarki lapisan CNN memungkinkan model membangun representasi dari fitur sederhana seperti tepi di lapisan awal hingga konsep kompleks seperti wajah di lapisan dalam. RNN dirancang untuk data sekuensial di mana urutan penting. Tidak seperti feedforward network yang memproses setiap input secara independen, RNN mempertahankan state tersembunyi yang merangkum informasi dari langkah waktu sebelumnya. Ini memungkinkan RNN memproses urutan dengan panjang sembarang dan menangkap ketergantungan temporal. LSTM adalah varian RNN yang mengatasi masalah vanishing gradient dengan mekanisme gating eksplisit yang mengontrol aliran informasi melalui waktu. GRU adalah varian yang lebih sederhana dengan kinerja serupa."
  },
  {
    "instruction": "Bagaimana model BERT bekerja dan apa yang membuatnya berbeda dari model bahasa sebelumnya?",
    "output": "BERT atau Bidirectional Encoder Representations from Transformers adalah model bahasa pra-latih revolusioner yang diperkenalkan Google pada 2018. Model ini mengubah landscape NLP dengan menunjukkan bahwa pra-pelatihan mendalam berbasis Transformer diikuti fine-tuning sederhana dapat mencapai performa terdepan pada berbagai tugas NLP tanpa arsitektur tugas-spesifik yang kompleks. Inovasi utama BERT adalah sifat bidireksionalnya. Model bahasa sebelumnya seperti GPT memproses teks dari kiri ke kanan saja. Ini membatasi setiap token hanya melihat konteks sebelumnya. BERT menggunakan arsitektur encoder Transformer yang memungkinkan setiap token memperhatikan semua token lain dalam urutan baik yang ada di kiri maupun kanan secara bersamaan, menghasilkan representasi kontekstual yang jauh lebih kaya dan informatif. BERT menggunakan dua tugas pra-pelatihan yang saling melengkapi. Tugas pertama adalah Masked Language Modeling di mana 15 persen token dalam input diganti dengan token masker khusus, dan model dilatih untuk memprediksi token asli berdasarkan konteks dua arah. Ini mendorong model membangun pemahaman kontekstual mendalam tentang bahasa. Tugas kedua adalah Next Sentence Prediction di mana model diberikan dua kalimat dan harus memprediksi apakah kalimat kedua secara logis mengikuti kalimat pertama dalam teks asli. Untuk fine-tuning, cukup menambahkan satu lapisan klasifikasi sederhana di atas representasi BERT dan melatih seluruh model pada data tugas target dengan learning rate kecil. BERT mencapai performa terdepan baru pada 11 tugas NLP saat dirilis pertama kali."
  },
  {
    "instruction": "Jelaskan teknik kuantisasi model AI dan bagaimana ia membantu deployment model besar.",
    "output": "Kuantisasi dalam konteks model AI adalah teknik kompresi model yang mengurangi presisi numerik dari bobot dan aktivasi model dari representasi floating point standar ke format yang lebih efisien berdimensi lebih rendah. Teknik ini sangat penting untuk deployment model besar pada perangkat dengan memori dan komputasi terbatas seperti smartphone, perangkat edge, dan server dengan GPU terbatas. Model neural network secara default menggunakan floating point 32-bit atau FP32 untuk menyimpan bobot dan melakukan komputasi. Setiap parameter membutuhkan 4 byte memori. Untuk model dengan miliaran parameter, total memori yang dibutuhkan menjadi sangat besar dan seringkali tidak praktis. Kuantisasi mengurangi presisi ini ke INT8 yang menggunakan 8 bit per nilai, INT4 yang menggunakan 4 bit, atau bahkan lebih rendah. INT8 mengurangi kebutuhan memori 4 kali lipat dibandingkan FP32 dan komputasi integer seringkali lebih cepat di hardware modern. Post-training quantization diterapkan setelah model selesai dilatih tanpa mengubah proses pelatihan. Model yang sudah dilatih dalam FP32 dikonversi ke presisi lebih rendah. Teknik ini sederhana namun dapat menyebabkan degradasi performa, terutama untuk model kecil atau tugas yang membutuhkan presisi tinggi. Quantization-aware training mengintegrasikan simulasi kuantisasi ke dalam proses pelatihan. Model belajar beradaptasi dengan keterbatasan presisi rendah selama pelatihan, menghasilkan model terkuantisasi yang biasanya lebih akurat dibandingkan post-training quantization."
  },
  {
    "instruction": "Apa itu RLHF dan bagaimana teknik ini digunakan untuk melatih model bahasa seperti ChatGPT?",
    "output": "Reinforcement Learning from Human Feedback atau RLHF adalah teknik pelatihan model bahasa yang menggunakan umpan balik manusia untuk menyelaraskan output model dengan preferensi dan nilai manusia. Teknik ini menjadi komponen kunci dalam melatih model seperti ChatGPT dan Claude untuk menghasilkan respons yang lebih berguna, aman, dan sesuai dengan maksud pengguna. Proses RLHF terdiri dari tiga tahap utama yang saling berkesinambungan. Tahap pertama adalah supervised fine-tuning di mana model bahasa dasar dilatih menggunakan demonstrasi berkualitas tinggi dari perilaku yang diinginkan. Data ini biasanya dikumpulkan dari manusia yang menulis respons ideal untuk berbagai prompt. Ini memberi model baseline yang baik tentang bagaimana merespons secara umum dan mendalam. Tahap kedua adalah pelatihan reward model. Manusia diberikan beberapa output berbeda dari model untuk prompt yang sama dan diminta untuk meranking kualitasnya berdasarkan preferensi. Data ranking ini digunakan untuk melatih model reward terpisah yang memprediksi skor kualitas untuk setiap output. Model reward belajar memberikan skor tinggi pada output yang lebih disukai manusia dan skor rendah pada yang kurang disukai. Tahap ketiga adalah optimisasi menggunakan Proximal Policy Optimization. Model bahasa diperlakukan sebagai policy dalam framework reinforcement learning. Ia menghasilkan respons yang kemudian dievaluasi oleh model reward. Gradien dari skor reward digunakan untuk memperbarui bobot model bahasa ke arah yang menghasilkan skor lebih tinggi."
  },
  {
    "instruction": "Jelaskan arsitektur Mixture of Experts dan keunggulannya dalam scaling model bahasa besar.",
    "output": "Mixture of Experts atau MoE adalah arsitektur neural network yang membagi komputasi antara beberapa subnetwork spesialis yang disebut expert, menggunakan jaringan gating untuk memilih subset expert yang diaktifkan untuk setiap input. Pendekatan ini memungkinkan model dengan kapasitas parameter sangat besar dijalankan secara efisien karena hanya sebagian kecil parameter yang aktif untuk setiap token selama inferensi. Konsep dasar MoE adalah bahwa berbagai jenis input mungkin membutuhkan jenis pemrosesan yang berbeda. Daripada menggunakan satu jaringan monolitik untuk semua input, MoE membiarkan model secara dinamis memilih expert yang paling sesuai untuk konten yang diproses saat ini. Setiap expert adalah subnetwork dengan spesialisasi berbeda yang berkembang organik selama pelatihan. Router atau gating network adalah komponen yang memutuskan expert mana yang menangani setiap input token. Dalam implementasi sparse MoE untuk model bahasa, setiap token biasanya hanya dikirim ke dua dari beberapa ratus expert yang tersedia. Ini memungkinkan model memiliki miliaran parameter total tetapi hanya mengaktifkan sebagian kecil untuk setiap token selama inferensi maupun pelatihan. Total parameter dan parameter aktif adalah dua konsep berbeda dalam MoE. Mixtral 8x7B misalnya memiliki total parameter setara dengan sekitar 46 miliar tetapi hanya mengaktifkan sekitar 12 miliar untuk setiap token karena hanya memilih 2 dari 8 expert per token. Ini memberikan kualitas model yang mendekati model 46B tetapi dengan biaya komputasi mendekati model 12B."
  },
  {
    "instruction": "Bagaimana algoritma BPE tokenisasi bekerja dan mengapa ia penting dalam model bahasa modern?",
    "output": "Byte Pair Encoding atau BPE adalah algoritma tokenisasi yang banyak digunakan dalam model bahasa modern termasuk GPT dan Qwen. Berbeda dari tokenisasi berbasis kata yang memerlukan kosakata sangat besar dan tidak dapat menangani kata baru, BPE menemukan kosakata unit sub-kata yang optimal yang menyeimbangkan representasi efisien dan kemampuan menangani teks tidak terduga. Algoritma BPE awalnya dikembangkan untuk kompresi data pada tahun 1990-an. Dalam konteks NLP, ia diadaptasi untuk menemukan kosakata sub-kata secara otomatis dari data pelatihan. Proses dimulai dengan representasi setiap karakter sebagai unit terkecil. Frekuensi setiap pasangan simbol yang berdekatan dalam data dihitung secara menyeluruh. Pasangan paling sering kemudian digabung menjadi simbol baru dan ditambahkan ke kosakata. Proses ini diulang sejumlah kali yang telah ditentukan menghasilkan kosakata yang terdiri dari karakter individu, fragmen kata umum, dan kata utuh yang sering muncul. Keunggulan utama BPE adalah kemampuannya menangani kata baru yang tidak ada dalam data pelatihan. Kata tersebut akan dipecah menjadi sub-kata yang dikenal dari kosakata. Misalnya kata yang tidak biasa mungkin dipecah menjadi awalan atau akhiran yang lebih umum ditambah akar kata. Ini jauh lebih baik dari tokenisasi berbasis kata yang harus menggunakan token khusus untuk kata tidak dikenal. Ukuran kosakata dalam BPE adalah hyperparameter yang menentukan jumlah operasi penggabungan yang dilakukan."
  },
  {
    "instruction": "Jelaskan konsep positional encoding dalam Transformer dan perbedaan berbagai pendekatannya.",
    "output": "Positional encoding adalah komponen dalam arsitektur Transformer yang memberikan informasi tentang posisi relatif atau absolut token dalam urutan kepada model. Karena mekanisme self-attention dalam Transformer memproses semua token secara paralel tanpa mempertimbangkan urutan secara inheren, model tidak dapat membedakan antara susunan token yang berbeda tanpa informasi posisi yang eksplisit. Tanpa positional encoding, kalimat dengan susunan kata berbeda yang menggunakan kata yang sama akan menghasilkan representasi identik karena self-attention tidak peka terhadap urutan, yang jelas tidak diinginkan karena urutan kata fundamental untuk makna dalam bahasa alami. Positional encoding ditambahkan ke embedding token sebelum dimasukkan ke lapisan Transformer pertama. Ini menggabungkan informasi identitas token dengan informasi posisinya dalam satu representasi terpadu. Implementasi asli dalam paper Transformer menggunakan fungsi sinus dan kosinus dengan frekuensi berbeda untuk setiap dimensi embedding. Dimensi genap menggunakan fungsi sinus dan dimensi ganjil menggunakan fungsi kosinus, masing-masing dengan frekuensi yang menurun secara geometris seiring dimensi. Pendekatan ini memiliki beberapa keunggulan penting. Pertama, ia menghasilkan encoding unik untuk setiap posisi. Kedua, ia memungkinkan model bekerja dengan urutan yang lebih panjang dari yang dilihat selama pelatihan karena fungsinya kontinu dan deterministik. Ketiga, jarak relatif antar posisi dapat diwakili sebagai transformasi linear. Learned positional encoding adalah alternatif di mana embedding posisi dipelajari dari data selama pelatihan seperti embedding token biasa."
  },
  {
    "instruction": "Apa perbedaan antara full fine-tuning dan parameter-efficient fine-tuning, serta kapan menggunakan masing-masing?",
    "output": "Fine-tuning adalah proses mengadaptasi model yang telah dipra-latih untuk tugas atau domain spesifik. Tersedia berbagai strategi dengan trade-off berbeda antara kinerja, efisiensi komputasi, dan kebutuhan memori. Memilih strategi yang tepat bergantung pada sumber daya yang tersedia, ukuran dataset, dan persyaratan kinerja akhir. Full fine-tuning atau full parameter fine-tuning memperbarui semua bobot model selama proses pelatihan. Ini memberikan fleksibilitas maksimal untuk mengadaptasi model karena semua parameter dapat berubah sesuai data baru. Namun pendekatan ini membutuhkan memori yang sangat besar karena harus menyimpan gradien untuk semua parameter sekaligus, dan berisiko catastrophic forgetting di mana model kehilangan kemampuan umum yang dipelajari selama pra-pelatihan pada data skala besar. Parameter-efficient fine-tuning atau PEFT adalah keluarga teknik yang hanya melatih sebagian kecil parameter sambil membekukan sebagian besar bobot pra-latih. Ini mengurangi kebutuhan memori dan komputasi secara signifikan sambil mempertahankan sebagian besar pengetahuan pra-pelatihan. LoRA atau Low-Rank Adaptation adalah teknik PEFT yang paling populer saat ini. Ia menambahkan pasangan matriks berdimensi rendah ke lapisan attention tertentu. Hanya matriks tambahan yang jauh lebih kecil ini yang dilatih sementara bobot asli dibekukan sepenuhnya. Prefix tuning menambahkan urutan token virtual yang dapat dipelajari di awal konteks model. Model belajar menggunakan prefix ini untuk mengkondisikan perilakunya pada tugas spesifik tanpa mengubah bobot asli."
  },
  {
    "instruction": "Jelaskan cara kerja Flash Attention dan mengapa ia penting untuk training model dengan konteks panjang.",
    "output": "Flash Attention adalah algoritma yang mengimplementasikan operasi attention dalam Transformer dengan cara yang sadar terhadap hierarki memori hardware, menghasilkan peningkatan kecepatan dan efisiensi memori yang signifikan dibandingkan implementasi attention standar. Diperkenalkan oleh Tri Dao dan kolaborator dari Stanford pada 2022, Flash Attention telah menjadi komponen standar dalam pelatihan dan inferensi model bahasa besar modern hampir tanpa terkecuali. Masalah utama yang diselesaikan Flash Attention adalah bottleneck bandwidth memori GPU. Operasi attention standar harus menulis dan membaca matriks attention berukuran kuadrat dari jumlah token ke dan dari memori GPU global yang relatif lambat. Untuk urutan panjang, matriks attention ini bisa sangat besar dan transfer data ini mendominasi waktu komputasi total, bukan operasi matematika itu sendiri. Flash Attention mengatasi ini dengan menggunakan teknik tiling atau pemblokiran. Alih-alih menghitung seluruh matriks attention sekaligus, Flash Attention membagi komputasi menjadi blok-blok kecil yang muat dalam SRAM cache cepat GPU. Komputasi dilakukan pada blok yang lebih kecil ini di dalam cache, sangat mengurangi jumlah transfer ke memori lambat yang mahal secara waktu. Numerically stable online softmax adalah teknik kunci lain yang memungkinkan Flash Attention menghitung softmax secara inkremental tanpa menyimpan seluruh matriks attention. Dengan melacak statistik running maksimum dan jumlah normalisasi, softmax dapat dihitung secara numerik stabil dalam satu pass maju tanpa menyimpan matriks penengah."
  },
  {
    "instruction": "Apa itu KV cache dalam inferensi model bahasa dan bagaimana ia meningkatkan efisiensi generasi teks?",
    "output": "KV cache atau Key-Value cache adalah teknik optimisasi inferensi kritis dalam model bahasa autoregresif yang menghindari komputasi ulang representasi Key dan Value dari token yang telah diproses sebelumnya dalam konteks. Tanpa KV cache, menghasilkan setiap token baru akan membutuhkan komputasi ulang seluruh history konteks dari awal, membuat inferensi menjadi sangat lambat dan tidak praktis untuk konteks yang panjang. Model bahasa autoregresif seperti GPT menghasilkan teks token demi token secara sekuensial. Untuk menghasilkan token baru, model harus menjalankan attention antara token baru dan semua token sebelumnya dalam konteks. Dalam operasi attention, ini berarti menghitung Key dan Value untuk semua token konteks setiap kali. Karena token konteks yang ada tidak berubah antara langkah generasi, menghitungnya ulang setiap langkah merupakan pemborosan komputasi yang sangat besar dan tidak perlu. KV cache menyimpan hasil Key dan Value yang telah dihitung untuk setiap lapisan Transformer di memori GPU. Saat menghasilkan token baru, hanya Key dan Value dari token baru saja yang perlu dihitung dari awal. Kemudian attention dijalankan antara Query token baru dan semua Key yang tersimpan dalam cache termasuk token baru. Ini mengubah kompleksitas komputasi dari kuadratik menjadi linear terhadap panjang konteks untuk setiap langkah generasi individual. Kebutuhan memori adalah tantangan utama KV cache. Ukuran cache tumbuh linear dengan panjang konteks, jumlah lapisan, jumlah head, dan dimensi head model."
  },
  {
    "instruction": "Jelaskan paradigma Retrieval Augmented Generation dan bagaimana ia mengatasi keterbatasan model bahasa.",
    "output": "Retrieval Augmented Generation atau RAG adalah paradigma yang menggabungkan kemampuan generasi model bahasa dengan pengambilan informasi dari basis pengetahuan eksternal secara dinamis. Teknik ini memungkinkan model memberikan respons yang lebih akurat, terkini, dan dapat diverifikasi dibandingkan model bahasa yang hanya mengandalkan pengetahuan yang dipelajari selama pelatihan yang memiliki batas waktu. Model bahasa besar memiliki pengetahuan yang terbatas pada data pelatihan mereka dan tidak dapat mengakses informasi yang lebih baru dari setelah periode pelatihan. Mereka juga rentan terhadap halusinasi yaitu menghasilkan informasi yang terdengar meyakinkan namun faktanya salah. RAG mengatasi keterbatasan ini dengan memberikan konteks relevan yang diambil secara real-time dari basis data eksternal sebagai bagian dari input model. Komponen utama sistem RAG terdiri dari retriever dan generator yang bekerja bersama. Retriever bertugas menemukan dokumen atau potongan teks yang relevan dengan query dari basis pengetahuan yang bisa berupa dokumen internal perusahaan, artikel ilmiah, atau halaman web. Generator kemudian menggunakan dokumen yang diambil sebagai konteks tambahan untuk menghasilkan respons yang lebih informatif dan faktual. Dense retrieval menggunakan model embedding untuk mengkonversi query dan dokumen menjadi vektor dan mencari yang paling mirip secara semantik. Berbeda dari pencarian kata kunci tradisional yang mencari kecocokan leksikal eksak, dense retrieval dapat menemukan dokumen relevan yang menggunakan kata berbeda namun bermakna serupa."
  },
  {
    "instruction": "Jelaskan scaling laws dalam AI dan bagaimana ia mempengaruhi keputusan dalam pengembangan model besar.",
    "output": "Scaling laws dalam AI mengacu pada hubungan empiris yang dapat diprediksi antara ukuran model, jumlah data pelatihan, komputasi yang digunakan, dan kinerja model yang dihasilkan. Pemahaman tentang scaling laws memungkinkan peneliti dan praktisi membuat keputusan yang lebih rasional dan efisien tentang alokasi sumber daya dalam mengembangkan model baru yang lebih powerful. Penelitian awal dari OpenAI pada 2020 menemukan bahwa performa model bahasa mengikuti power law terhadap tiga faktor utama yaitu jumlah parameter model, jumlah token pelatihan, dan total FLOP komputasi yang digunakan. Setiap faktor ketika ditingkatkan secara independen menghasilkan peningkatan kinerja yang dapat diprediksi dan konsisten. Ini menunjukkan bahwa dengan meningkatkan skala, kita dapat secara deterministik mengharapkan model yang lebih baik tanpa perubahan arsitektur fundamental. Makalah Chinchilla dari DeepMind pada 2022 menyempurnakan pemahaman ini secara signifikan dengan menemukan bahwa penelitian sebelumnya melatih model yang terlalu besar untuk jumlah data yang digunakan. Untuk komputasi pelatihan optimal, jumlah parameter model dan jumlah token pelatihan harus ditingkatkan secara proporsional satu sama lain. Ini disebut compute-optimal training atau Chinchilla optimal. Model Chinchilla 70 miliar parameter yang dilatih pada 1,4 triliun token mengungguli model GPT-3 175 miliar parameter yang dilatih pada hanya 300 miliar token menggunakan komputasi pelatihan yang sebanding."
  }
]
EOF
fi

DATASET_N=$(python3 -c "import json; d=json.load(open('$DATASET_PATH')); print(len(d))")
echo ">>> Dataset siap: $DATASET_N entries unik (tidak ada repetisi)"

DATASET_TYPE='{"field_instruction":"instruction","field_output":"output","no_input_format":"{instruction}","format":"{instruction}"}'

# ── Build image jika belum ada ─────────────────────────────────────────────
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo ">>> Building Docker image (pertama kali ~20-30 menit)..."
    docker build -f dockerfiles/standalone-text-trainer.dockerfile -t "$IMAGE_NAME" .
fi

MODEL_DIR_NAME="$(echo $MODEL | tr '/' '--')"

# ── Bersihkan direktori dari run sebelumnya ────────────────────────────────
rm -rf "$CACHE_DIR/internal_datasets"
rm -rf "$CACHE_DIR/soutputs"
rm -rf "$CACHE_DIR/wandb_logs_run"
mkdir -p "$CACHE_DIR/internal_datasets"
mkdir -p "$CACHE_DIR/soutputs"
mkdir -p "$CACHE_DIR/wandb_logs_run"

# File log untuk seluruh output container (stdout + stderr)
FULL_LOG="$CACHE_DIR/test_run_${TASK_ID}.log"

echo ">>> Menjalankan container training... (log: $FULL_LOG)"
echo ""

# Jalankan container dan simpan seluruh output untuk verifikasi.
# set +e: jangan abort jika docker non-zero — kita mau tangkap exit code dulu.
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
    -v "$CACHE_DIR/wandb_logs_run:/cache/wandb_logs" \
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

        # Jalankan training — output ke stdout (ditangkap oleh tee di host)
        python text_trainer.py \
            --task-id "$TASK_ID" \
            --model "$MODEL" \
            --dataset "/cache/datasets/${TASK_ID}_train_data.json" \
            --dataset-type "$DATASET_TYPE" \
            --task-type InstructTextTask \
            --file-format json \
            --hours-to-complete "$HOURS" \
            --expected-repo-name "$REPO_NAME"
        TRAIN_EXIT=$?

        echo ""
        echo "=== TRAINING EXIT CODE: $TRAIN_EXIT ==="

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

        exit $TRAIN_EXIT
    ' 2>&1 | tee "$FULL_LOG"
# PIPESTATUS[0] = exit code dari docker run (sebelum pipe ke tee)
DOCKER_EXIT="${PIPESTATUS[0]}"
set -e   # aktifkan kembali exit-on-error

echo ""
echo "════════════════════════════════════════════════════════"
echo "  VERIFIKASI HASIL TEST"
echo "════════════════════════════════════════════════════════"

PASS=0
FAIL=0

_check() {
    local label="$1"
    local result="$2"   # "ok" atau "fail"
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

# 2. LR Estimator berjalan
grep -q "\[lr_estimator\]" "$FULL_LOG" \
    && _check "LR Estimator berjalan" "ok" "" \
    || _check "LR Estimator berjalan" "fail" "tidak ada '[lr_estimator]' di log"

# 3. Adaptive max_length berjalan
grep -q "\[seq_analyzer\]" "$FULL_LOG" \
    && _check "Adaptive max_length (seq_analyzer)" "ok" "" \
    || _check "Adaptive max_length (seq_analyzer)" "fail" "tidak ada '[seq_analyzer]' di log"

# 4. Single-run mode aktif (bukan while-True loop)
grep -q "\[sn56\] Single training run" "$FULL_LOG" \
    && _check "Single-run mode (no while-True loop)" "ok" "" \
    || _check "Single-run mode (no while-True loop)" "fail" "tidak ada '[sn56] Single training run'"

# 5. Tidak ada attempt ke-2 yang tidak perlu (menandakan loop tidak cycling)
ATTEMPT2=$(grep -c "Training attempt 2/" "$FULL_LOG" || true)
[ "$ATTEMPT2" -eq 0 ] \
    && _check "Tidak ada retry loop tak perlu (attempt 2)" "ok" "" \
    || _check "Tidak ada retry loop tak perlu (attempt 2)" "fail" "ada $ATTEMPT2 baris 'Training attempt 2/'"

# 6. Tokenized dataset ada (bukti tokenisasi berhasil)
TOK_FILE="$CACHE_DIR/internal_datasets/train_tokenized_${TASK_ID}.json"
[ -f "$TOK_FILE" ] \
    && _check "File tokenized dataset ada" "ok" "" \
    || _check "File tokenized dataset ada" "fail" "$TOK_FILE tidak ditemukan"

# 7. Submission dir ada dan berisi file (model berhasil disimpan)
SUBMIT_DIR="$CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME"
FILE_COUNT=$(ls "$SUBMIT_DIR" 2>/dev/null | wc -l || echo 0)
[ "$FILE_COUNT" -ge 2 ] \
    && _check "Submission dir berisi model ($FILE_COUNT files)" "ok" "" \
    || _check "Submission dir berisi model" "fail" "hanya $FILE_COUNT file di $SUBMIT_DIR"

# 8. loss.txt atau config.json ada di submission
[ -f "$SUBMIT_DIR/config.json" ] \
    && _check "config.json ada di submission" "ok" "" \
    || _check "config.json ada di submission" "fail" "tidak ada config.json"

echo ""
echo "  Total: $PASS lulus, $FAIL gagal"
echo "════════════════════════════════════════════════════════"

# ── Info tambahan ──────────────────────────────────────────────────────────
echo ""
echo "File output:"
echo "  Log lengkap   : $FULL_LOG"
echo "  Log training  : $CACHE_DIR/internal_datasets/train_${TASK_ID}.log"
echo "  Submission    : $SUBMIT_DIR"

# Tampilkan eval loss dari loss.txt (ditulis trainer setelah evaluasi terbaik)
LOSS_FILE="$SUBMIT_DIR/loss.txt"
if [ -f "$LOSS_FILE" ]; then
    LOSS_CONTENT=$(cat "$LOSS_FILE")
    EVAL_STEP=$(echo "$LOSS_CONTENT" | cut -d',' -f1)
    EVAL_LOSS=$(echo "$LOSS_CONTENT" | cut -d',' -f2)
    echo "  Eval loss     : $EVAL_LOSS  (best checkpoint: step $EVAL_STEP)"
    echo "  ↳ Makin rendah = makin bagus. Patokan InstructTask: <0.8 kompetitif"
else
    echo "  Eval loss     : tidak tersedia (loss.txt belum ditulis)"
fi
echo ""

# Tampilkan cuplikan LR Estimator dari log
LR_LINES=$(grep "\[lr_estimator\]" "$FULL_LOG" | head -5 || true)
if [ -n "$LR_LINES" ]; then
    echo "Cuplikan LR Estimator:"
    echo "$LR_LINES" | sed 's/^/  /'
    echo ""
fi

# Tampilkan hasil adaptive max_length
SEQ_LINES=$(grep "\[seq_analyzer\]" "$FULL_LOG" | head -3 || true)
if [ -n "$SEQ_LINES" ]; then
    echo "Cuplikan Adaptive max_length:"
    echo "$SEQ_LINES" | sed 's/^/  /'
    echo ""
fi

if [ -n "$HF_REPO" ] && grep -q "Upload selesai:" "$FULL_LOG" 2>/dev/null; then
    echo "✓ Model terupload → https://huggingface.co/$HF_REPO"
elif [ -n "$HF_REPO" ]; then
    echo "✗ Upload ke HuggingFace GAGAL — cek HF_TOKEN (perlu write permission)"
elif [ -z "$HF_REPO" ] && [ "$FILE_COUNT" -ge 2 ]; then
    echo "TIP: Untuk upload ke HF, jalankan:"
    echo "  HF_TOKEN=hf_xxx HF_REPO=yosa97/nama-repo bash examples/run_instruct.sh"
fi

# Kembalikan exit code berdasarkan hasil test
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
