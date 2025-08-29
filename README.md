# Sistem Deteksi Penyakit Daun Terong (YOLOv8 + Flask + Telegram Bot)

Proyek ini merupakan sistem berbasis web dan bot Telegram untuk mendeteksi penyakit pada daun terong menggunakan algoritma YOLOv8.  
Sistem dikembangkan untuk membantu petani dan peneliti dalam mendeteksi penyakit secara otomatis, cepat, dan akurat.

## 📂 Struktur Folder
│── static/ # File statis (CSS, JS, gambar)
│── templates/ # Template HTML (Flask)
│ ├── base.html
│ ├── dashboard.html
│ ├── history.html
│ ├── live.html
│ └── upload.html
│── app.py # File utama Flask + Telegram Bot
│── best.pt # Model YOLOv8 hasil training
│── bestii.pt # Model YOLOv8 alternatif
│── telegram_detection_history.json # Riwayat deteksi dari Telegram
│── web_detection_history.json # Riwayat deteksi dari Web
│── .env # Variabel lingkungan (Token Telegram, dll.)
│── requirements.txt # Daftar dependensi Python
│── README.md # Dokumentasi proyek


## ⚙️ Instalasi

1. Clone repository:
   ```bash
   git clone https://github.com/username/nama-repo.git
   cd nama-repo


Buat dan aktifkan virtual environment:

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependensi:

pip install -r requirements.txt


Buat file .env untuk menyimpan konfigurasi:

TELEGRAM_TOKEN=token_bot_anda
TELEGRAM_CHAT_ID=chat_id_anda

▶️ Cara Menjalankan

Menjalankan aplikasi web:

python app.py


Lalu akses melalui browser: http://127.0.0.1:5000

Menggunakan Telegram Bot:
Kirimkan gambar daun terong ke bot Telegram Anda, hasil deteksi akan dikirimkan kembali.

🧪 Pelatihan Model

Model YOLOv8 dilatih dengan:

Dataset: 874 gambar daun terong (hasil augmentasi → 2.097 gambar).

Platform: Google Colab (GPU Tesla T4).

Epoch: 25

Ukuran gambar: 360x360 piksel.

Contoh kode training:

from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(data="data.yaml", epochs=25, imgsz=360, batch=16)

📊 Hasil Deteksi

Akurasi Web App: 95%

Akurasi Telegram Bot: 93%

mAP50: 97,2%

👨‍💻 Pengembang

Syahrul Halik – Universitas Negeri Makassar (2025)