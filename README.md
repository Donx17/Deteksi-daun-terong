# 🌿 Sistem Deteksi Penyakit Daun Terong
**YOLOv8 + Flask + Telegram Bot**

Proyek ini merupakan sistem deteksi penyakit daun terong berbasis **deep learning (YOLOv8)**.  
Sistem diimplementasikan dalam **aplikasi web (Flask)** dan **bot Telegram** untuk mendeteksi penyakit secara cepat, akurat, dan dapat diakses dari berbagai platform.  

---

## 📂 Struktur Folder

```
ADAMA/
│── static/                           # File statis (CSS, JS, gambar)
│── templates/                        # Template HTML (Flask)
│   ├── base.html
│   ├── dashboard.html
│   ├── history.html
│   ├── live.html
│   └── upload.html
│── app.py                            # File utama Flask + Telegram Bot
│── best.pt                           # Model YOLOv8 hasil training
│── bestii.pt                         # Model YOLOv8 alternatif
│── telegram_detection_history.json   # Riwayat deteksi dari Telegram
│── web_detection_history.json        # Riwayat deteksi dari Web
│── .env                              # Variabel lingkungan (TOKEN, dsb.)
│── requirements.txt                  # Daftar dependensi Python
│── README.md                         # Dokumentasi proyek
```

---

## ⚙️ Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/username/eggplant-disease-detection.git
cd eggplant-disease-detection
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
```

Aktifkan environment:
- **Windows**
  ```bash
  venv\Scripts\activate
  ```
- **Linux/Mac**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Buat File `.env`
Buat file bernama `.env` di folder utama, isi dengan:

```env
TELEGRAM_TOKEN=masukkan_token_bot_anda
TELEGRAM_CHAT_ID=masukkan_chat_id_anda
```

---

## ▶️ Cara Menjalankan

### Menjalankan Aplikasi Web
```bash
python app.py
```
Kemudian buka browser:  
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Menggunakan Telegram Bot
1. Buka aplikasi **Telegram**.  
2. Cari bot Anda (misalnya: `@nama_bot_anda`).  
3. Kirim gambar daun terong.  
4. Bot akan mengembalikan hasil deteksi penyakit.  

---

## 🧪 Training Model YOLOv8

Model dilatih menggunakan dataset 874 gambar daun terong (augmentasi → 2.097 gambar).  
Training dilakukan di Google Colab dengan GPU Tesla T4.

Contoh script training:
```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(data="data.yaml", epochs=25, imgsz=360, batch=16)
```

---

## 📊 Hasil Penelitian

- Akurasi Aplikasi Web: **95%**  
- Akurasi Telegram Bot: **93%**  
- mAP50: **97,2%**

---

## 🚀 Teknologi yang Digunakan

- Python 3.10+
- Flask
- YOLOv8 (Ultralytics)
- OpenCV
- Python-Telegram-Bot
- Numpy
- HTML, CSS, JavaScript

---

## 👨‍💻 Pengembang
- **Syahrul Halik** – Universitas Negeri Makassar (2025)  

---

## 📌 Catatan
- File model `.pt` (misalnya `best.pt`) berukuran besar. Jika tidak bisa diupload ke GitHub, Anda bisa menyediakannya via **Google Drive** dan mencantumkan link di sini.  
- Jangan upload file `.env` ke publik karena berisi **token rahasia**.  
