from flask import Flask, render_template, request, Response, send_file, jsonify, redirect, url_for
import cv2
import os
from dotenv import load_dotenv
import base64
from ultralytics import YOLO
import threading
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask_cors import CORS
import numpy as np
import time
import platform
from queue import Queue, Empty
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import logging
import sys
import uuid
import mimetypes
import traceback

# Setup logging yang lebih detail dan profesional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Ambil token dari environment variable (lebih aman)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TELEGRAM_BOT_TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN tidak disetel! Sistem Telegram tidak akan berfungsi.")
else:
    logger.info("TELEGRAM_BOT_TOKEN telah disetel dengan benar")

# Inisialisasi queue untuk komunikasi Telegram
telegram_queue = Queue()
telegram_chat_ids = set()
telegram_chat_ids_lock = threading.Lock()

# Direktori untuk unggahan dan hasil deteksi
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'
WEB_HISTORY_FILE = 'web_detection_history.json'
TELEGRAM_HISTORY_FILE = 'telegram_detection_history.json'

# Buat direktori jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Variabel global untuk penanganan kamera
camera_active = False
detection_results = {}
detection_results_lock = threading.Lock()  # Lock untuk thread safety
camera_thread = None
camera = None
camera_lock = threading.Lock()
camera_cache = None

# Muat riwayat deteksi dengan pemisahan sumber
def load_history():
    """Muat riwayat deteksi dari file JSON dengan pemisahan sumber."""
    web_history = []
    telegram_history = []
    
    # Muat history web
    if os.path.exists(WEB_HISTORY_FILE):
        try:
            with open(WEB_HISTORY_FILE, 'r') as f:
                web_history = json.load(f)
            logger.info(f"Riwayat deteksi web dimuat: {len(web_history)} entri")
        except Exception as e:
            logger.error(f"Gagal memuat riwayat deteksi web: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            web_history = []
    
    # Muat history telegram
    if os.path.exists(TELEGRAM_HISTORY_FILE):
        try:
            with open(TELEGRAM_HISTORY_FILE, 'r') as f:
                telegram_history = json.load(f)
            logger.info(f"Riwayat deteksi Telegram dimuat: {len(telegram_history)} entri")
        except Exception as e:
            logger.error(f"Gagal memuat riwayat deteksi Telegram: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            telegram_history = []
    
    return web_history, telegram_history

web_history, telegram_history = load_history()

def save_history(source):
    """Simpan riwayat deteksi ke file JSON yang sesuai dengan sumber."""
    try:
        if source == 'web':
            with open(WEB_HISTORY_FILE, 'w') as f:
                json.dump(web_history, f, indent=4)
            logger.debug("Riwayat deteksi web berhasil disimpan")
        elif source == 'telegram':
            with open(TELEGRAM_HISTORY_FILE, 'w') as f:
                json.dump(telegram_history, f, indent=4)
            logger.debug("Riwayat deteksi Telegram berhasil disimpan")
    except Exception as e:
        logger.error(f"Gagal menyimpan riwayat deteksi {source}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def allowed_file(filename):
    """Validasi ekstensi file yang diizinkan."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_objects(img):
    """Deteksi objek dalam gambar menggunakan YOLOv8 dengan error handling lebih baik."""
    try:
        # Resize gambar untuk deteksi yang lebih cepat (416x416)
        img_for_detection = cv2.resize(img, (416, 416))
        
        results = model(img_for_detection, conf=0.5, verbose=False)
        detection_info = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Hitung koordinat relatif ke gambar asli
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 * img.shape[1] / 416)
                y1 = int(y1 * img.shape[0] / 416)
                x2 = int(x2 * img.shape[1] / 416)
                y2 = int(y2 * img.shape[0] / 416)
                
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0]) * 100
                class_name = model.names[cls_id]
                
                # Validasi koordinat
                height, width = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Gambar kotak pembatas
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Tambahkan label
                label = f"{class_name}: {confidence:.2f}%"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(img, (x1, y1 - 20), (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                detection_info.append({
                    'objek': class_name,
                    'akurasi': f"{confidence:.2f}%",
                    'koordinat': [x1, y1, x2, y2]
                })
        return img, detection_info
    except Exception as e:
        logger.error(f"Error dalam deteksi objek: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return img, []

def send_to_telegram(filename, detection_info, output_path, detection_time, total_objects, average_confidence):
    """Kirim hasil deteksi ke Telegram melalui queue dengan validasi lebih ketat"""
    try:
        # Validasi file sebelum dikirim
        if not os.path.exists(output_path):
            logger.error(f"File tidak ditemukan untuk dikirim ke Telegram: {output_path}")
            return
        # Pastikan ekstensi file valid
        file_ext = os.path.splitext(output_path)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png', '.mp4']:
            logger.error(f"Format file tidak didukung untuk Telegram: {file_ext}")
            return
        # Masukkan permintaan ke queue
        telegram_queue.put({
            'filename': filename,
            'detection_info': detection_info,
            'output_path': output_path,
            'detection_time': detection_time,
            'total_objects': total_objects,
            'average_confidence': average_confidence
        })
        logger.info(f"Permintaan Telegram ditambahkan ke queue untuk {filename}")
    except Exception as e:
        logger.error(f"Error menambahkan ke queue Telegram: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

@app.route('/')
def index():
    """Tampilkan halaman utama."""
    return render_template('dashboard.html')

@app.route('/upload', methods=['GET'])
def upload():
    """Tangani halaman unggah."""
    return render_template('upload.html')

@app.route('/history')
def history():
    """Tampilkan halaman riwayat deteksi."""
    return render_template('history.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Proses gambar atau video yang diunggah dengan perbaikan kritis."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Generate filename unik untuk menghindari tabrakan
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(filepath)
            
            # Deteksi gambar
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(filepath)
                if img is None:
                    return jsonify({'error': 'Gagal membaca gambar'}), 400
                
                start_time = time.time()
                img, detection_info = detect_objects(img)
                detection_time = time.time() - start_time
                total_objects = len(detection_info)
                average_confidence = sum(float(info['akurasi'].strip('%')) for info in detection_info) / total_objects if total_objects > 0 else 0.0
                
                # Simpan hasil dengan filename unik
                output_filename = f"detected_{unique_filename}"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                cv2.imwrite(output_path, img)
                
                # Simpan riwayat dengan informasi lengkap dan sumber 'web'
                history_entry = {
                    'id': str(uuid.uuid4()),
                    'original_filename': filename,
                    'filename': unique_filename,
                    'output_filename': output_filename,
                    'results': detection_info,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'detection_time': detection_time,
                    'total_objects': total_objects,
                    'average_confidence': average_confidence,
                    'media_type': 'image',
                    'source': 'web'  # TANDAI SUMBER SEBAGAI WEB
                }
                
                web_history.append(history_entry)
                save_history('web')
                
                # Encode gambar untuk respons API
                _, buffer = cv2.imencode(os.path.splitext(filename)[1], img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Kembalikan respons JSON untuk API
                return jsonify({
                    'success': True,
                    'image': img_base64,
                    'results': detection_info,
                    'detection_time': detection_time,
                    'total_objects': total_objects,
                    'average_confidence': average_confidence,
                    'filename': filename
                })
            
            # Deteksi video
            elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
                video_capture = cv2.VideoCapture(filepath)
                if not video_capture.isOpened():
                    return jsonify({'error': 'Gagal membuka video'}), 400
                
                # Gunakan codec yang kompatibel Telegram (MP4)
                base, ext = os.path.splitext(filename)
                output_filename = f"detected_{unique_filename}.mp4"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                
                # Tentukan resolusi video
                width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Gunakan codec H264 untuk kompatibilitas Telegram yang lebih baik
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
                
                detection_info = []
                start_time = time.time()
                frame_count = 0
                processed_frames = 0
                
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    
                    # Proses setiap frame ke-3 untuk menghemat sumber daya
                    if frame_count % 3 == 0:
                        frame, frame_detection_info = detect_objects(frame)
                        detection_info.extend(frame_detection_info)
                        processed_frames += 1
                    
                    out.write(frame)
                    frame_count += 1
                
                video_capture.release()
                out.release()
                
                detection_time = time.time() - start_time
                total_objects = len(detection_info)
                average_confidence = sum(float(info['akurasi'].strip('%')) for info in detection_info) / total_objects if total_objects > 0 else 0.0
                
                # Simpan riwayat dengan informasi lengkap dan sumber 'web'
                history_entry = {
                    'id': str(uuid.uuid4()),
                    'original_filename': filename,
                    'filename': unique_filename,
                    'output_filename': output_filename,
                    'results': detection_info,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'detection_time': detection_time,
                    'total_objects': total_objects,
                    'average_confidence': average_confidence,
                    'media_type': 'video',
                    'source': 'web'  # TANDAI SUMBER SEBAGAI WEB
                }
                
                web_history.append(history_entry)
                save_history('web')
                
                # Kembalikan respons JSON untuk API
                return jsonify({
                    'success': True,
                    'video': output_filename,
                    'results': detection_info,
                    'detection_time': detection_time,
                    'total_objects': total_objects,
                    'average_confidence': average_confidence,
                    'filename': filename
                })
            
            else:
                return jsonify({'error': 'Format file tidak didukung'}), 400
        else:
            return jsonify({'error': 'Format file tidak didukung'}), 400
    except Exception as e:
        logger.exception(f"Error dalam deteksi: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/api/history', methods=['GET'])
def api_history():
    """Endpoint API untuk mengambil riwayat deteksi web dalam format JSON."""
    return jsonify(web_history)

@app.route('/api/telegram-history', methods=['GET'])
def api_telegram_history():
    """Endpoint API untuk mengambil riwayat deteksi Telegram dalam format JSON (hanya untuk internal)."""
    # Endpoint ini seharusnya tidak diakses dari web, hanya untuk internal
    if request.remote_addr not in ['127.0.0.1', 'localhost']:
        return jsonify({'error': 'Akses ditolak'}), 403
    return jsonify(telegram_history)

@app.route('/api/history/stats', methods=['GET'])
def api_history_stats():
    """Endpoint API untuk statistik riwayat deteksi web."""
    total_images = 0
    successful_detections = 0
    total_processing_time = 0
    disease_count = {}
    
    for entry in web_history:
        if entry['media_type'] == 'image' or entry['media_type'] == 'video':
            total_images += 1
            if entry['results'] and len(entry['results']) > 0:
                successful_detections += 1
                # Hitung waktu pemrosesan
                if 'detection_time' in entry:
                    total_processing_time += entry['detection_time']
                # Hitung distribusi penyakit
                for result in entry['results']:
                    disease = result['objek']
                    disease_count[disease] = disease_count.get(disease, 0) + 1
    
    avg_processing_time = total_processing_time / successful_detections if successful_detections > 0 else 0
    
    # Konversi ke format yang dibutuhkan chart
    disease_labels = list(disease_count.keys())
    disease_values = [disease_count[label] for label in disease_labels]
    
    return jsonify({
        'total_images': total_images,
        'successful_detections': successful_detections,
        'avg_processing_time': avg_processing_time,
        'disease_labels': disease_labels,
        'disease_values': disease_values
    })

@app.route('/api/system/status', methods=['GET'])
def api_system_status():
    """Endpoint API untuk status sistem."""
    # Periksa ketersediaan kamera
    try:
        response = request.get('/check_camera', timeout=2)
        camera_available = response.status_code == 200
    except:
        camera_available = False
    
    return jsonify({
        'camera_connected': camera_available,
        'telegram_connected': bool(TELEGRAM_BOT_TOKEN)
    })

@app.route('/check_camera', methods=['GET'])
def check_camera():
    """Periksa kamera yang tersedia dengan optimasi yang diperbaiki."""
    global camera_cache
    if camera_cache is not None:
        logger.info("Menggunakan cache kamera")
        return jsonify({'cameras': camera_cache}), 200
    
    available_cameras = []
    system = platform.system()
    backends = [
        (cv2.CAP_ANY, "Default"),
        (cv2.CAP_FFMPEG, "FFMPEG")
    ]
    
    if system == "Linux":
        backends.append((cv2.CAP_V4L2, "V4L2"))
    elif system == "Windows":
        backends.append((cv2.CAP_DSHOW, "DSHOW"))
    elif system == "Darwin":  # macOS
        backends.append((cv2.CAP_AVFOUNDATION, "AVFoundation"))
    
    result_queue = Queue()
    threads = []
    max_indices = 10  # Periksa lebih banyak indeks untuk kompatibilitas lebih baik
    
    # Buat thread untuk setiap kombinasi kamera dan backend
    for backend, backend_name in backends:
        for i in range(max_indices):
            thread = threading.Thread(
                target=check_camera_with_timeout, 
                args=(i, backend, backend_name, result_queue, 1.5),
                daemon=True
            )
            threads.append(thread)
            thread.start()
    
    # Kumpulkan hasil dengan timeout yang lebih realistis
    cameras_found = set()
    for _ in range(len(threads)):
        try:
            result = result_queue.get(timeout=1.5)
            if result is not None and result['index'] not in cameras_found:
                available_cameras.append(result)
                cameras_found.add(result['index'])
        except Empty:
            continue
    
    # Tunggu thread selesai dengan timeout yang lebih baik
    for thread in threads:
        thread.join(timeout=0.5)
    
    if not available_cameras:
        error_msg = "Tidak ada kamera yang tersedia. Pastikan kamera terhubung dan izin diberikan."
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 404
    
    camera_cache = available_cameras
    logger.info(f"Kamera ditemukan: {available_cameras}")
    return jsonify({'cameras': available_cameras}), 200

def check_camera_with_timeout(index, backend, backend_name, result_queue, timeout=1.5):
    """Periksa kamera dengan batas waktu yang lebih realistis."""
    try:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            # Coba baca beberapa frame untuk memastikan kamera benar-benar berfungsi
            for _ in range(3):
                success, _ = cap.read()
                if success:
                    result_queue.put({'index': index, 'name': f"Kamera {index} ({backend_name})", 'backend': backend_name})
                    break
        cap.release()
    except Exception as e:
        logger.debug(f"Error memeriksa kamera indeks {index}, backend {backend_name}: {e}")
    finally:
        # Pastikan selalu mengirim sinyal selesai
        result_queue.put(None)

@app.route('/test_camera', methods=['POST'])
def test_camera():
    """Uji kamera dengan menangkap satu frame dengan penanganan backend yang lebih baik."""
    try:
        camera_index = int(request.json.get('camera_index', 0))
        logger.info(f"Menguji kamera: indeks={camera_index}")
        
        # Coba beberapa backend dengan urutan prioritas
        system = platform.system()
        backends = []
        
        if system == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_FFMPEG]
        elif system == "Linux":
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY, cv2.CAP_FFMPEG]
        elif system == "Darwin":  # macOS
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY, cv2.CAP_FFMPEG]
        else:
            backends = [cv2.CAP_ANY, cv2.CAP_FFMPEG]
        
        cap = None
        for backend in backends:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                # Set resolusi yang lebih rendah untuk deteksi real-time
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Coba baca beberapa frame
                for _ in range(5):
                    success, frame = cap.read()
                    if success and frame is not None and frame.size > 0:
                        cap.release()
                        _, buffer = cv2.imencode('.jpg', frame)
                        img_base64 = base64.b64encode(buffer).decode('utf-8')
                        logger.info(f"Frame berhasil ditangkap pada indeks {camera_index}")
                        return jsonify({'message': 'Frame berhasil ditangkap', 'image': img_base64}), 200
                
                cap.release()
        
        logger.error(f"Gagal membuka kamera pada indeks {camera_index}")
        return jsonify({'error': f"Gagal membuka kamera pada indeks {camera_index}"}), 500
    except Exception as e:
        logger.exception(f"Error menguji kamera: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Gagal menguji kamera: {str(e)}'}), 500

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Mulai feed kamera dengan penanganan error yang lebih baik."""
    global camera_active, camera_thread, camera
    
    with camera_lock:
        if camera_active:
            return jsonify({'message': 'Kamera sudah berjalan'}), 200
        
        try:
            camera_index = int(request.json.get('camera_index', 0))
            logger.info(f"Memulai kamera: indeks={camera_index}")
            
            # Coba beberapa backend dengan urutan prioritas
            system = platform.system()
            backends = []
            
            if system == "Windows":
                backends = [cv2.CAP_DSHOW, cv2.CAP_ANY, cv2.CAP_FFMPEG]
            elif system == "Linux":
                backends = [cv2.CAP_V4L2, cv2.CAP_ANY, cv2.CAP_FFMPEG]
            elif system == "Darwin":  # macOS
                backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY, cv2.CAP_FFMPEG]
            else:
                backends = [cv2.CAP_ANY, cv2.CAP_FFMPEG]
            
            cap = None
            for backend in backends:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    # Set resolusi yang lebih rendah untuk deteksi real-time
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    # Verifikasi dengan membaca beberapa frame
                    for _ in range(5):
                        success, frame = cap.read()
                        if success and frame is not None and frame.size > 0:
                            camera = cap
                            break
                    
                    if camera is not None:
                        break
                    
                    cap.release()
            
            if camera is None or not camera.isOpened():
                logger.error(f"Gagal membuka kamera pada indeks {camera_index}")
                if camera is not None:
                    camera.release()
                    camera = None
                return jsonify({
                    'error': 'Gagal membuka kamera. Pastikan kamera terhubung dan izin diberikan.'
                }), 500
            
            # Reset hasil deteksi
            with detection_results_lock:
                detection_results.clear()
            
            camera_active = True
            
            # Mulai thread kamera
            camera_thread = threading.Thread(target=gen_frames, daemon=True)
            camera_thread.start()
            
            logger.info(f"Kamera dimulai: indeks={camera_index} dengan resolusi 640x480")
            return jsonify({'message': 'Kamera berhasil dimulai'}), 200
        except Exception as e:
            if camera is not None:
                camera.release()
                camera = None
            logger.exception(f"Error memulai kamera: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Gagal memulai kamera: {str(e)}'}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Hentikan feed kamera dengan pembersihan yang lebih baik."""
    global camera_active, camera, camera_thread
    
    with camera_lock:
        if not camera_active:
            return jsonify({'message': 'Kamera sudah berhenti'}), 200
        
        try:
            camera_active = False
            
            if camera is not None:
                camera.release()
                camera = None
            
            if camera_thread is not None and camera_thread.is_alive():
                camera_thread.join(timeout=2.0)
            
            # Pastikan kamera benar-benar berhenti
            time.sleep(0.5)
            
            with detection_results_lock:
                detection_results.clear()
            
            logger.info("Kamera dihentikan")
            return jsonify({'message': 'Kamera berhasil dihentikan'}), 200
        except Exception as e:
            logger.exception(f"Error menghentikan kamera: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Gagal menghentikan kamera: {str(e)}'}), 500

def gen_frames():
    """Hasilkan frame kamera dengan deteksi dan thread safety."""
    global camera_active, camera
    
    frame_count = 0
    last_detection_time = 0
    detection_interval = 0.2  # Deteksi setiap 0.2 detik (5 FPS)
    frame_skip = 2  # Lewati 1 frame dari setiap 2
    
    while camera_active and camera is not None:
        with camera_lock:
            if not camera_active or camera is None:
                break
            
            # Ambil beberapa frame dan hanya proses salah satunya
            for _ in range(frame_skip):
                success, frame = camera.read()
                if not success or frame is None or frame.size == 0:
                    break
        
        if not success or frame is None or frame.size == 0:
            logger.warning("Gagal membaca frame kamera, mencoba reset...")
            time.sleep(0.5)
            continue
        
        try:
            current_time = time.time()
            
            # Lakukan deteksi pada interval tertentu
            if current_time - last_detection_time >= detection_interval:
                start_time = time.time()
                frame, detection_info = detect_objects(frame)
                detection_time = time.time() - start_time
                last_detection_time = current_time
                
                total_objects = len(detection_info)
                average_confidence = sum(float(info['akurasi'].strip('%')) for info in detection_info) / total_objects if total_objects > 0 else 0.0
                
                # Simpan hasil deteksi terbaru dengan thread safety
                results = {
                    'results': detection_info,
                    'detection_time': detection_time,
                    'total_objects': total_objects,
                    'average_confidence': average_confidence,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with detection_results_lock:
                    detection_results.update(results)
            
            # Encode frame untuk streaming dengan kualitas lebih rendah
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            frame_count += 1
        except Exception as e:
            logger.exception(f"Error memproses frame: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            time.sleep(0.1)
    
    # Pembersihan
    with camera_lock:
        camera_active = False
        if camera is not None:
            camera.release()
            camera = None
        
        with detection_results_lock:
            detection_results.clear()
        
        logger.info("Pembersihan kamera selesai")

@app.route('/video_feed')
def video_feed():
    """Rute streaming video dengan penanganan error yang lebih baik."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detection_results')
def get_detection_results():
    """Kembalikan hasil deteksi terbaru dengan thread safety."""
    with detection_results_lock:
        return jsonify(detection_results)

@app.route('/delete_entry/<entry_id>', methods=['DELETE'])
def delete_entry(entry_id):
    """Hapus entri riwayat web dengan penanganan file yang lebih baik."""
    try:
        global web_history
        entry_to_delete = None
        
        for entry in web_history:
            if entry.get('id') == entry_id:
                entry_to_delete = entry
                break
        
        if not entry_to_delete:
            return jsonify({'success': False, 'error': 'Entri tidak ditemukan'}), 404
        
        # Hapus file terkait
        upload_path = os.path.join(UPLOAD_FOLDER, entry_to_delete['filename'])
        output_path = os.path.join(OUTPUT_FOLDER, entry_to_delete['output_filename'])
        
        deleted_files = []
        
        for path in [upload_path, output_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    deleted_files.append(path)
                    logger.info(f"File dihapus: {path}")
                except Exception as e:
                    logger.error(f"Gagal menghapus {path}: {e}")
        
        # Perbarui riwayat web
        web_history = [entry for entry in web_history if entry.get('id') != entry_id]
        save_history('web')
        
        return jsonify({
            'success': True,
            'deleted_files': deleted_files,
            'message': f"{len(deleted_files)} file berhasil dihapus"
        })
    except Exception as e:
        logger.exception(f"Error menghapus entri: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download/<entry_id>')
def download_file(entry_id):
    """Unduh file hasil deteksi web dengan penanganan yang lebih baik."""
    try:
        # Cari entri berdasarkan ID di riwayat web
        entry = next((e for e in web_history if e.get('id') == entry_id), None)
        
        if not entry:
            return jsonify({'error': 'Entri tidak ditemukan'}), 404
        
        # Tentukan path file berdasarkan tipe media
        file_path = os.path.join(OUTPUT_FOLDER, entry['output_filename'])
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File tidak ditemukan'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=entry['original_filename'])
    except Exception as e:
        logger.exception(f"Error mengunduh file: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# ======================
# Telegram Bot Handlers
# ======================

async def start(update: Update, context: CallbackContext) -> None:
    """Handler untuk perintah /start dengan penanganan error yang lebih baik"""
    try:
        chat_id = update.message.chat_id
        with telegram_chat_ids_lock:
            if chat_id not in telegram_chat_ids:
                telegram_chat_ids.add(chat_id)
                logger.info(f"Pengguna Telegram terdaftar: {chat_id}")
        
        await update.message.reply_text(
            '🤖 *Sistem Deteksi Daun Terong*\n'
            'Anda akan menerima hasil deteksi di sini.\n'
            'Kirim foto atau video untuk deteksi objek.\n'
            'Perintah tersedia:\n'
            '/start - Mulai bot\n'
            '/help - Bantuan',
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.exception(f"Error di /start: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await update.message.reply_text("Maaf, terjadi kesalahan teknis.")

async def help_command(update: Update, context: CallbackContext) -> None:
    """Handler untuk perintah /help"""
    try:
        help_text = (
            "📚 *Bantuan Sistem Deteksi Daun Terong*\n"
            "Anda dapat:\n"
            "📸 Mengirim foto untuk deteksi daun terong\n"
            "🎥 Mengirim video untuk analisis\n"
            "Format yang didukung:\n"
            "- Gambar: JPG, JPEG, PNG\n"
            "- Video: MP4, AVI, MOV\n"
            "Sistem akan memberikan:\n"
            "- Gambar/video dengan bounding box\n"
            "- Jenis objek dan tingkat kepercayaan\n"
            "- Statistik deteksi"
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')
    except Exception as e:
        logger.exception(f"Error di /help: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await update.message.reply_text("Maaf, terjadi kesalahan teknis.")

async def handle_image(update: Update, context: CallbackContext) -> None:
    """Handler untuk foto yang dikirim dengan penanganan error yang lebih baik"""
    try:
        user = update.message.from_user
        logger.info(f"Foto diterima dari {user.id}")
        
        # Ambil foto terbesar
        photo_file = await update.message.photo[-1].get_file()
        filename = f"{user.id}_{int(time.time())}_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Download foto
        await photo_file.download_to_drive(filepath)
        
        # Proses deteksi
        img = cv2.imread(filepath)
        if img is None:
            await update.message.reply_text("Gagal membaca gambar. Pastikan format gambar valid.")
            return
        
        start_time = time.time()
        img, detection_info = detect_objects(img)
        detection_time = time.time() - start_time
        total_objects = len(detection_info)
        average_confidence = sum(float(info['akurasi'].strip('%')) for info in detection_info) / total_objects if total_objects > 0 else 0.0
        
        # Simpan hasil
        output_filename = f"bot_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, img)
        
        # Simpan riwayat dengan sumber 'telegram'
        history_entry = {
            'id': str(uuid.uuid4()),
            'original_filename': f"telegram_{filename}",
            'filename': filename,
            'output_filename': output_filename,
            'results': detection_info,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detection_time': detection_time,
            'total_objects': total_objects,
            'average_confidence': average_confidence,
            'media_type': 'image',
            'source': 'telegram'  # TANDAI SUMBER SEBAGAI TELEGRAM
        }
        
        telegram_history.append(history_entry)
        save_history('telegram')
        
        # Kirim hasil ke pengguna
        with open(output_path, 'rb') as f:
            await update.message.reply_photo(
                photo=f, 
                caption="📸 Hasil Deteksi dari Bot"
            )
        
        # Format pesan statistik
        if detection_info:
            detection_text = "\n".join([f"• {info['objek']} ({info['akurasi']})" for info in detection_info])
            stats_text = (
                f"\n📊 Statistik:\n"
                f"• Waktu Deteksi: {detection_time:.2f} detik\n"
                f"• Total Objek: {total_objects}\n"
                f"• Rata-rata Akurasi: {average_confidence:.2f}%"
            )
            await update.message.reply_text(f"🔍 Hasil Deteksi:\n{detection_text}{stats_text}")
        else:
            await update.message.reply_text("⚠️ Tidak ada daun terong yang terdeteksi dalam gambar ini.")
    except Exception as e:
        logger.exception(f"Error memproses foto: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await update.message.reply_text("Maaf, terjadi kesalahan saat memproses gambar.")

async def handle_video(update: Update, context: CallbackContext) -> None:
    """Handler untuk video yang dikirim dengan penanganan error yang lebih baik"""
    try:
        user = update.message.from_user
        logger.info(f"Video diterima dari {user.id}")
        
        video_file = await update.message.video.get_file()
        filename = f"{user.id}_{int(time.time())}_{uuid.uuid4().hex}.mp4"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Download video
        await video_file.download_to_drive(filepath)
        
        # Proses video
        video_capture = cv2.VideoCapture(filepath)
        if not video_capture.isOpened():
            await update.message.reply_text("Gagal membuka video. Pastikan format video valid.")
            return
        
        # Persiapkan output
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_filename = f"bot_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        
        detection_info = []
        start_time = time.time()
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Proses setiap frame ke-3
            if frame_count % 3 == 0:
                frame, frame_detection_info = detect_objects(frame)
                detection_info.extend(frame_detection_info)
                processed_frames += 1
            
            out.write(frame)
            frame_count += 1
        
        video_capture.release()
        out.release()
        
        detection_time = time.time() - start_time
        total_objects = len(detection_info)
        average_confidence = sum(float(info['akurasi'].strip('%')) for info in detection_info) / total_objects if total_objects > 0 else 0.0
        
        # Simpan riwayat dengan sumber 'telegram'
        history_entry = {
            'id': str(uuid.uuid4()),
            'original_filename': f"telegram_{filename}",
            'filename': filename,
            'output_filename': output_filename,
            'results': detection_info,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detection_time': detection_time,
            'total_objects': total_objects,
            'average_confidence': average_confidence,
            'media_type': 'video',
            'source': 'telegram'  # TANDAI SUMBER SEBAGAI TELEGRAM
        }
        
        telegram_history.append(history_entry)
        save_history('telegram')
        
        # Kirim hasil ke pengguna
        with open(output_path, 'rb') as f:
            await update.message.reply_video(
                video=f,
                caption="🎥 Hasil Deteksi Video"
            )
        
        # Format pesan statistik
        if detection_info:
            detection_text = "\n".join([f"• {info['objek']} ({info['akurasi']})" for info in detection_info[:5]])  # Batasi 5 objek
            if len(detection_info) > 5:
                detection_text += f"\n... dan {len(detection_info)-5} objek lainnya"
            
            stats_text = (
                f"\n📊 Statistik:\n"
                f"• Durasi Deteksi: {detection_time:.2f} detik\n"
                f"• Total Frame Diproses: {processed_frames}\n"
                f"• Total Objek Terdeteksi: {total_objects}\n"
                f"• Rata-rata Akurasi: {average_confidence:.2f}%"
            )
            await update.message.reply_text(f"🔍 Hasil Deteksi:\n{detection_text}{stats_text}")
        else:
            await update.message.reply_text("⚠️ Tidak ada daun terong yang terdeteksi dalam video ini.")
    except Exception as e:
        logger.exception(f"Error memproses video: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await update.message.reply_text("Maaf, terjadi kesalahan saat memproses video.")

# ======================
# Telegram Bot Runner
# ======================

def run_telegram_bot():
    """Jalankan bot Telegram dalam thread terpisah dengan error handling yang lebih baik"""
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Token Telegram tidak disetel. Bot Telegram tidak akan berjalan.")
        return
    
    logger.info("Memulai bot Telegram...")
    
    # Buat event loop untuk thread ini
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Inisialisasi bot dengan konfigurasi yang lebih baik
        application = Application.builder() \
            .token(TELEGRAM_BOT_TOKEN) \
            .connect_timeout(30.0) \
            .read_timeout(30.0) \
            .write_timeout(30.0) \
            .pool_timeout(30.0) \
            .build()
        
        # Tambahkan handler
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(MessageHandler(filters.PHOTO, handle_image))
        application.add_handler(MessageHandler(filters.VIDEO, handle_video))
        
        # Jalankan background task untuk queue Telegram
        async def process_telegram_queue():
            while True:
                try:
                    # Ambil request dari queue dengan timeout
                    try:
                        request = await loop.run_in_executor(None, telegram_queue.get, True, 2.0)
                    except Empty:
                        continue
                    
                    # Kirim ke semua pengguna terdaftar
                    with telegram_chat_ids_lock:
                        chat_ids = list(telegram_chat_ids)
                    
                    if not chat_ids:
                        continue
                    
                    for chat_id in chat_ids:
                        try:
                            # Tentukan tipe konten
                            is_image = request['output_path'].lower().endswith(('.png', '.jpg', '.jpeg'))
                            
                            # Kirim media
                            if is_image:
                                with open(request['output_path'], 'rb') as f:
                                    await application.bot.send_photo(
                                        chat_id=chat_id,
                                        photo=f,
                                        caption="📸 Hasil Deteksi Langsung"
                                    )
                            else:
                                with open(request['output_path'], 'rb') as f:
                                    await application.bot.send_video(
                                        chat_id=chat_id,
                                        video=f,
                                        caption="🎥 Hasil Deteksi Langsung"
                                    )
                            
                            # Format pesan statistik
                            if request['detection_info']:
                                detection_text = "\n".join([f"• {info['objek']} ({info['akurasi']})" for info in request['detection_info'][:5]])
                                if len(request['detection_info']) > 5:
                                    detection_text += f"\n... dan {len(request['detection_info'])-5} objek lainnya"
                                
                                stats_text = (
                                    f"\n📊 Statistik:\n"
                                    f"• Waktu Deteksi: {request['detection_time']:.2f} detik\n"
                                    f"• Total Objek: {request['total_objects']}\n"
                                    f"• Rata-rata Akurasi: {request['average_confidence']:.2f}%"
                                )
                                message = f"🔍 Hasil Deteksi:\n{detection_text}{stats_text}"
                                await application.bot.send_message(chat_id=chat_id, text=message)
                            else:
                                await application.bot.send_message(
                                    chat_id=chat_id, 
                                    text="⚠️ Tidak ada daun terong yang terdeteksi."
                                )
                            
                            logger.info(f"Hasil dikirim ke Telegram: {chat_id}")
                        except Exception as e:
                            logger.error(f"Error mengirim ke {chat_id}: {e}")
                except Exception as e:
                    logger.exception(f"Error dalam queue Telegram: {e}")
                    await asyncio.sleep(1.0)
        
        # Jalankan background task
        loop.create_task(process_telegram_queue())
        
        # Jalankan bot dengan mekanisme retry yang lebih baik
        max_retries = 10
        retry_delay = 2  # detik
        
        for retry in range(max_retries):
            try:
                logger.info(f"Menjalankan bot Telegram (percobaan {retry + 1}/{max_retries})...")
                application.run_polling(
                    drop_pending_updates=True,
                    close_loop=False
                )
                break  # Keluar dari loop jika berhasil
            except Exception as e:
                logger.error(f"Error saat menjalankan bot Telegram (percobaan {retry + 1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    logger.info(f"Mencoba lagi dalam {retry_delay} detik...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.critical(f"Gagal menjalankan bot Telegram setelah {max_retries} percobaan")
    except Exception as e:
        logger.critical(f"Error kritis saat menjalankan bot Telegram: {str(e)}")
        logger.critical(f"Traceback: {traceback.format_exc()}")

# ======================
# Health Check Endpoint
# ======================

@app.route('/health')
def health_check():
    """Endpoint untuk memeriksa kesehatan aplikasi."""
    try:
        # Periksa komponen kritis
        model_loaded = 'model' in globals()
        web_history_loaded = len(web_history) >= 0
        telegram_status = bool(TELEGRAM_BOT_TOKEN)
        
        status = {
            'status': 'healthy' if all([model_loaded, web_history_loaded]) else 'degraded',
            'components': {
                'model': 'loaded' if model_loaded else 'not loaded',
                'web_history': 'loaded' if web_history_loaded else 'not loaded',
                'telegram': 'connected' if telegram_status else 'disconnected'
            }
        }
        
        return jsonify(status), 200 if status['status'] == 'healthy' else 503
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/live')
def live():
    """Tampilkan halaman deteksi langsung dengan error handling yang lebih baik."""
    try:
        # Periksa ketersediaan kamera sebelum menampilkan halaman
        camera_status = check_camera_availability()
        return render_template('live.html', camera_available=camera_status)
    except Exception as e:
        logger.error(f"Error saat memuat halaman live: {e}")
        return render_template('error.html', error="Gagal memuat halaman deteksi langsung"), 500

def check_camera_availability():
    """Periksa ketersediaan kamera dengan lebih aman."""
    try:
        # Coba beberapa indeks kamera umum
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return True
            cap.release()
        return False
    except Exception as e:
        logger.warning(f"Error saat memeriksa ketersediaan kamera: {e}")
        return False

if __name__ == '__main__':
    # Coba muat model YOLO dengan penanganan error yang lebih baik
    try:
        model = YOLO('best.pt')
        logger.info("Model YOLOv8 berhasil dimuat")
    except Exception as e:
        logger.critical(f"Gagal memuat model YOLOv8: {str(e)}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        logger.critical("Pastikan file 'best.pt' ada di direktori yang benar")
        # Tidak raise exception agar aplikasi tetap berjalan, hanya dengan fitur terbatas
    
    # Jalankan bot Telegram di thread terpisah jika token tersedia
    if TELEGRAM_BOT_TOKEN:
        telegram_thread = threading.Thread(target=run_telegram_bot, daemon=True)
        telegram_thread.start()
        logger.info("Bot Telegram dimulai dalam thread terpisah")
    else:
        logger.warning("Token Telegram tidak disetel. Bot Telegram tidak akan berjalan.")
    
    # Jalankan aplikasi Flask dengan konfigurasi yang lebih aman
    try:
        app.run(
            host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)),
            debug=False,
            threaded=True,
            use_reloader=False,
            ssl_context=None
        )
    except Exception as e:
        logger.critical(f"Error kritis saat menjalankan aplikasi Flask: {str(e)}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        # Coba jalankan di port alternatif jika port 5000 terblokir
        try:
            logger.info("Mencoba menjalankan di port alternatif 5001...")
            app.run(
                host='0.0.0.0',
                port=5001,
                debug=False,
                threaded=True,
                use_reloader=False
            )
        except Exception as e2:
            logger.critical(f"Error saat mencoba port alternatif: {str(e2)}")
            logger.critical(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)