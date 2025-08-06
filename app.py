from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image
import io
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import os
from supabase import create_client, Client
import json
import gzip
import shutil
import gdown

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Using system environment variables.")
except Exception:
    print("ℹ️ No .env file found. Using system environment variables")

app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Missing Supabase configuration!")

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("✅ Supabase client initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize Supabase client: {e}")
    raise

# In-memory storage for IP tracking
ip_usage_tracker = defaultdict(list)
ip_lock = threading.Lock()
MAX_ANONYMOUS_UPLOADS = 3

# Model loading with compression handling
def load_compressed_model():
    MODEL_PATH = "best_model.h5"
    COMPRESSED_PATH = "best_model.h5.gz"
    
    if not os.path.exists(MODEL_PATH):
        if os.path.exists(COMPRESSED_PATH):
            print("⏳ Decompressing model file...")
            with gzip.open(COMPRESSED_PATH, 'rb') as f_in:
                with open(MODEL_PATH, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            # Alternatif: Download dari Google Drive jika diperlukan
            MODEL_URL = "https://drive.google.com/file/d/1sdxe5h9HL5-GcUb24w3ljLH_vIBtgMmE/view?usp=drive_link"  # Ganti dengan ID Anda
            print("⏳ Downloading model file...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
    return load_model(MODEL_PATH)

print("⏳ Loading model...")
model = load_compressed_model()
print("✅ Model loaded successfully!")

# Data definitions
class_names = [
    'Baterai', 'Daun', 'Elektronik', 'Kaca', 'Kardus',
    'Kertas', 'Lampu', 'Logam', 'Pakaian',
    'Plastik', 'Sampah Makanan', 'Sterofom'
]

group_map = {
    'Logam': 'Anorganik', 'Plastik': 'Anorganik', 'Pakaian': 'Anorganik', 
    'Kaca': 'Anorganik', 'Sterofom': 'Anorganik', 'Daun': 'Organik',
    'Kardus': 'Organik', 'Sampah Makanan': 'Organik', 'Kertas': 'Organik',
    'Baterai': 'B3', 'Lampu': 'B3', 'Elektronik': 'B3'
}

kategori_info = {
    'Organik': {
        'description': 'Sampah organik berasal dari bahan-bahan alami yang dapat terurai secara biologis seperti sisa makanan, daun, dan ranting.',
        'disposalSteps': [
            'Pisahkan dari sampah anorganik',
            'Buang di tempat sampah organik',
            'Bisa digunakan untuk kompos jika memungkinkan'
        ]
    },
    'Anorganik': {
        'description': 'Sampah anorganik adalah sampah yang tidak dapat terurai secara alami seperti plastik, kaca, dan logam.',
        'disposalSteps': [
            'Pisahkan berdasarkan jenis material (plastik, kaca, logam)',
            'Cuci bersih jika terkontaminasi makanan',
            'Buang di tempat sampah daur ulang atau tempat sampah anorganik'
        ]
    },
    'B3': {
        'description': 'Sampah B3 mengandung bahan berbahaya seperti baterai, elektronik, dan bahan kimia yang memerlukan penanganan khusus.',
        'disposalSteps': [
            'Jangan dibuang bersama sampah biasa',
            'Bawa ke tempat pengumpulan sampah B3',
            'Hubungi layanan pengelolaan limbah berbahaya di daerah Anda'
        ]
    }
}

# Helper functions
def get_ip_address():
    if request.headers.getlist("X-Forwarded-For"):
        return request.headers.getlist("X-Forwarded-For")[0].split(',')[0].strip()
    return request.remote_addr

def check_and_record_upload(ip_address):
    with ip_lock:
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)

        ip_usage_tracker[ip_address] = [
            t for t in ip_usage_tracker[ip_address] if t > cutoff_time
        ]

        current_uploads = len(ip_usage_tracker[ip_address])
        if current_uploads >= MAX_ANONYMOUS_UPLOADS:
            return False, current_uploads

        ip_usage_tracker[ip_address].append(current_time)
        return True, current_uploads + 1

def prepare_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def save_scan_to_database(user_uid, waste_type, category, description, disposal_steps, ip_address):
    try:
        result = supabase.table('scan_history').insert({
            'user_uid': user_uid,
            'waste_type': waste_type,
            'category': category,
            'description': description,
            'disposal_steps': disposal_steps,
            'ip_address': ip_address
        }).execute()
        
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"Error saving to database: {e}")
        return None

def get_user_scan_history(user_uid, limit=50, offset=0):
    try:
        result = supabase.table('scan_history') \
            .select('*') \
            .eq('user_uid', user_uid) \
            .order('scan_date', desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
        
        return result.data or []
    except Exception as e:
        print(f"Error fetching scan history: {e}")
        return []

def get_user_scan_statistics(user_uid):
    try:
        # Total scans
        total_result = supabase.table('scan_history') \
            .select('id', count='exact') \
            .eq('user_uid', user_uid) \
            .execute()
        
        # Category breakdown
        category_result = supabase.table('scan_history') \
            .select('category') \
            .eq('user_uid', user_uid) \
            .execute()
        
        category_breakdown = {}
        for row in (category_result.data or []):
            category_breakdown[row['category']] = category_breakdown.get(row['category'], 0) + 1
        
        # Recent activity
        recent_result = supabase.table('scan_history') \
            .select('scan_date') \
            .eq('user_uid', user_uid) \
            .gte('scan_date', (datetime.now() - timedelta(days=7)).isoformat()) \
            .execute()
        
        daily_activity = {}
        for row in (recent_result.data or []):
            date_str = row['scan_date'][:10]
            daily_activity[date_str] = daily_activity.get(date_str, 0) + 1
        
        return {
            'total_scans': total_result.count or 0,
            'category_breakdown': [{'category': k, 'count': v} for k, v in category_breakdown.items()],
            'recent_activity': [{'date': k, 'count': v} for k, v in daily_activity.items()]
        }
    except Exception as e:
        print(f"Error fetching statistics: {e}")
        return None

# API Endpoints
@app.route('/')
def index():
    return 'API Sampah Pintar is running!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    user_uid = request.headers.get('User-Uid')
    ip_user = get_ip_address()

    if not user_uid:
        allowed, current_count = check_and_record_upload(ip_user)
        if not allowed:
            return jsonify({
                'error': f'Upload anonim dibatasi {MAX_ANONYMOUS_UPLOADS}x/24 jam',
                'current_uploads': current_count
            }), 429

    try:
        img_ready = prepare_image_from_bytes(file.read())
        pred_class = class_names[np.argmax(model.predict(img_ready), axis=1)[0]]
        pred_group = group_map.get(pred_class, 'Unknown')
        info = kategori_info.get(pred_group, {
            'description': 'Kategori tidak diketahui.',
            'disposalSteps': ['Silakan periksa ulang jenis sampah ini.']
        })

        response_data = {
            'Sampah': pred_class,
            'Kategori': pred_group,
            'Deskripsi': info['description'],
            'LangkahPembuangan': info['disposalSteps']
        }

        if user_uid:
            db_result = save_scan_to_database(
                user_uid=user_uid,
                waste_type=pred_class,
                category=pred_group,
                description=info['description'],
                disposal_steps=info['disposalSteps'],
                ip_address=ip_user
            )
            if db_result:
                response_data.update({
                    'scan_id': db_result['id'],
                    'scan_date': db_result['scan_date']
                })

        return jsonify(response_data)

    except Exception as e:
        print(f"Error during prediction: {e}")
        if not user_uid and ip_user in ip_usage_tracker:
            with ip_lock:
                ip_usage_tracker[ip_user].pop()
        return jsonify({'error': str(e)}), 500

@app.route('/history/<user_uid>', methods=['GET'])
def get_scan_history(user_uid):
    try:
        limit = min(max(request.args.get('limit', 50, type=int), 1), 100)
        offset = max(request.args.get('offset', 0, type=int), 0)
        
        history = get_user_scan_history(user_uid, limit, offset)
        return jsonify({
            'user_uid': user_uid,
            'history': history,
            'count': len(history),
            'limit': limit,
            'offset': offset
        })
    except Exception as e:
        print(f"Error in get_scan_history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/statistics/<user_uid>', methods=['GET'])
def get_scan_stats(user_uid):
    try:
        stats = get_user_scan_statistics(user_uid)
        return jsonify({'user_uid': user_uid, 'statistics': stats}) if stats else \
               jsonify({'error': 'Failed to get statistics'}), 500
    except Exception as e:
        print(f"Error in get_scan_stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history/<user_uid>/<int:scan_id>', methods=['DELETE'])
def delete_scan_history(user_uid, scan_id):
    try:
        result = supabase.table('scan_history') \
            .delete() \
            .eq('id', scan_id) \
            .eq('user_uid', user_uid) \
            .execute()
        
        return jsonify({
            'message': 'Riwayat scan berhasil dihapus',
            'deleted_id': scan_id
        }) if result.data else \
        jsonify({'error': 'Data tidak ditemukan'}), 404
    except Exception as e:
        print(f"Error deleting scan history: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)