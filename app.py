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

# Load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables only.")
    print("   Install with: pip install python-dotenv")

app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)

# Supabase configuration from environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

# Validate required environment variables
if not SUPABASE_URL:
    raise ValueError("❌ SUPABASE_URL environment variable is required!")
if not SUPABASE_SERVICE_KEY:
    raise ValueError("❌ SUPABASE_SERVICE_KEY environment variable is required!")

print(f"🔗 Supabase URL: {SUPABASE_URL}")
print(f"🔑 Service Key: {'*' * 20}...{SUPABASE_SERVICE_KEY[-10:] if len(SUPABASE_SERVICE_KEY) > 10 else '*' * len(SUPABASE_SERVICE_KEY)}")

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("✅ Supabase client initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize Supabase client: {e}")
    raise

# In-memory storage for tracking IP usage with thread safety
ip_usage_tracker = defaultdict(list)
ip_lock = threading.Lock()
MAX_ANONYMOUS_UPLOADS = 3

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, User-Uid"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# Initialize database table
def init_database():
    """Create the scan_history table if it doesn't exist"""
    try:
        # Check if table exists, if not create it
        result = supabase.table('scan_history').select("*").limit(1).execute()
        print("Database table exists and is accessible")
    except Exception as e:
        print(f"Table might not exist or RLS not configured properly: {e}")
        print("Please run the SQL commands in Supabase SQL Editor to set up RLS")

def get_authenticated_supabase_client(user_uid):
    """Get Supabase client with user authentication"""
    try:
        # Create a new client instance for this user
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        
        # Set the user context for RLS
        # Note: This simulates the user being authenticated
        client.auth.set_session({
            'access_token': f'mock-token-{user_uid}',
            'user': {'id': user_uid}
        })
        
        return client
    except Exception as e:
        print(f"Error creating authenticated client: {e}")
        return None

# Load model
def load_model_from_drive():
    MODEL_PATH = "best_model.h5"
    FILE_ID = "1EW4wiNPtZKl2iLAmV0SjMs6OFPBXB8OR"  # Ganti dengan ID model-mu

    if not os.path.exists(MODEL_PATH):
        print("⏳ Downloading model from Google Drive...")
        try:
            import gdown
            gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)
            print("✅ Model downloaded!")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            raise
    return load_model(MODEL_PATH)

print("⏳ Loading model...")
model = load_model_from_drive()
print("✅ Model loaded successfully!")

class_names = [
    'Baterai', 'Daun', 'Elektronik', 'Kaca', 'Kardus',
    'Kertas', 'Lampu', 'Logam', 'Pakaian',
    'Plastik', 'Sampah Makanan', 'Sterofom'
]

group_map = {
    'Logam': 'Anorganik',
    'Plastik': 'Anorganik',
    'Pakaian': 'Anorganik',
    'Kaca': 'Anorganik',
    'Sterofom': 'Anorganik',
    'Daun': 'Organik',
    'Kardus': 'Organik',
    'Sampah Makanan': 'Organik',
    'Kertas': 'Organik',
    'Baterai': 'B3',
    'Lampu': 'B3',
    'Elektronik': 'B3'
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

def get_ip_address():
    """Get the real IP address, falling back to request.remote_addr"""
    if request.headers.getlist("X-Forwarded-For"):
        return request.headers.getlist("X-Forwarded-For")[0].split(',')[0].strip()
    return request.remote_addr

def clean_old_entries():
    """Remove entries older than 24 hours from ip_usage_tracker"""
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=24)

    with ip_lock:
        for ip in list(ip_usage_tracker.keys()):
            ip_usage_tracker[ip] = [
                timestamp for timestamp in ip_usage_tracker[ip]
                if timestamp > cutoff_time
            ]
            if not ip_usage_tracker[ip]:
                del ip_usage_tracker[ip]

def check_and_record_upload(ip_address):
    """Thread-safe check and record in single operation"""
    with ip_lock:
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)

        ip_usage_tracker[ip_address] = [
            timestamp for timestamp in ip_usage_tracker[ip_address]
            if timestamp > cutoff_time
        ]

        current_uploads = len(ip_usage_tracker[ip_address])

        if current_uploads >= MAX_ANONYMOUS_UPLOADS:
            return False, current_uploads

        ip_usage_tracker[ip_address].append(current_time)
        return True, current_uploads + 1

def save_scan_to_database(user_uid, waste_type, category, description, disposal_steps, ip_address):
    """Save scan result to Supabase database with RLS"""
    try:
        # Use service role to insert data but set the correct user_uid
        result = supabase.table('scan_history').insert({
            'user_uid': user_uid,
            'waste_type': waste_type,
            'category': category,
            'description': description,
            'disposal_steps': disposal_steps,
            'ip_address': ip_address
        }).execute()
        
        if result.data:
            scan_data = result.data[0]
            return {
                'id': scan_data['id'],
                'scan_date': scan_data['scan_date']
            }
        return None
        
    except Exception as e:
        print(f"Error saving to database: {e}")
        return None

def get_user_scan_history(user_uid, limit=50, offset=0):
    """Get scan history for a specific user with RLS"""
    try:
        # Use service role but filter by user_uid for RLS compliance
        result = supabase.table('scan_history')\
            .select('id, waste_type, category, description, disposal_steps, scan_date')\
            .eq('user_uid', user_uid)\
            .order('scan_date', desc=True)\
            .range(offset, offset + limit - 1)\
            .execute()
        
        if result.data:
            history = []
            for row in result.data:
                history.append({
                    'id': row['id'],
                    'waste_type': row['waste_type'],
                    'category': row['category'],
                    'description': row['description'],
                    'disposal_steps': row['disposal_steps'],
                    'scan_date': row['scan_date']
                })
            return history
        return []
        
    except Exception as e:
        print(f"Error fetching scan history: {e}")
        return []

def get_user_scan_statistics(user_uid):
    """Get statistics for user scans with RLS"""
    try:
        # Total scans
        total_result = supabase.table('scan_history')\
            .select('id', count='exact')\
            .eq('user_uid', user_uid)\
            .execute()
        
        total_scans = total_result.count if total_result.count is not None else 0
        
        # Category breakdown
        category_result = supabase.table('scan_history')\
            .select('category')\
            .eq('user_uid', user_uid)\
            .execute()
        
        category_breakdown = {}
        if category_result.data:
            for row in category_result.data:
                category = row['category']
                category_breakdown[category] = category_breakdown.get(category, 0) + 1
        
        # Convert to list format
        category_list = [{'category': k, 'count': v} for k, v in category_breakdown.items()]
        category_list.sort(key=lambda x: x['count'], reverse=True)
        
        # Recent activity (last 7 days)
        recent_result = supabase.table('scan_history')\
            .select('scan_date')\
            .eq('user_uid', user_uid)\
            .gte('scan_date', (datetime.now() - timedelta(days=7)).isoformat())\
            .execute()
        
        daily_activity = {}
        if recent_result.data:
            for row in recent_result.data:
                date_str = row['scan_date'][:10]  # Get date part only
                daily_activity[date_str] = daily_activity.get(date_str, 0) + 1
        
        recent_activity = [{'date': k, 'count': v} for k, v in daily_activity.items()]
        recent_activity.sort(key=lambda x: x['date'], reverse=True)
        
        return {
            'total_scans': total_scans,
            'category_breakdown': category_list,
            'recent_activity': recent_activity
        }
        
    except Exception as e:
        print(f"Error fetching statistics: {e}")
        return None

def prepare_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return 'API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    user_uid = request.headers.get('User-Uid')
    ip_user = get_ip_address()

    print(f"Request from IP: {ip_user}, User-Uid: {user_uid}")

    # Check upload limits for anonymous users
    if not user_uid:
        allowed, current_count = check_and_record_upload(ip_user)
        if not allowed:
            return jsonify({
                'error': f'Upload anonim dibatasi maksimal {MAX_ANONYMOUS_UPLOADS} kali dalam 24 jam.',
                'current_uploads': current_count,
                'message': 'Silakan coba lagi dalam 24 jam atau daftar untuk akses tanpa batas.',
                'ip_address': ip_user
            }), 429

    try:
        # Image processing and prediction
        img_bytes = file.read()
        img_ready = prepare_image_from_bytes(img_bytes)

        preds = model.predict(img_ready)
        pred_idx = np.argmax(preds, axis=1)[0]
        pred_class = class_names[pred_idx]
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

        # Save to database if user has UID
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
                response_data['scan_id'] = db_result['id']
                response_data['scan_date'] = db_result['scan_date']
            else:
                response_data['warning'] = 'Prediksi berhasil, tetapi gagal menyimpan ke database'

        # Add debug info for anonymous users
        if not user_uid:
            with ip_lock:
                current_uploads = len(ip_usage_tracker[ip_user])
            response_data['debug_info'] = {
                'remaining_uploads': MAX_ANONYMOUS_UPLOADS - current_uploads,
                'ip_address': ip_user
            }

        return jsonify(response_data)

    except Exception as e:
        # Rollback for anonymous users
        if not user_uid:
            with ip_lock:
                if ip_user in ip_usage_tracker and ip_usage_tracker[ip_user]:
                    ip_usage_tracker[ip_user].pop()

        print(f"Terjadi kesalahan saat prediksi: {e}")
        return jsonify({
            'error': 'Terjadi kesalahan saat memproses gambar',
            'details': str(e)
        }), 500

@app.route('/history/<user_uid>', methods=['GET'])
def get_scan_history(user_uid):
    """Get scan history for a user"""
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Validate limit and offset
        limit = min(max(limit, 1), 100)  # Between 1 and 100
        offset = max(offset, 0)
        
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
        return jsonify({
            'error': 'Terjadi kesalahan saat mengambil riwayat scan',
            'details': str(e)
        }), 500

@app.route('/statistics/<user_uid>', methods=['GET'])
def get_scan_stats(user_uid):
    """Get scan statistics for a user"""
    try:
        stats = get_user_scan_statistics(user_uid)
        
        if stats is None:
            return jsonify({
                'error': 'Terjadi kesalahan saat mengambil statistik'
            }), 500
        
        return jsonify({
            'user_uid': user_uid,
            'statistics': stats
        })
    
    except Exception as e:
        print(f"Error in get_scan_stats: {e}")
        return jsonify({
            'error': 'Terjadi kesalahan saat mengambil statistik',
            'details': str(e)
        }), 500

@app.route('/history/<user_uid>/<int:scan_id>', methods=['DELETE'])
def delete_scan_history(user_uid, scan_id):
    """Delete a specific scan from history"""
    try:
        result = supabase.table('scan_history')\
            .delete()\
            .eq('id', scan_id)\
            .eq('user_uid', user_uid)\
            .execute()
        
        if result.data:
            return jsonify({
                'message': 'Riwayat scan berhasil dihapus',
                'deleted_id': scan_id
            })
        else:
            return jsonify({
                'error': 'Riwayat scan tidak ditemukan atau Anda tidak memiliki akses'
            }), 404
    
    except Exception as e:
        print(f"Error deleting scan history: {e}")
        return jsonify({
            'error': 'Terjadi kesalahan saat menghapus riwayat scan',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # Initialize database when starting the app
    init_database()
    app.run(host='0.0.0.0', port=7860)
