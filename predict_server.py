"""
=======================================================
  PREDICT SERVER — REST API + Firebase + RUL
=======================================================
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
from scipy.fft import fft, fftfreq
import firebase_admin
from firebase_admin import credentials, db
import os, time, datetime
from rul_estimator import RULEstimator

# ============================================================
# CẤU HÌNH
# ============================================================
MODEL_PATH    = 'data/rf_model.pkl'
FIREBASE_CRED = 'firebase_key.json'
FIREBASE_URL  = 'https://proj-1-1d7a8-default-rtdb.asia-southeast1.firebasedatabase.app'
HOST          = '0.0.0.0'
PORT          = 5000
SEGMENT_LEN   = 512
MAX_HISTORY   = 100

RPM       = 1480
F_ROT     = RPM / 60
F_SUPPLY  = 50
N_BALLS   = 8
BALL_DIA  = 6.75
PITCH_DIA = 28.5
BPFO = (N_BALLS / 2) * F_ROT * (1 - BALL_DIA / PITCH_DIA)
BPFI = (N_BALLS / 2) * F_ROT * (1 + BALL_DIA / PITCH_DIA)

CLASS_NAMES = [
    'Normal', 'Unbalance', 'Misalignment', 'Looseness',
    'Broken Rotor Bar', 'Bearing Fault', 'Shorted Winding',
]

STATUS_KEYS = {
    0:'normal', 1:'unbalance', 2:'misalignment', 3:'looseness',
    4:'broken_rotor_bar', 5:'bearing_fault', 6:'shorted_winding',
}

# ============================================================
# KHỞI ĐỘNG
# ============================================================
print("=" * 52)
print("  MOTOR FAULT PREDICT SERVER  (7 classes + RUL)")
print("=" * 52)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy: {MODEL_PATH}")

rf  = joblib.load(MODEL_PATH)
rul = RULEstimator()
print(f"✓ Model loaded  : {MODEL_PATH}")
print(f"✓ RUL Estimator : sẵn sàng")

# ============================================================
# FIREBASE
# ============================================================
firebase_ok = False
try:
    if not os.path.exists(FIREBASE_CRED):
        print(f"⚠️  Không tìm thấy {FIREBASE_CRED}")
    else:
        cred = credentials.Certificate(FIREBASE_CRED)
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        firebase_ok = True
        print(f"✓ Firebase      : proj-1-1d7a8 (connected)")
        status_ref = db.reference('/status')
        if status_ref.get() is None:
            status_ref.set({k: 0 for k in STATUS_KEYS.values()})
except Exception as e:
    print(f"⚠️  Firebase error: {e}")

# ============================================================
# EXTRACT FEATURES
# ============================================================
def extract_features(signal, fs):
    signal = np.array(signal, dtype=float) - np.mean(signal)
    N = len(signal)
    if N < 128:
        return None

    rms   = np.sqrt(np.mean(signal**2))
    peak  = np.max(np.abs(signal))
    crest = peak / (rms + 1e-10)
    std   = np.std(signal)
    kurt  = np.mean((signal - np.mean(signal))**4) / (std**4 + 1e-10)
    skew  = np.mean((signal - np.mean(signal))**3) / (std**3 + 1e-10)
    shape = rms / (np.mean(np.abs(signal)) + 1e-10)

    freqs   = fftfreq(N, 1.0 / fs)[:N // 2]
    fft_mag = np.abs(fft(signal))[:N // 2] * 2.0 / N

    def peak_at(fc, bw=3.0):
        mask = (freqs >= fc - bw) & (freqs <= fc + bw)
        return float(np.max(fft_mag[mask])) if mask.any() else 0.0

    def band_energy(f_lo, f_hi):
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        return float(np.sum(fft_mag[mask]**2))

    p_1x   = peak_at(F_ROT);     p_2x  = peak_at(2*F_ROT)
    p_3x   = peak_at(3*F_ROT);   p_4x  = peak_at(4*F_ROT)
    p_5x   = peak_at(5*F_ROT)
    p_50   = peak_at(F_SUPPLY);   p_100 = peak_at(2*F_SUPPLY)
    p_150  = peak_at(3*F_SUPPLY)
    p_bpfo = peak_at(BPFO);       p_bpfi = peak_at(BPFI)

    harm_ratio = (p_2x + p_3x + p_4x) / (p_1x + 1e-10)
    side_ratio = (p_50  + p_100)       / (p_1x + 1e-10)

    e0 = band_energy(0,50);    e1 = band_energy(50,200)
    e2 = band_energy(200,500); e3 = band_energy(500,2000)
    et = e0+e1+e2+e3+1e-10

    return [rms, peak, crest, kurt, skew, shape,
            p_1x, p_2x, p_3x, p_4x, p_5x,
            p_50, p_100, p_150, p_bpfo, p_bpfi,
            harm_ratio, side_ratio,
            e0/et, e1/et, e2/et, e3/et]

# ============================================================
# FIREBASE UPLOAD
# ============================================================
def push_to_firebase(result: dict, rul_result: dict):
    if not firebase_ok:
        return
    try:
        label    = result['label']
        stat_key = STATUS_KEYS[label]
        ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        # /status counter
        # /status — reset tất cả về 0, chỉ set key hiện tại = 1
        new_status = {k: 0 for k in STATUS_KEYS.values()}
        new_status[stat_key] = 1
        db.reference('/status').set(new_status)

        # /current — realtime
        db.reference('/current').set({
            'fault':         result['fault'],
            'label':         label,
            'confidence':    result['confidence'],
            'rms':           result['rms'],
            'timestamp':     result['timestamp'],
            'is_fault':      label != 0,
            'is_electrical': label in [4, 6],
            'is_mechanical': label in [1, 2, 3, 5],
            'all_probs': {
                'normal':           result['all_probs'].get('Normal', 0),
                'unbalance':        result['all_probs'].get('Unbalance', 0),
                'misalignment':     result['all_probs'].get('Misalignment', 0),
                'looseness':        result['all_probs'].get('Looseness', 0),
                'broken_rotor_bar': result['all_probs'].get('Broken Rotor Bar', 0),
                'bearing_fault':    result['all_probs'].get('Bearing Fault', 0),
                'shorted_winding':  result['all_probs'].get('Shorted Winding', 0),
            },
            # RUL data
            'rul': {
                'health_score':   rul_result['health_score'],
                'rul_percent':    rul_result['rul_percent'],
                'rul_minutes':    rul_result['rul_minutes'],
                'status':         rul_result['status'],
                'warning_level':  rul_result['warning_level'],
                'recommendation': rul_result['recommendation'],
                'rms_trend':      rul_result['rms_trend'],
                'conf_trend':     rul_result['conf_trend'],
            }
        })

        # /history
        db.reference(f'/history/{ts}').set({
            'fault':        result['fault'],
            'label':        label,
            'confidence':   result['confidence'],
            'rms':          result['rms'],
            'timestamp':    result['timestamp'],
            'is_fault':     label != 0,
            'health_score': rul_result['health_score'],
            'rul_minutes':  rul_result['rul_minutes'],
            'warning_level':rul_result['warning_level'],
        })

        # Xóa lịch sử cũ
        hist_ref  = db.reference('/history')
        hist_data = hist_ref.get()
        if hist_data and len(hist_data) > MAX_HISTORY:
            for k in sorted(hist_data.keys())[:len(hist_data) - MAX_HISTORY]:
                hist_ref.child(k).delete()

    except Exception as e:
        print(f"  [Firebase ERROR] {e}")

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":   "ok",
        "firebase": firebase_ok,
        "classes":  CLASS_NAMES,
        "rul":      "enabled",
    })

@app.route('/predict', methods=['POST'])
def predict():
    t0   = time.time()
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    samples = data.get('samples')
    fs      = float(data.get('fs', 1600))

    if not samples or len(samples) < 128:
        n = len(samples) if samples else 0
        return jsonify({"error": f"Cần ít nhất 128 samples, nhận {n}"}), 400

    signal = np.array(samples[:SEGMENT_LEN], dtype=float)
    feat   = extract_features(signal, fs)
    if feat is None:
        return jsonify({"error": "Không trích được features"}), 500

    X    = np.array(feat).reshape(1, -1)
    pred = int(rf.predict(X)[0])
    prob = rf.predict_proba(X)[0]

    rms_val = float(np.sqrt(np.mean((signal - np.mean(signal))**2)))
    elapsed = round((time.time() - t0) * 1000, 1)
    ts_str  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    result = {
        "fault":      CLASS_NAMES[pred],
        "label":      pred,
        "confidence": round(float(prob[pred]) * 100, 2),
        "rms":        round(rms_val, 6),
        "all_probs":  {CLASS_NAMES[i]: round(float(prob[i])*100, 2)
                       for i in range(len(CLASS_NAMES))},
        "n_samples":  len(samples),
        "latency_ms": elapsed,
        "timestamp":  ts_str,
    }

    # Cập nhật RUL
    rul_result = rul.update(rms_val, result["confidence"], pred)
    result["rul"] = rul_result

    # Log
    icons = ['✅','⚠️ ','⚠️ ','⚠️ ','⚡','⚠️ ','⚡']
    warn_icons = ['','','⚠️ ','🚨']
    print(f"{icons[pred]} [{ts_str}]  {result['fault']:20s}  "
          f"{result['confidence']:5.1f}%  "
          f"Health:{rul_result['health_score']:5.1f}%  "
          f"RUL:{str(rul_result['rul_minutes'])+'min' if rul_result['rul_minutes'] else 'N/A':>8}  "
          f"{warn_icons[rul_result['warning_level']]} {rul_result['status']}")

    push_to_firebase(result, rul_result)
    return jsonify(result)

@app.route('/rul/reset', methods=['POST'])
def reset_rul():
    """Reset RUL sau khi bảo trì xong."""
    rul.reset()
    if firebase_ok:
        db.reference('/current/rul').update({
            'health_score':  100.0,
            'rul_percent':   100.0,
            'rul_minutes':   None,
            'status':        'Healthy',
            'warning_level': 0,
            'recommendation':'Vừa được bảo trì. Hoạt động bình thường.',
        })
    return jsonify({"status": "ok", "message": "RUL đã được reset"})

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    import socket
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = '127.0.0.1'

    print(f"\n  ESP32 POST đến : http://{local_ip}:{PORT}/predict")
    print(f"  Health check   : http://{local_ip}:{PORT}/health")
    print(f"  RUL Reset      : POST http://{local_ip}:{PORT}/rul/reset")
    print(f"  Firebase       : {'✓ BẬT' if firebase_ok else '✗ TẮT'}")
    print("\n  Đang chờ data từ ESP32...\n")
    app.run(host=HOST, port=PORT, debug=False)