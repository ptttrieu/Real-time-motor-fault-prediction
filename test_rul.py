"""
Test RUL Estimator — chạy trong Terminal 2
Terminal 1 phải đang chạy predict_server.py
"""
import requests
import numpy as np
import json
import time

URL = 'http://localhost:5000/predict'

print("=" * 60)
print("  TEST RUL ESTIMATOR")
print("  Mô phỏng motor đang xấu dần (35 requests)")
print("=" * 60)

for i in range(35):
    # RMS tăng dần theo thời gian — mô phỏng motor xuống cấp
    rms_base = 0.05 + i * 0.015
    noise    = np.random.randn(512) * rms_base
    samples  = noise.tolist()

    try:
        r = requests.post(URL, json={'samples': samples, 'fs': 1600}, timeout=5)
        d = r.json()
    except Exception as e:
        print(f"[{i+1:02d}] Lỗi kết nối: {e}")
        continue

    rul        = d.get('rul', {})
    fault      = d.get('fault', '—')
    rms_val    = d.get('rms', 0)
    health     = rul.get('health_score', 100)
    rul_mins   = rul.get('rul_minutes')
    status     = rul.get('status', '—')
    warn_level = rul.get('warning_level', 0)
    rec        = rul.get('recommendation', '')

    rul_str = f"{rul_mins}min" if rul_mins is not None else "N/A"

    warn_icon = ['  ', '👁 ', '⚠️ ', '🚨'][warn_level]

    print(
        f"{warn_icon}[{i+1:02d}] "
        f"RMS={rms_val:.4f}  "
        f"Fault={fault:20s}  "
        f"Health={health:5.1f}%  "
        f"RUL={rul_str:>8}  "
        f"Status={status}"
    )

    if warn_level >= 2:
        print(f"       → {rec}")

    time.sleep(0.3)

print("\n" + "=" * 60)
print("  XONG! Kiểm tra Dashboard để thấy kết quả realtime.")
print("=" * 60)