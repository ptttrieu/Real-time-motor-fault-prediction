"""
=======================================================
  RUL ESTIMATOR — Remaining Useful Life
=======================================================
Phương pháp:
  1. Trend Analysis trên RMS (hồi quy tuyến tính)
  2. Confidence Decay của Random Forest
  3. Kết hợp 2 chỉ số → RUL score tổng hợp

Tích hợp vào predict_server.py:
  from rul_estimator import RULEstimator
  rul = RULEstimator()
  rul_result = rul.update(rms, confidence, label)
"""

import numpy as np
from collections import deque
import datetime


class RULEstimator:
    """
    Ước lượng RUL dựa trên:
      - RMS trend: hồi quy tuyến tính trên cửa sổ N mẫu gần nhất
      - Confidence decay: khi RF kém tự tin hơn → sắp chuyển trạng thái
    """

    # Ngưỡng RMS coi là hỏng hoàn toàn (g) — chỉnh theo motor thực tế
    RMS_FAILURE_THRESHOLD = 2.0

    # Ngưỡng confidence bình thường (%)
    CONF_HEALTHY_THRESHOLD = 90.0

    # Cửa sổ lịch sử
    WINDOW = 30   # số mẫu để tính trend

    # Thời gian giữa 2 lần đo (giây) — khớp với tốc độ ESP32 gửi
    SAMPLE_INTERVAL_S = 2.0

    def __init__(self):
        self.rms_history    = deque(maxlen=self.WINDOW)
        self.conf_history   = deque(maxlen=self.WINDOW)
        self.label_history  = deque(maxlen=self.WINDOW)
        self.time_history   = deque(maxlen=self.WINDOW)
        self.sample_count   = 0

    # ----------------------------------------------------------
    def update(self, rms: float, confidence: float, label: int) -> dict:
        """
        Cập nhật 1 mẫu mới và trả về dict chứa thông tin RUL.
        Gọi hàm này mỗi lần predict xong.
        """
        self.rms_history.append(rms)
        self.conf_history.append(confidence)
        self.label_history.append(label)
        self.time_history.append(datetime.datetime.now())
        self.sample_count += 1

        result = {
            "rul_samples":       None,   # số mẫu còn lại
            "rul_minutes":       None,   # thời gian còn lại (phút)
            "rul_percent":       100.0,  # sức khỏe %
            "rms_trend":         0.0,    # độ dốc RMS (g/sample)
            "conf_trend":        0.0,    # độ dốc confidence (%/sample)
            "health_score":      100.0,  # điểm sức khỏe tổng hợp 0-100
            "status":            "Healthy",
            "warning_level":     0,      # 0=OK 1=Watch 2=Warning 3=Critical
            "recommendation":    "Hoạt động bình thường",
            "n_samples_used":    len(self.rms_history),
        }

        # Cần ít nhất 5 mẫu để tính trend
        if len(self.rms_history) < 5:
            result["status"] = "Đang thu thập dữ liệu..."
            return result

        rms_arr  = np.array(self.rms_history)
        conf_arr = np.array(self.conf_history)
        x        = np.arange(len(rms_arr), dtype=float)

        # --------------------------------------------------
        # 1. RMS TREND — hồi quy tuyến tính
        # --------------------------------------------------
        rms_slope, rms_intercept = np.polyfit(x, rms_arr, 1)
        result["rms_trend"] = round(float(rms_slope), 6)

        # RUL từ RMS: bao nhiêu bước nữa đến ngưỡng failure
        current_rms = rms_arr[-1]
        if rms_slope > 1e-6:
            rul_rms = (self.RMS_FAILURE_THRESHOLD - current_rms) / rms_slope
            rul_rms = max(0.0, float(rul_rms))
        else:
            rul_rms = float('inf')   # RMS không tăng → không ước được

        # --------------------------------------------------
        # 2. CONFIDENCE DECAY
        # --------------------------------------------------
        conf_slope, _ = np.polyfit(x, conf_arr, 1)
        result["conf_trend"] = round(float(conf_slope), 4)

        current_conf = conf_arr[-1]
        if conf_slope < -0.1:   # confidence đang giảm
            rul_conf = (current_conf - self.CONF_HEALTHY_THRESHOLD) / abs(conf_slope)
            rul_conf = max(0.0, float(rul_conf))
        else:
            rul_conf = float('inf')

        # --------------------------------------------------
        # 3. HEALTH SCORE tổng hợp (0–100)
        # --------------------------------------------------
        # Thành phần RMS: 0% khi RMS >= threshold, 100% khi RMS = 0
        rms_score = max(0.0, 100.0 * (1.0 - current_rms / self.RMS_FAILURE_THRESHOLD))

        # Thành phần confidence
        conf_score = max(0.0, min(100.0, current_conf))

        # Thành phần trend penalty
        trend_penalty = min(50.0, max(0.0, rms_slope * 5000))

        # Kết hợp: 50% RMS score + 40% conf + 10% trend
        health = 0.50 * rms_score + 0.40 * conf_score - trend_penalty
        health = max(0.0, min(100.0, health))
        result["health_score"] = round(health, 1)

        # --------------------------------------------------
        # 4. RUL kết hợp
        # --------------------------------------------------
        # Lấy min của 2 ước tính (worst case)
        rul_samples = min(
            rul_rms  if rul_rms  != float('inf') else 9999,
            rul_conf if rul_conf != float('inf') else 9999,
        )
        if rul_samples >= 9999:
            rul_samples = None
            rul_minutes = None
            rul_pct     = 100.0
        else:
            rul_samples = int(rul_samples)
            rul_minutes = round(rul_samples * self.SAMPLE_INTERVAL_S / 60.0, 1)
            # RUL % dựa trên health score
            rul_pct = round(health, 1)

        result["rul_samples"] = rul_samples
        result["rul_minutes"] = rul_minutes
        result["rul_percent"] = rul_pct if rul_pct is not None else 100.0

        # --------------------------------------------------
        # 5. WARNING LEVEL & STATUS
        # --------------------------------------------------
        # Dựa trên label hiện tại + health score
        fault_labels = list(self.label_history)
        recent_faults = sum(1 for l in fault_labels[-5:] if l != 0)

        if label == 0 and health >= 80:
            level = 0
            status = "Healthy"
            rec    = "Hoạt động bình thường. Không cần bảo trì."
        elif health >= 60 or recent_faults <= 1:
            level = 1
            status = "Watch"
            rec    = "Theo dõi thêm. Lên lịch kiểm tra định kỳ."
        elif health >= 40 or recent_faults <= 3:
            level = 2
            status = "Warning"
            rec    = "⚠️ Cần kiểm tra sớm. Lên lịch bảo trì trong 48h."
        else:
            level = 3
            status = "Critical"
            rec    = "🚨 Nguy hiểm! Dừng máy và bảo trì ngay lập tức."

        result["status"]         = status
        result["warning_level"]  = level
        result["recommendation"] = rec

        return result

    def reset(self):
        """Reset toàn bộ lịch sử (sau khi bảo trì xong)."""
        self.rms_history.clear()
        self.conf_history.clear()
        self.label_history.clear()
        self.time_history.clear()
        self.sample_count = 0
