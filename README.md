# ⚙️ Motor Fault Detection System

> **Real-time motor fault prediction using Random Forest + ESP32 + Firebase**  
> Course Project — Ho Chi Minh City University of Technology and Education (HCMUTE)  
> Faculty of Electrical and Electronics Engineering

---

## 👥 Team Members

| Name | Student ID |
|------|-----------|
| Nguyễn Thế Bảo | 23119050 |
| Trần Hạo Khiêm | 23119074 |
| Phùng Thanh Triều | 23119116 |

**Supervisor:** Dr. Huỳnh Thế Thiện

---

## 📌 Project Overview

This project designs and implements a **Predictive Maintenance System** for three-phase induction motors. The system collects vibration signals from a BMI160 sensor mounted on the motor, transmits data via Wi-Fi to a Python prediction server, classifies fault types using a trained Random Forest model, and visualizes results on a real-time web dashboard connected to Firebase.

---

## 🎯 Detected Fault Types (7 Classes)

| Label | Fault | Type | Dataset Source |
|-------|-------|------|---------------|
| 0 | Normal | — | MaFaulDa |
| 1 | Unbalance | Mechanical | MaFaulDa |
| 2 | Misalignment | Mechanical | MaFaulDa |
| 3 | Looseness | Mechanical | Mendeley (zx8pfhdtnb) |
| 4 | Broken Rotor Bar | Electrical | IEEE DataPort |
| 5 | Bearing Fault | Mechanical | CWRU + MaFaulDa |
| 6 | Shorted Winding | Electrical | University of Ottawa |

---

## 🏗️ System Architecture

```
BMI160 Sensor
     │
     ▼
ESP32 (Wi-Fi)
     │  HTTP POST /predict
     ▼
Python Flask Server  ──►  Random Forest Model
     │                         (22 FFT features)
     ▼
Firebase Realtime Database
     │
     ▼
Web Dashboard (index.html)
  • Real-time fault status
  • RUL (Remaining Useful Life)
  • Maintenance advice
  • Fault history
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **97.67%** |
| 5-Fold Cross-Validation | **98.27% ± 0.18%** |
| Algorithm | Random Forest (300 trees) |
| Features | 22 (FFT + time-domain) |
| Training samples | 3,865 segments |

---

## 🔧 Features Extracted (22 features)

**Time Domain:** RMS, Peak, Crest Factor, Kurtosis, Skewness, Shape Factor

**Frequency Domain:** 1×RPM, 2×RPM, 3×RPM, 4×RPM, 5×RPM, 50Hz, 100Hz, 150Hz, BPFO, BPFI, Harmonic Ratio, Sideband Ratio

**Band Energy:** 0–50Hz, 50–200Hz, 200–500Hz, 500–2kHz

---

## 🔮 Remaining Useful Life (RUL) Estimation

The system estimates RUL by combining:
1. **RMS Trend Analysis** — linear regression on rolling window of 30 samples
2. **Confidence Decay** — monitors Random Forest prediction confidence over time

**Health Score** = 50% × RMS Score + 40% × Confidence − Trend Penalty

| Warning Level | Health Score | Action |
|--------------|-------------|--------|
| 🟢 Healthy | ≥ 80% | Normal operation |
| 🔵 Watch | 60–80% | Schedule inspection |
| 🟡 Warning | 40–60% | Maintenance within 48h |
| 🔴 Critical | < 40% | Stop immediately |

---

## 📁 Project Structure

```
Motor-Fault-Detection/
│
├── src/
│   ├── pipeline.py          # Train Random Forest model
│   ├── predict_server.py    # Flask REST API + Firebase upload
│   ├── rul_estimator.py     # RUL estimation module
│   └── test_rul.py          # RUL test script
│
├── esp32/
│   └── esp32_bmi160_http.ino  # ESP32 Arduino code (BMI160 + HTTP)
│
├── dashboard/
│   └── index.html           # Real-time web dashboard
│
├── data/
│   ├── raw/                 # Raw dataset folders
│   │   ├── normal/
│   │   ├── unbalance/
│   │   ├── misalignment/
│   │   ├── looseness/
│   │   ├── broken_rotor_bar/
│   │   ├── bearing_fault/
│   │   └── shorted_winding/
│   ├── features.csv         # Extracted features (generated)
│   └── rf_model.pkl         # Trained model (generated)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
cd Motor-Fault-Detection
python src/pipeline.py
```

### 3. Configure Firebase

- Download `firebase_key.json` from Firebase Console → Project Settings → Service Accounts
- Place it in the project root directory

### 4. Run prediction server

```bash
python src/predict_server.py
```

### 5. Open dashboard

Open `dashboard/index.html` in Chrome/Edge browser.

### 6. Flash ESP32

Open `esp32/esp32_bmi160_http.ino` in Arduino IDE, update Wi-Fi credentials and server IP, then flash to ESP32.

---

## 🔌 Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Microcontroller | ESP32 (any variant) |
| Vibration Sensor | BMI160 (I2C) |
| Motor | 3-phase induction motor |
| Connection | SDA → GPIO21, SCL → GPIO22 |
| Power | 3.3V |

**BMI160 Settings:**
- Sampling Rate: 1,600 Hz (ODR)
- Accelerometer Range: ±2g
- Segment Length: 512 samples per request

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Send vibration samples, receive fault prediction |
| GET | `/health` | Check server status |
| POST | `/rul/reset` | Reset RUL after maintenance |

**Request format:**
```json
{
  "samples": [0.12, -0.34, 0.56, ...],
  "fs": 1600
}
```

**Response format:**
```json
{
  "fault": "Bearing Fault",
  "label": 5,
  "confidence": 97.5,
  "rms": 0.1234,
  "all_probs": { "Normal": 0.5, "Bearing Fault": 97.5, ... },
  "rul": {
    "health_score": 72.3,
    "rul_minutes": 94,
    "status": "Warning",
    "warning_level": 2,
    "recommendation": "⚠️ Schedule maintenance within 48h."
  }
}
```

---

## 🗄️ Firebase Database Structure

```
proj-1-1d7a8-default-rtdb/
├── status/                  # Current fault state (0 or 1)
│   ├── normal: 1
│   ├── unbalance: 0
│   ├── misalignment: 0
│   ├── looseness: 0
│   ├── broken_rotor_bar: 0
│   ├── bearing_fault: 0
│   └── shorted_winding: 0
│
├── current/                 # Latest prediction + RUL
│   ├── fault: "Normal"
│   ├── confidence: 97.5
│   ├── rms: 0.0523
│   ├── timestamp: "2026-06-05 14:33:10"
│   └── rul/
│       ├── health_score: 95.2
│       ├── rul_minutes: null
│       └── status: "Healthy"
│
└── history/                 # Last 100 predictions
    └── 20260605_143310_xxx/
        ├── fault: "Normal"
        └── ...
```

---

## 📦 Requirements

```
flask
joblib
numpy
scipy
scikit-learn
pandas
matplotlib
seaborn
h5py
firebase-admin
requests
```

---

## 📚 Datasets Used

| Dataset | Fault Types | Source |
|---------|------------|--------|
| MaFaulDa | Normal, Unbalance, Misalignment, Bearing | [UFRJ Brazil](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html) |
| Mendeley (zx8pfhdtnb) | Looseness | [Mendeley Data](https://data.mendeley.com/datasets/zx8pfhdtnb/2) |
| IEEE DataPort | Broken Rotor Bar | [IEEE DataPort](https://ieee-dataport.org/open-access/experimental-database-detecting-and-diagnosing-rotor-broken-bar-three-phase-induction) |
| CWRU | Bearing Fault | [Case Western Reserve](https://engineering.case.edu/bearingdatacenter) |
| University of Ottawa | Shorted Winding | [Mendeley Data](https://data.mendeley.com/datasets/msxs4vj48g/2) |

---

## 📄 License

This project is developed for academic purposes at HCMUTE.  
© 2026 Nguyễn Thế Bảo, Trần Hạo Khiêm, Phùng Thanh Triều. All rights reserved.
