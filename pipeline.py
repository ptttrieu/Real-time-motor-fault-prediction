"""
=======================================================
  HỆ THỐNG DỰ ĐOÁN HỎNG HÓC ĐỘNG CƠ - Random Forest
=======================================================
Dataset sources:
  - normal/          : MaFaulDa CSV  (N_rows, 8)
  - unbalance/       : MaFaulDa CSV  subfolder 6g/10g/...
  - misalignment/    : MaFaulDa CSV  subfolder horizontal/vertical
  - looseness/       : Mendeley .npy (4, 25000)
  - broken_rotor_bar : IEEE .mat v7.3 (h5py)
  - bearing_fault/   : CWRU .mat (scipy)
  - shorted_winding/ : Ottawa CSV    (has header row)
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import h5py
import scipy.io
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH
# ============================================================
DATA_DIR     = 'data/raw'
FEATURES_CSV = 'data/features.csv'
MODEL_PATH   = 'data/rf_model.pkl'
PLOT_PATH    = 'data/rf_results.png'

FS_MAFAULDA = 50000
FS_CWRU     = 12000
FS_OTTAWA   = 42000
SEGMENT_LEN = 4096

RPM       = 1480
F_ROT     = RPM / 60
F_SUPPLY  = 50
N_BALLS   = 8
BALL_DIA  = 6.75
PITCH_DIA = 28.5
BPFO = (N_BALLS / 2) * F_ROT * (1 - BALL_DIA / PITCH_DIA)
BPFI = (N_BALLS / 2) * F_ROT * (1 + BALL_DIA / PITCH_DIA)

CLASS_NAMES = ['Normal', 'Unbalance', 'Misalignment', 'Looseness',
               'Broken Rotor Bar', 'Bearing Fault', 'Shorted Winding']

FEATURE_NAMES = [
    'RMS', 'Peak', 'Crest', 'Kurtosis', 'Skewness', 'ShapeFactor',
    '1xRPM', '2xRPM', '3xRPM', '4xRPM', '5xRPM',
    '50Hz', '100Hz', '150Hz', 'BPFO', 'BPFI',
    'HarmonicRatio', 'SidebandRatio',
    'E_0-50Hz', 'E_50-200Hz', 'E_200-500Hz', 'E_500-2kHz',
]

print("=" * 55)
print("  HỆ THỐNG DỰ ĐOÁN HỎNG HÓC ĐỘNG CƠ 3 PHA")
print("=" * 55)
print(f"  F_ROT = {F_ROT:.2f} Hz | BPFO = {BPFO:.1f} Hz | BPFI = {BPFI:.1f} Hz")


# ============================================================
# 1. EXTRACT FEATURES
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

    p_1x  = peak_at(F_ROT);     p_2x    = peak_at(2*F_ROT)
    p_3x  = peak_at(3*F_ROT);   p_4x    = peak_at(4*F_ROT)
    p_5x  = peak_at(5*F_ROT)
    p_50  = peak_at(F_SUPPLY);  p_100   = peak_at(2*F_SUPPLY)
    p_150 = peak_at(3*F_SUPPLY)
    p_bpfo = peak_at(BPFO);     p_bpfi  = peak_at(BPFI)

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


def segment_and_extract(signal, fs, label, max_seg=50):
    rows  = []
    n_seg = min(max_seg, len(signal) // SEGMENT_LEN)
    for i in range(n_seg):
        seg  = signal[i*SEGMENT_LEN:(i+1)*SEGMENT_LEN]
        feat = extract_features(seg, fs)
        if feat:
            rows.append(feat + [label])
    return rows


# ============================================================
# 2. LOAD FUNCTIONS
# ============================================================

def load_mafaulda_csv(folder, label, col=2, max_files=30):
    """MaFaulDa CSV: (N_rows, 8), không có header, col=2 = underhang radial."""
    files = glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True)
    rows, count = [], 0
    for f in files:
        if count >= max_files:
            break
        try:
            df  = pd.read_csv(f, header=None)
            sig = df.iloc[:, col].values.astype(float)
            rows.extend(segment_and_extract(sig, FS_MAFAULDA, label))
            count += 1
        except Exception as e:
            print(f"    [WARN] {os.path.basename(f)}: {e}")
    print(f"    → {count} files | {len(rows)} segments")
    return rows


def load_ottawa_csv(folder, label, col=0, max_files=30):
    """
    Ottawa CSV: có header 'Accelerometer 1 (m/s^2)' ở hàng đầu.
    fs = 42000 Hz.
    col=0 → Accelerometer 1 (drive end) ← dùng cái này
    col=1 → Acoustic
    col=2 → Accelerometer 2
    """
    files = glob.glob(os.path.join(folder, '**', '*.csv'), recursive=True)
    rows, count = [], 0
    for f in files:
        if count >= max_files:
            break
        try:
            df  = pd.read_csv(f, header=0)          # bỏ dòng header
            sig = df.iloc[:, col].values.astype(float)
            rows.extend(segment_and_extract(sig, FS_OTTAWA, label))
            count += 1
        except Exception as e:
            print(f"    [WARN] {os.path.basename(f)}: {e}")
    print(f"    → {count} files | {len(rows)} segments")
    return rows


def load_looseness_npy(folder, label, max_files=50):
    """Looseness .npy: shape (4, 25000), dùng kênh 1 (radial)."""
    files = glob.glob(os.path.join(folder, '**', '*.npy'), recursive=True)
    rows, count = [], 0
    for f in files:
        if count >= max_files:
            break
        try:
            arr = np.load(f)
            sig = arr[1, :] if arr.ndim == 2 else arr
            rows.extend(segment_and_extract(sig.astype(float), FS_MAFAULDA, label))
            count += 1
        except Exception as e:
            print(f"    [WARN] {os.path.basename(f)}: {e}")
    print(f"    → {count} files | {len(rows)} segments")
    return rows


def load_broken_rotor_mat(folder, label, max_files=4):
    """IEEE DataPort .mat v7.3 (h5py): r1b/r2b/r3b/r4b, bỏ struct_rs."""
    files = glob.glob(os.path.join(folder, '*.mat'))
    rows, count = [], 0
    for f in files:
        if count >= max_files:
            break
        if 'struct_rs' in os.path.basename(f):
            continue
        try:
            with h5py.File(f, 'r') as hf:
                root_keys = [k for k in hf.keys() if not k.startswith('#')]
                if not root_keys:
                    continue
                root = hf[root_keys[0]]
                for torque_key in root.keys():
                    tq = root[torque_key]
                    if 'Vib_acpe' not in tq:
                        continue
                    refs = tq['Vib_acpe'][:]
                    for ref_row in refs:
                        try:
                            sig = hf[ref_row[0]][:].flatten().astype(float)
                            rows.extend(segment_and_extract(
                                sig, FS_MAFAULDA, label, max_seg=8))
                        except Exception:
                            pass
            count += 1
        except Exception as e:
            print(f"    [WARN] {os.path.basename(f)}: {e}")
    print(f"    → {count} files | {len(rows)} segments")
    return rows


def load_cwru_mat(folder, label, max_files=20):
    """CWRU .mat (scipy): key X###_DE_time."""
    files = glob.glob(os.path.join(folder, '**', '*.mat'), recursive=True)
    rows, count = [], 0
    for f in files:
        if count >= max_files:
            break
        try:
            mat    = scipy.io.loadmat(f)
            de_key = next((k for k in mat if 'DE_time' in k), None)
            if de_key is None:
                de_key = next((k for k in mat if 'FE_time' in k), None)
            if de_key:
                sig = mat[de_key].flatten().astype(float)
                rows.extend(segment_and_extract(sig, FS_CWRU, label))
                count += 1
        except Exception as e:
            print(f"    [WARN] {os.path.basename(f)}: {e}")
    print(f"    → {count} files | {len(rows)} segments")
    return rows


# ============================================================
# 3. BUILD DATASET
# ============================================================
def build_dataset():
    all_rows = []

    print("\n[1/7] NORMAL — MaFaulDa CSV")
    all_rows += load_mafaulda_csv(
        os.path.join(DATA_DIR, 'normal'), label=0, col=2, max_files=30)

    print("\n[2/7] UNBALANCE — MaFaulDa CSV")
    all_rows += load_mafaulda_csv(
        os.path.join(DATA_DIR, 'unbalance'), label=1, col=2, max_files=30)

    print("\n[3/7] MISALIGNMENT — MaFaulDa CSV")
    all_rows += load_mafaulda_csv(
        os.path.join(DATA_DIR, 'misalignment'), label=2, col=2, max_files=30)

    print("\n[4/7] LOOSENESS — .npy")
    all_rows += load_looseness_npy(
        os.path.join(DATA_DIR, 'looseness'), label=3, max_files=50)

    print("\n[5/7] BROKEN ROTOR BAR — .mat h5py")
    all_rows += load_broken_rotor_mat(
        os.path.join(DATA_DIR, 'broken_rotor_bar'), label=4, max_files=4)

    print("\n[6/7] BEARING FAULT — CWRU .mat scipy")
    all_rows += load_cwru_mat(
        os.path.join(DATA_DIR, 'bearing_fault'), label=5, max_files=30)

    print("\n[7/7] SHORTED WINDING — Ottawa CSV")
    all_rows += load_ottawa_csv(
        os.path.join(DATA_DIR, 'shorted_winding'), label=6, col=0, max_files=30)

    cols = FEATURE_NAMES + ['label']
    df   = pd.DataFrame(all_rows, columns=cols).dropna()

    # Cân bằng dataset
    MAX_PER_CLASS = 600
    balanced = []
    for lbl in sorted(df['label'].unique()):
        subset = df[df['label'] == lbl]
        n = min(len(subset), MAX_PER_CLASS)
        balanced.append(subset.sample(n, random_state=42))
    df = pd.concat(balanced, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n" + "=" * 45)
    print("  DATASET SUMMARY (sau cân bằng)")
    print("=" * 45)
    for i, name in enumerate(CLASS_NAMES):
        n   = (df['label'] == i).sum()
        bar = '█' * (n // 10)
        print(f"  {i} {name:20s}: {n:5d}  {bar}")
    print(f"  {'TOTAL':22s}: {len(df):5d}")
    print(f"  Features per sample: {len(FEATURE_NAMES)}")

    df.to_csv(FEATURES_CSV, index=False)
    print(f"\n  ✓ Saved: {FEATURES_CSV}")
    return df


# ============================================================
# 4. TRAIN
# ============================================================
def train_model(df):
    X = df[FEATURE_NAMES].values
    y = df['label'].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n" + "=" * 45)
    print("  TRAINING RANDOM FOREST")
    print("=" * 45)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc    = np.mean(y_pred == y_test)
    print(f"\n  Test Accuracy : {acc*100:.2f}%")

    # Chỉ report các class có trong data thực tế
    labels_present = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    names_present  = [CLASS_NAMES[i] for i in labels_present]
    print("\n" + classification_report(
        y_test, y_pred, labels=labels_present, target_names=names_present))

    cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"  5-Fold CV: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

    joblib.dump(rf, MODEL_PATH)
    print(f"\n  ✓ Model saved: {MODEL_PATH}")
    return rf, X_test, y_test, y_pred


# ============================================================
# 5. PLOT
# ============================================================
def plot_results(rf, X_test, y_test, y_pred):
    labels_present = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    names_present  = [CLASS_NAMES[i] for i in labels_present]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor('#1e1e2e')
    for ax in axes:
        ax.set_facecolor('#1e1e2e')

    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0],
                xticklabels=names_present, yticklabels=names_present,
                cmap='YlOrRd', linewidths=0.5,
                annot_kws={'color':'white','fontsize':9})
    axes[0].set_title('Confusion Matrix', color='white', fontsize=13, pad=10)
    axes[0].set_ylabel('Actual',    color='white')
    axes[0].set_xlabel('Predicted', color='white')
    axes[0].tick_params(colors='white')
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right', color='white', fontsize=8)
    plt.setp(axes[0].get_yticklabels(), rotation=0,  color='white', fontsize=8)

    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:15]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 15))
    axes[1].barh(range(15), imp[idx][::-1], color=colors[::-1])
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels([FEATURE_NAMES[i] for i in idx[::-1]], color='white', fontsize=9)
    axes[1].set_title('Top 15 Feature Importance', color='white', fontsize=13, pad=10)
    axes[1].set_xlabel('Importance Score', color='white')
    axes[1].tick_params(colors='white')
    axes[1].spines[:].set_color('#444')

    fig.suptitle('Random Forest — Motor Fault Detection',
                 color='white', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight', facecolor='#1e1e2e')
    plt.close()
    print(f"  ✓ Plot saved: {PLOT_PATH}")


# ============================================================
# PREDICT ĐƠN LẺ
# ============================================================
def predict_signal(signal, fs=FS_MAFAULDA, model_path=MODEL_PATH):
    rf   = joblib.load(model_path)
    feat = extract_features(signal[:SEGMENT_LEN], fs)
    if feat is None:
        return "Error: signal quá ngắn", 0.0
    X    = np.array(feat).reshape(1, -1)
    pred = int(rf.predict(X)[0])
    prob = rf.predict_proba(X)[0][pred] * 100
    return CLASS_NAMES[pred], round(prob, 2)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    df = build_dataset()
    rf, X_test, y_test, y_pred = train_model(df)

    print("\nGenerating plots...")
    plot_results(rf, X_test, y_test, y_pred)

    print("\n" + "=" * 55)
    print("  HOÀN THÀNH!")
    print(f"  features.csv → {FEATURES_CSV}")
    print(f"  model        → {MODEL_PATH}")
    print(f"  plot         → {PLOT_PATH}")
    print("=" * 55)