# Complete preprocessing and feature extraction for NILSPOD .bin
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from biopsykit.io.nilspod import load_dataset_nilspod
from biopsykit.signals.ecg import EcgProcessor

# -------------------------
# 1️⃣ Load NILSPOD dataset
# -------------------------
file_path = "NilsPodX-B0C2_20251106_175809.bin"

# Load dataset safely even if counter is not monotonous
df, fs_dict = load_dataset_nilspod(
    file_path=file_path,
    datastreams=None,
    handle_counter_inconsistency="warn"
)

# Check if single or multiple streams
if isinstance(df, dict):
    print("Datastreams present:", list(df.keys()))
    stream_name = list(df.keys())[0]
    df = df[stream_name]
    fs = fs_dict[stream_name]
else:
    print("Single datastream detected")
    fs = fs_dict

print(f"Sampling rate: {fs} Hz")
print(f"Columns: {list(df.columns)}")
print(f"Number of samples: {len(df)}")
print(df.head())

# -------------------------
# 2️⃣ Preprocess ECG
# -------------------------
ecg_signal = df[['ecg']]

ep = EcgProcessor(data=ecg_signal, sampling_rate=fs)
ep.ecg_process()  # filtering, R-peak detection, outlier removal

# Access results
ecg_clean = ep.ecg_result['Data']['ECG_Clean']
r_peaks = ep.rpeaks['Data']
heart_rate = ep.heart_rate['Data']

print("\n--- ECG Preview ---")
print(ecg_clean.head())
print("Detected R-peaks (first 10):")
print(r_peaks.head(10))
print("Heart rate (first 10):")
print(heart_rate.head(10))

# -------------------------
# 3️⃣ Preprocess IMU (Accelerometer + Gyroscope)
# -------------------------
acc = df[['acc_x', 'acc_y', 'acc_z']]
gyr = df[['gyr_x', 'gyr_y', 'gyr_z']]

# Low-pass filter function
def lowpass_filter(data, cutoff=20, fs=256, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Apply filtering
acc_filtered = acc.apply(lambda col: lowpass_filter(col, cutoff=20, fs=fs))
gyr_filtered = gyr.apply(lambda col: lowpass_filter(col, cutoff=20, fs=fs))

# Compute vector magnitude
acc_mag = np.sqrt(acc_filtered['acc_x']**2 + acc_filtered['acc_y']**2 + acc_filtered['acc_z']**2)
gyr_mag = np.sqrt(gyr_filtered['gyr_x']**2 + gyr_filtered['gyr_y']**2 + gyr_filtered['gyr_z']**2)

# Smooth with 1-second rolling average (min_periods=1 to avoid NaN at start)
acc_mag_smooth = acc_mag.rolling(window=int(fs), min_periods=1).mean()
gyr_mag_smooth = gyr_mag.rolling(window=int(fs), min_periods=1).mean()

print("\n--- IMU Preview ---")
print("Accelerometer magnitude (first 10):")
print(acc_mag_smooth.head(10))
print("Gyroscope magnitude (first 10):")
print(gyr_mag_smooth.head(10))

# -------------------------
# 4️⃣ Feature Extraction (basic)
# -------------------------
features = {}

# ECG features
features['HR_mean'] = heart_rate['Heart_Rate'].mean()
features['HR_std'] = heart_rate['Heart_Rate'].std()
features['R_peak_count'] = len(r_peaks)  # total R-peaks detected

# IMU features
features['acc_mag_mean'] = acc_mag.mean()
features['acc_mag_std'] = acc_mag.std()
features['gyr_mag_mean'] = gyr_mag.mean()
features['gyr_mag_std'] = gyr_mag.std()

print("\n--- Summary of Features ---")
for k, v in features.items():
    print(f"{k}: {v:.3f}")

# -------------------------
# 5️⃣ Optional: Save processed data
# -------------------------
df_processed = pd.DataFrame({
     'ECG_Clean': ecg_clean,
     'Heart_Rate': heart_rate['Heart_Rate'],
     'Acc_Mag': acc_mag_smooth,
     'Gyr_Mag': gyr_mag_smooth
 })
df_processed.to_csv("processed_nilspod_data.csv")
