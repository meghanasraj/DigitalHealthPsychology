import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from biopsykit.io.nilspod import load_dataset_nilspod
from biopsykit.signals.ecg import EcgProcessor

# -------------------------
# 1️⃣ Load NILSPOD dataset
# -------------------------
file_path = "NilsPodX-B0C2_20251106_175809.bin"

df, fs_dict = load_dataset_nilspod(
    file_path=file_path,
    datastreams=None,
    handle_counter_inconsistency="warn"
)

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
ecg_signal = df[["ecg"]]

ep = EcgProcessor(data=ecg_signal, sampling_rate=fs)
ep.ecg_process()

ecg_clean = ep.ecg_result["Data"]["ECG_Clean"]   # sample-level
r_peaks = ep.rpeaks["Data"]                      # beat-level
heart_rate = ep.heart_rate["Data"]               # beat-level

print("\n--- ECG Preview ---")
print(ecg_clean.head())
print("Detected R-peaks (first 10):")
print(r_peaks.head(10))
print("Heart rate (first 10):")
print(heart_rate.head(10))

# -------------------------
# 3️⃣ Preprocess IMU
# -------------------------
acc = df[["acc_x", "acc_y", "acc_z"]]
gyr = df[["gyr_x", "gyr_y", "gyr_z"]]

def lowpass_filter(data, cutoff=20, fs=256, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

acc_filtered = acc.apply(lambda col: lowpass_filter(col, cutoff=20, fs=fs))
gyr_filtered = gyr.apply(lambda col: lowpass_filter(col, cutoff=20, fs=fs))

acc_mag = np.sqrt(acc_filtered["acc_x"]**2 + acc_filtered["acc_y"]**2 + acc_filtered["acc_z"]**2)
gyr_mag = np.sqrt(gyr_filtered["gyr_x"]**2 + gyr_filtered["gyr_y"]**2 + gyr_filtered["gyr_z"]**2)

acc_mag_smooth = acc_mag.rolling(window=int(fs), min_periods=1).mean()
gyr_mag_smooth = gyr_mag.rolling(window=int(fs), min_periods=1).mean()

# -------------------------
# 4️⃣ Features (basic)
# -------------------------
features = {
    "HR_mean": heart_rate["Heart_Rate"].mean(),
    "HR_std": heart_rate["Heart_Rate"].std(),
    "R_peak_count": len(r_peaks),
    "acc_mag_mean": acc_mag.mean(),
    "acc_mag_std": acc_mag.std(),
    "gyr_mag_mean": gyr_mag.mean(),
    "gyr_mag_std": gyr_mag.std(),
}

print("\n--- Summary of Features ---")
for k, v in features.items():
    print(f"{k}: {v:.3f}")

# -------------------------
# 5️⃣ Method A: Align HR to sample timeline and interpolate
# -------------------------

# (A) Sample timeline in seconds for all samples
t = np.arange(len(df)) / fs
sample_index = pd.Index(t, name="time_s")

# Sample-level signals -> time_s index
if isinstance(ecg_clean, pd.DataFrame):
    ecg_clean_series = ecg_clean.iloc[:, 0]
else:
    ecg_clean_series = ecg_clean

ecg_clean_s = pd.Series(ecg_clean_series.to_numpy(), index=sample_index, name="ECG_Clean")
acc_s = pd.Series(acc_mag_smooth.to_numpy(), index=sample_index, name="Acc_Mag")
gyr_s = pd.Series(gyr_mag_smooth.to_numpy(), index=sample_index, name="Gyr_Mag")

# (B) Create a heart-rate series with x-axis = seconds
hr_values = heart_rate["Heart_Rate"].to_numpy()

# 1) BEST: use peak sample positions from r_peaks (if present)
peak_col_candidates = ["R_Peaks", "RPeaks", "r_peaks", "peak_index", "peak_sample", "peaks"]
peak_col = next((c for c in peak_col_candidates if c in r_peaks.columns), None)

if peak_col is not None:
    peak_samples = r_peaks[peak_col].to_numpy()
    hr_time_s = peak_samples / fs
    hr_series = pd.Series(hr_values, index=hr_time_s, name="Heart_Rate")

else:
    # 2) If HR index is datetime/tz-aware: convert to seconds from start
    idx = heart_rate.index

    if isinstance(idx, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        # Convert datetime to seconds from first timestamp
        if isinstance(idx, pd.DatetimeIndex):
            t0 = idx[0]
            hr_time_s = (idx - t0).total_seconds()
        else:
            # TimedeltaIndex
            hr_time_s = idx.total_seconds()

        hr_series = pd.Series(hr_values, index=hr_time_s, name="Heart_Rate")

    else:
        # 3) If numeric index: assume it's already in seconds
        if np.issubdtype(np.asarray(idx).dtype, np.number):
            hr_series = pd.Series(hr_values, index=np.asarray(idx, dtype=float), name="Heart_Rate")
        else:
            # 4) Last resort: spread beats across recording duration (not ideal)
            duration_s = (len(df) - 1) / fs
            hr_time_s = np.linspace(0, duration_s, len(hr_values))
            hr_series = pd.Series(hr_values, index=hr_time_s, name="Heart_Rate")

# Sort before interpolation
hr_series = hr_series.sort_index()

# (C) Reindex onto sample timeline and interpolate
hr_sampled = (
    hr_series.reindex(sample_index)
             .interpolate(method="index")  # numeric seconds axis
             .ffill()
             .bfill()
)

# -------------------------
# 6️⃣ Save processed data
# -------------------------
df_processed = pd.DataFrame({
    "ECG_Clean": ecg_clean_s,
    "Heart_Rate": hr_sampled,
    "Acc_Mag": acc_s,
    "Gyr_Mag": gyr_s
})

print("\n--- Processed Data Preview ---")
print(df_processed.head())
print("\nNaNs in Heart_Rate column:", df_processed["Heart_Rate"].isna().sum())

df_processed.to_csv("processed_nilspod_data.csv", index=True)
print("\nSaved: processed_nilspod_data.csv")
