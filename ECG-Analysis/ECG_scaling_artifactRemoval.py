import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from biopsykit.io.nilspod import load_dataset_nilspod
from biopsykit.signals.ecg import EcgProcessor

# -------------------------------------------------
# 1️⃣ Load NILSPOD dataset
# -------------------------------------------------
file_path = "NilsPodX-56BB_20251212_163521.bin"

df, fs_dict = load_dataset_nilspod(
    file_path=file_path,
    datastreams=None,
    handle_counter_inconsistency="warn",
)
print(df.columns)
print(df.head())

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

# -------------------------------------------------
# 2️⃣ ECG processing (flip raw before processing)
# -------------------------------------------------
# Raw ECG
ecg_signal = df[["ecg"]].copy()

# Force polarity here: make R-peaks positive before detection
FORCE_ECG_UP = True
if FORCE_ECG_UP:
    ecg_signal = -ecg_signal
    print("Flipping raw ECG before EcgProcessor (forcing upright QRS).")

ep = EcgProcessor(data=ecg_signal, sampling_rate=fs)
ep.ecg_process()

ecg_clean = ep.ecg_result["Data"]["ECG_Clean"]   # sample-level
r_peaks = ep.rpeaks["Data"]                      # beat-level
heart_rate = ep.heart_rate["Data"]               # beat-level

# 1D Series for convenience
if isinstance(ecg_clean, pd.DataFrame):
    ecg_clean_series = ecg_clean.iloc[:, 0].copy()
else:
    ecg_clean_series = pd.Series(ecg_clean, name="ECG_Clean")

print("\n--- ECG Preview (raw clean) ---")
print(ecg_clean_series.head())

# -------------------------------------------------
# 3️⃣ Remove initial artifact (warm-up) + scale ECG
# -------------------------------------------------
warmup_s = 4.0                     # seconds to cut off at the beginning
warmup_samples = int(warmup_s * fs)

# trim ECG
ecg_clean_trim = ecg_clean_series.iloc[warmup_samples:].reset_index(drop=True)

# adjust R-peak indices to trimmed signal
peak_samples_orig = r_peaks["R_Peak_Idx"].to_numpy().astype(int)
mask_after = peak_samples_orig >= warmup_samples
peak_samples_trim = peak_samples_orig[mask_after] - warmup_samples

r_peaks_trim = r_peaks.loc[mask_after].copy()
r_peaks_trim["R_Peak_Idx"] = peak_samples_trim
heart_rate_trim = heart_rate.loc[mask_after].copy()

print(f"\nDropped first {warmup_s} s ({warmup_samples} samples).")
print(f"Peaks before: {len(peak_samples_orig)}, after trimming: {len(peak_samples_trim)}")

# baseline removal with moving median
baseline = ecg_clean_trim.rolling(
    window=int(fs * 1.5), center=True, min_periods=1
).median()
ecg_detrended = ecg_clean_trim - baseline

# robust scaling: 99th percentile of absolute value
scale = np.percentile(np.abs(ecg_detrended.values), 99)
if scale == 0:
    scale = 1.0
ecg_scaled = ecg_detrended / scale

print(f"Scaling factor (99th percentile): {scale:.1f}")

# trimmed time axis
t_trim = np.arange(len(ecg_scaled)) / fs

# -------------------------------------------------
# 4️⃣ IMU preprocessing (also trimmed)
# -------------------------------------------------
acc = df[["acc_x", "acc_y", "acc_z"]]
gyr = df[["gyr_x", "gyr_y", "gyr_z"]]

def lowpass_filter(data, cutoff=20, fs=256, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

acc_filtered = acc.apply(lambda col: lowpass_filter(col, cutoff=20, fs=fs))
gyr_filtered = gyr.apply(lambda col: lowpass_filter(col, cutoff=20, fs=fs))

acc_mag = np.sqrt(
    acc_filtered["acc_x"]**2
    + acc_filtered["acc_y"]**2
    + acc_filtered["acc_z"]**2
)
gyr_mag = np.sqrt(
    gyr_filtered["gyr_x"]**2
    + gyr_filtered["gyr_y"]**2
    + gyr_filtered["gyr_z"]**2
)

# smooth + trim to match ECG
acc_mag_smooth = acc_mag.rolling(window=int(fs), min_periods=1).mean()
gyr_mag_smooth = gyr_mag.rolling(window=int(fs), min_periods=1).mean()

acc_trim = acc_mag_smooth.iloc[warmup_samples:].reset_index(drop=True)
gyr_trim = gyr_mag_smooth.iloc[warmup_samples:].reset_index(drop=True)

def detect_motion_artifacts_from_imu(
    acc_mag_series: pd.Series,
    gyr_mag_series: pd.Series,
    fs: float,
    window_s: float = 0.5,
    acc_factor: float = 2.0,
    gyr_factor: float = 2.0,
) -> pd.Series:
    """
    Use rolling std of smoothed acceleration and gyroscope magnitude
    to flag motion-heavy samples. Returns a boolean Series where
    True = motion artifact.
    """
    win = max(1, int(window_s * fs))

    acc_std = acc_mag_series.rolling(win, center=True, min_periods=1).std()
    gyr_std = gyr_mag_series.rolling(win, center=True, min_periods=1).std()

    acc_thr = acc_std.mean() + acc_factor * acc_std.std()
    gyr_thr = gyr_std.mean() + gyr_factor * gyr_std.std()

    motion_mask = (acc_std > acc_thr) | (gyr_std > gyr_thr)
    return motion_mask.fillna(False)

motion_mask_trim = detect_motion_artifacts_from_imu(acc_trim, gyr_trim, fs)
print(f"\nFraction of trimmed samples flagged as motion: {motion_mask_trim.mean():.3f}")

# -------------------------------------------------
# 5️⃣ Remove R-peaks that fall into motion-artifact regions
# -------------------------------------------------
peak_samples_trim = peak_samples_trim[
    (peak_samples_trim >= 0) & (peak_samples_trim < len(ecg_scaled))
]

motion_at_peaks = motion_mask_trim.iloc[peak_samples_trim].to_numpy()
good_peaks_mask = ~motion_at_peaks

print(
    f"Trimmed peaks: {len(peak_samples_trim)}, "
    f"removed due to motion: {motion_at_peaks.sum()}, "
    f"kept: {good_peaks_mask.sum()}"
)

r_peaks_final = r_peaks_trim.iloc[good_peaks_mask].copy()
heart_rate_final = heart_rate_trim.iloc[good_peaks_mask].copy()

# -------------------------------------------------
# 6️⃣ HR on sample timeline (trimmed)
# -------------------------------------------------
hr_values = heart_rate_final["Heart_Rate"].to_numpy()
peak_samples_final = r_peaks_final["R_Peak_Idx"].to_numpy().astype(int)

# times of each beat in seconds (after warmup)
hr_time_s = peak_samples_final / fs

hr_series = pd.Series(hr_values, index=hr_time_s, name="Heart_Rate").sort_index()

sample_index_trim = pd.Index(t_trim, name="time_s")

hr_sampled = (
    hr_series.reindex(sample_index_trim)
             .interpolate(method="index")
             .ffill()
             .bfill()
)

# -------------------------------------------------
# 6️⃣.5 HR & HRV metrics for the plotted window (layman-friendly)
# -------------------------------------------------
window_s = 20.0  # duration shown in the plot (seconds after warmup)

# use only beats inside the window
mask_hr_win = (hr_time_s >= 0) & (hr_time_s <= window_s)
hr_values_win = hr_values[mask_hr_win]
hr_time_win = hr_time_s[mask_hr_win]

if len(hr_values_win) >= 2:
    # Mean HR (bpm)
    mean_hr = float(np.mean(hr_values_win))

    # RR intervals (s) and HRV metrics
    rr_intervals_s = np.diff(hr_time_win)
    rr_ms = rr_intervals_s * 1000.0

    if len(rr_ms) > 1:
        sdnn_ms = float(np.std(rr_ms, ddof=1))
    else:
        sdnn_ms = np.nan

    diff_rr = np.diff(rr_ms)
    if len(diff_rr) > 0:
        rmssd_ms = float(np.sqrt(np.mean(diff_rr**2)))
    else:
        rmssd_ms = np.nan
else:
    mean_hr = np.nan
    sdnn_ms = np.nan
    rmssd_ms = np.nan

# Layman-friendly HRV level derived from RMSSD
if np.isnan(rmssd_ms):
    hrv_level = "N/A"
    stress_score = np.nan
else:
    # HRV Level
    if rmssd_ms < 20:
        hrv_level = "Low"
    elif rmssd_ms < 50:
        hrv_level = "Moderate"
    else:
        hrv_level = "High"

    # Stress Score (0–100), higher = more stressed
    rmssd_clamped = np.clip(rmssd_ms, 10, 100)
    stress_score = int(
        100 - (rmssd_clamped - 10) / (100 - 10) * 100
    )

print("\n--- HR / HRV (plotted window) ---")
print(f"Mean HR: {mean_hr:.1f} bpm")
print(f"SDNN:    {sdnn_ms:.1f} ms")
print(f"RMSSD:   {rmssd_ms:.1f} ms")
print(f"HRV Level: {hrv_level}")
if not np.isnan(stress_score):
    print(f"Stress Score: {stress_score}/100")

# -------------------------------------------------
# 7️⃣ Plot (TRIMMED + SCALED + MOTION-CLEANED ONLY)
# -------------------------------------------------
ecg_vals_at_peaks = ecg_scaled.iloc[peak_samples_final].to_numpy()

if window_s is not None:
    mask_time = t_trim <= window_s
    t_plot = t_trim[mask_time]
    ecg_plot = ecg_scaled.iloc[mask_time]

    mask_peaks = hr_time_s <= window_s
    peak_times_plot = hr_time_s[mask_peaks]
    ecg_peaks_plot = ecg_vals_at_peaks[mask_peaks]
else:
    t_plot = t_trim
    ecg_plot = ecg_scaled
    peak_times_plot = hr_time_s
    ecg_peaks_plot = ecg_vals_at_peaks

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t_plot, ecg_plot.values, linewidth=1, label="ECG (clean, scaled)")
ax.scatter(peak_times_plot, ecg_peaks_plot, marker="x", s=35,
           label="R-peaks (trimmed + motion-cleaned)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("ECG (normalized)")
ax.set_title("ECG with detected R-peaks (scaled, IMU-cleaned)")
ax.legend(loc="upper right")

# 🔳 Add layman HR / HRV info box on the plot
if not np.isnan(mean_hr):
    text_lines = [
        f"Mean HR: {mean_hr:.1f} bpm",
        f"HRV Level: {hrv_level}",
    ]
    if not np.isnan(stress_score):
        text_lines.append(f"Stress: {stress_score}/100")

    textstr = "\n".join(text_lines)

    ax.text(
        0.02, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

plt.tight_layout()
plt.show()

# -------------------------------------------------
# 8️⃣ Save processed data (trimmed)
# -------------------------------------------------
df_processed = pd.DataFrame({
    "ECG_Clean_Scaled": ecg_scaled,
    "Heart_Rate": hr_sampled,
    "Acc_Mag": acc_trim,
    "Gyr_Mag": gyr_trim,
    "Motion_Artifact": motion_mask_trim,
}, index=sample_index_trim)

print("\n--- Processed Data Preview ---")
print(df_processed.head())
print("\nNaNs in Heart_Rate column:", df_processed["Heart_Rate"].isna().sum())

df_processed.to_csv("processed_nilspod_data_trimmed.csv", index=True)
print("\nSaved: processed_nilspod_data_trimmed.csv")
