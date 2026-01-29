import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from biopsykit.io.nilspod import load_dataset_nilspod
from biopsykit.signals.ecg import EcgProcessor

import statsmodels.api as sm
import statsmodels.formula.api as smf

# ======================================================
# 0. CONFIG
# ======================================================

DATA_DIR = "."  # folder where the .bin and Participant Data-2.xlsx live
PARTICIPANT_FILE = os.path.join(DATA_DIR, "Participant Data-2.xlsx")

WARMUP_S = 4.0  # seconds to drop at start, as in your script
FORCE_ECG_UP = True  # flip ECG polarity so R-peaks are positive

# ======================================================
# 1. HELPER FUNCTIONS
# ======================================================

def lowpass_filter(data, cutoff=20, fs=256, order=4):
    """Simple low-pass Butterworth filter for IMU data."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


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


def process_nilspod_file(file_path: str, warmup_s: float = 4.0):
    """
    Run the full ECG + IMU processing pipeline on one NilsPod .bin file.
    Returns:
        dict with:
            - fs
            - t_trim (time axis after warmup)
            - ecg_scaled (trimmed, baseline-corrected, normalized ECG)
            - hr_time_s (seconds of each beat after warmup)
            - hr_values (beat-wise HR in bpm, motion-cleaned)
            - hr_sampled (sample-wise HR aligned to t_trim)
    """

    # -------------------
    # Load NILSPOD data
    # -------------------
    df, fs_dict = load_dataset_nilspod(
        file_path=file_path,
        datastreams=None,
        handle_counter_inconsistency="warn",
    )

    # handle multiple datastreams or single
    if isinstance(df, dict):
        stream_name = list(df.keys())[0]
        df = df[stream_name]
        fs = fs_dict[stream_name]
    else:
        fs = fs_dict

    # -------------------
    # ECG processing
    # -------------------
    ecg_signal = df[["ecg"]].copy()

    if FORCE_ECG_UP:
        ecg_signal = -ecg_signal

    ep = EcgProcessor(data=ecg_signal, sampling_rate=fs)
    ep.ecg_process()

    ecg_clean = ep.ecg_result["Data"]["ECG_Clean"]
    r_peaks = ep.rpeaks["Data"]
    heart_rate = ep.heart_rate["Data"]

    # ensure 1D Series for ECG
    if isinstance(ecg_clean, pd.DataFrame):
        ecg_clean_series = ecg_clean.iloc[:, 0].copy()
    else:
        ecg_clean_series = pd.Series(ecg_clean, name="ECG_Clean")

    # -------------------
    # Trim warmup
    # -------------------
    warmup_samples = int(warmup_s * fs)
    ecg_clean_trim = ecg_clean_series.iloc[warmup_samples:].reset_index(drop=True)

    peak_samples_orig = r_peaks["R_Peak_Idx"].to_numpy().astype(int)
    mask_after = peak_samples_orig >= warmup_samples
    peak_samples_trim = peak_samples_orig[mask_after] - warmup_samples

    r_peaks_trim = r_peaks.loc[mask_after].copy()
    r_peaks_trim["R_Peak_Idx"] = peak_samples_trim
    heart_rate_trim = heart_rate.loc[mask_after].copy()

    # -------------------
    # Baseline removal + scaling
    # -------------------
    baseline = ecg_clean_trim.rolling(
        window=int(fs * 1.5), center=True, min_periods=1
    ).median()
    ecg_detrended = ecg_clean_trim - baseline

    scale = np.percentile(np.abs(ecg_detrended.values), 99)
    if scale == 0:
        scale = 1.0
    ecg_scaled = ecg_detrended / scale

    t_trim = np.arange(len(ecg_scaled)) / fs

    # -------------------
    # IMU preprocessing (acc, gyr magnitudes)
    # -------------------
    acc = df[["acc_x", "acc_y", "acc_z"]]
    gyr = df[["gyr_x", "gyr_y", "gyr_z"]]

    acc_filtered = acc.apply(lambda col: lowpass_filter(col, cutoff=20, fs=fs))
    gyr_filtered = gyr.apply(lambda col: lowpass_filter(col, cutoff=20, fs=fs))

    acc_mag = np.sqrt(
        acc_filtered["acc_x"] ** 2
        + acc_filtered["acc_y"] ** 2
        + acc_filtered["acc_z"] ** 2
    )
    gyr_mag = np.sqrt(
        gyr_filtered["gyr_x"] ** 2
        + gyr_filtered["gyr_y"] ** 2
        + gyr_filtered["gyr_z"] ** 2
    )

    # smooth + trim IMU
    acc_mag_smooth = acc_mag.rolling(window=int(fs), min_periods=1).mean()
    gyr_mag_smooth = gyr_mag.rolling(window=int(fs), min_periods=1).mean()

    acc_trim = acc_mag_smooth.iloc[warmup_samples:].reset_index(drop=True)
    gyr_trim = gyr_mag_smooth.iloc[warmup_samples:].reset_index(drop=True)

    motion_mask_trim = detect_motion_artifacts_from_imu(acc_trim, gyr_trim, fs)

    # -------------------
    # Remove R-peaks inside motion artifacts
    # -------------------
    peak_samples_trim = peak_samples_trim[
        (peak_samples_trim >= 0) & (peak_samples_trim < len(ecg_scaled))
    ]

    motion_at_peaks = motion_mask_trim.iloc[peak_samples_trim].to_numpy()
    good_peaks_mask = ~motion_at_peaks

    r_peaks_final = r_peaks_trim.iloc[good_peaks_mask].copy()
    heart_rate_final = heart_rate_trim.iloc[good_peaks_mask].copy()

    # -------------------
    # Heart rate on beat timeline + sample timeline
    # -------------------
    hr_values = heart_rate_final["Heart_Rate"].to_numpy()
    peak_samples_final = r_peaks_final["R_Peak_Idx"].to_numpy().astype(int)

    hr_time_s = peak_samples_final / fs  # seconds after warmup

    hr_series = pd.Series(hr_values, index=hr_time_s, name="Heart_Rate").sort_index()
    sample_index_trim = pd.Index(t_trim, name="time_s")

    hr_sampled = (
        hr_series.reindex(sample_index_trim)
        .interpolate(method="index")
        .ffill()
        .bfill()
    )

    return {
        "fs": fs,
        "t_trim": t_trim,
        "ecg_scaled": ecg_scaled,
        "hr_time_s": hr_time_s,
        "hr_values": hr_values,
        "hr_sampled": hr_sampled,
        "motion_mask_trim": motion_mask_trim,
    }


def compute_hr_metrics(hr_time_s: np.ndarray, hr_values: np.ndarray):
    """
    Compute session-level HR metrics:
        - mean HR (bpm)
        - max increase HR (max - min of beat-wise HR)
    Uses all beats after warmup and motion cleaning.
    """
    if hr_values is None or len(hr_values) < 2:
        return np.nan, np.nan

    mean_hr = float(np.mean(hr_values))
    max_inc_hr = float(np.max(hr_values) - np.min(hr_values))
    return mean_hr, max_inc_hr


# ======================================================
# 2. PROCESS ALL PARTICIPANTS
# ======================================================

df_participants = pd.read_excel(PARTICIPANT_FILE)

# clean up relevant columns
df_participants = df_participants.copy()
df_participants["condition"] = df_participants["condition"].str.strip()
df_participants["Bin file"] = df_participants["Bin file"].astype(str).str.strip()

results = []

for _, row in df_participants.iterrows():
    vp_id = row["VP_ID"]
    condition = row["condition"]       # speech / math
    sex_code = row["sex"]             # 1 / 2 from sheet

    bin_base = row["Bin file"]
    # Ensure .bin extension
    if not bin_base.endswith(".bin"):
        bin_base = bin_base + ".bin"

    file_path = os.path.join(DATA_DIR, bin_base)

    if not os.path.exists(file_path):
        print(f"WARNING: file not found for {vp_id}: {file_path}")
        mean_hr = np.nan
        max_inc_hr = np.nan
    else:
        print(f"Processing {vp_id}: {file_path}")
        out = process_nilspod_file(file_path, warmup_s=WARMUP_S)
        mean_hr, max_inc_hr = compute_hr_metrics(out["hr_time_s"], out["hr_values"])

    results.append(
        {
            "VP_ID": vp_id,
            "condition": condition,
            "sex_code": sex_code,
            "mean_HR": mean_hr,
            "max_increase_HR": max_inc_hr,
        }
    )

df_results = pd.DataFrame(results)

# add a readable sex label (if 1 = women, 2 = men, as in your sheet)
sex_map = {1: "women", 2: "men"}
df_results["sex"] = df_results["sex_code"].map(sex_map)

print("\n=== Per-participant HR summary ===")
print(df_results)

# ======================================================
# 3. DESCRIPTIVE STATISTICS
# ======================================================

print("\n=== Descriptive stats by condition ===")
desc_by_cond = df_results.groupby("condition")[["mean_HR", "max_increase_HR"]].agg(
    ["mean", "std", "min", "max", "count"]
)
print(desc_by_cond)

print("\n=== Descriptive stats by condition × sex ===")
desc_by_cond_sex = df_results.groupby(["condition", "sex"])[
    ["mean_HR", "max_increase_HR"]
].agg(["mean", "std", "count"])
print(desc_by_cond_sex)

# ======================================================
# 4. 2×2 BETWEEN-SUBJECT ANOVA (Condition × Sex)
#    DV: max_increase_HR
# ======================================================

# keep only rows with valid sex and HR
df_anova = df_results.dropna(subset=["max_increase_HR", "sex"])
df_anova["condition"] = df_anova["condition"].astype("category")
df_anova["sex"] = df_anova["sex"].astype("category")

model = smf.ols("max_increase_HR ~ C(condition) * C(sex)", data=df_anova).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\n=== 2×2 ANOVA on max_increase_HR (Condition × Sex) ===")
print(anova_table)

# (If you also want ANOVA on mean HR, you can duplicate with DV = mean_HR)

# ======================================================
# 5. PLOTS
# ======================================================

# --------- Plot 1: Condition vs Max increase HR ---------
fig, ax = plt.subplots(figsize=(6, 4))

# simple bar plot with mean ± SD
group_stats = df_results.groupby("condition")["max_increase_HR"].agg(["mean", "std", "count"])

x_labels = group_stats.index.tolist()
means = group_stats["mean"].values
stds = group_stats["std"].values

x = np.arange(len(x_labels))
ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylabel("Max increase HR (bpm)")
ax.set_xlabel("Condition")
ax.set_title("Max increase HR by Condition (speech vs. math)")

plt.tight_layout()
plt.show()

# --------- Plot 2: Condition vs Mean HR ---------
fig, ax = plt.subplots(figsize=(6, 4))

group_stats_mean = df_results.groupby("condition")["mean_HR"].agg(["mean", "std", "count"])

x_labels2 = group_stats_mean.index.tolist()
means2 = group_stats_mean["mean"].values
stds2 = group_stats_mean["std"].values

x2 = np.arange(len(x_labels2))
ax.bar(x2, means2, yerr=stds2, capsize=5, alpha=0.8)
ax.set_xticks(x2)
ax.set_xticklabels(x_labels2)
ax.set_ylabel("Mean HR (bpm)")
ax.set_xlabel("Condition")
ax.set_title("Mean HR by Condition (speech vs. math)")

plt.tight_layout()
plt.show()

# ======================================================
# 6. OPTIONAL: SAVE RESULTS
# ======================================================

df_results.to_csv("HR_summary_by_participant.csv", index=False)
print("\nSaved: HR_summary_by_participant.csv")
