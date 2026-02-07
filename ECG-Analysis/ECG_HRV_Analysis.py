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

DATA_DIR = "."   # folder where .bin files and Participant Data-2.xlsx live
PARTICIPANT_FILE = os.path.join(DATA_DIR, "Participant Data-2.xlsx")

WARMUP_S = 4.0       # seconds of initial data to drop
FORCE_ECG_UP = True  # flip ECG polarity so R-peaks are positive


# ======================================================
# 1. HELPER FUNCTIONS
# ======================================================

def lowpass_filter(data, cutoff=20, fs=256, order=4):
    """
    Simple low-pass Butterworth filter for IMU signals.
    """
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
    Full ECG + IMU processing for one NilsPod .bin file.
    Returns:
        fs, t_trim, ecg_scaled, hr_time_s, hr_values, hr_sampled, motion_mask_trim
    """
    # -------------------
    # Load NILSPOD data
    # -------------------
    df, fs_dict = load_dataset_nilspod(
        file_path=file_path,
        datastreams=None,
        handle_counter_inconsistency="warn",
    )

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

    if isinstance(ecg_clean, pd.DataFrame):
        ecg_clean_series = ecg_clean.iloc[:, 0].copy()
    else:
        ecg_clean_series = pd.Series(ecg_clean, name="ECG_Clean")

    # -------------------
    # Trim initial warm-up
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
    # IMU preprocessing
    # -------------------
    acc = df[["acc_x", "acc_y", "acc_z"]]
    gyr = df[["gyr_x", "gyr_y", "gyr_z"]]

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

    acc_mag_smooth = acc_mag.rolling(window=int(fs), min_periods=1).mean()
    gyr_mag_smooth = gyr_mag.rolling(window=int(fs), min_periods=1).mean()

    acc_trim = acc_mag_smooth.iloc[warmup_samples:].reset_index(drop=True)
    gyr_trim = gyr_mag_smooth.iloc[warmup_samples:].reset_index(drop=True)

    motion_mask_trim = detect_motion_artifacts_from_imu(acc_trim, gyr_trim, fs)

    # -------------------
    # Remove R-peaks in motion artifacts
    # -------------------
    peak_samples_trim = peak_samples_trim[
        (peak_samples_trim >= 0) & (peak_samples_trim < len(ecg_scaled))
    ]

    motion_at_peaks = motion_mask_trim.iloc[peak_samples_trim].to_numpy()
    good_peaks_mask = ~motion_at_peaks

    r_peaks_final = r_peaks_trim.iloc[good_peaks_mask].copy()
    heart_rate_final = heart_rate_trim.iloc[good_peaks_mask].copy()

    # -------------------
    # Heart rate timelines
    # -------------------
    hr_values = heart_rate_final["Heart_Rate"].to_numpy()  # bpm
    peak_samples_final = r_peaks_final["R_Peak_Idx"].to_numpy().astype(int)

    hr_time_s = peak_samples_final / fs

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


def compute_hr_and_hrv_metrics(hr_time_s: np.ndarray, hr_values: np.ndarray):
    """
    Compute Mean HR, Max increase HR, SDNN, RMSSD, and a simple HRV-derived
    stress score (0–100, higher = more stressed).
    """
    if hr_values is None or len(hr_values) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Mean HR (bpm)
    mean_hr = float(np.mean(hr_values))

    # Max increase HR (bpm)
    max_inc_hr = float(np.max(hr_values) - np.min(hr_values))

    # HRV from RR intervals
    rr_intervals_s = np.diff(hr_time_s)
    rr_ms = rr_intervals_s * 1000.0

    if len(rr_ms) < 2:
        sdnn_ms = np.nan
        rmssd_ms = np.nan
        stress_score = np.nan
    else:
        # SDNN (ms)
        sdnn_ms = float(np.std(rr_ms, ddof=1))

        # RMSSD (ms)
        diff_rr = np.diff(rr_ms)
        if len(diff_rr) == 0:
            rmssd_ms = np.nan
        else:
            rmssd_ms = float(np.sqrt(np.mean(diff_rr**2)))

        # HRV-derived stress score from RMSSD
        if np.isnan(rmssd_ms):
            stress_score = np.nan
        else:
            rmssd_clamped = np.clip(rmssd_ms, 10.0, 100.0)
            stress_score = 100.0 - (rmssd_clamped - 10.0) / (100.0 - 10.0) * 100.0

    return mean_hr, max_inc_hr, sdnn_ms, rmssd_ms, stress_score


def barplot_by_condition(df, column, ylabel, title):
    """
    Bar plot (mean ± SD) of a column vs condition.
    """
    group_stats = df.groupby("condition")[column].agg(["mean", "std", "count"])

    x_labels = group_stats.index.tolist()
    means = group_stats["mean"].values
    stds = group_stats["std"].values

    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Condition")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def scatter_hrv_vs_stress(df, hrv_col="RMSSD_ms", stress_col="Stress_Score"):
    """
    Scatter plot showing HRV trend vs stress score.
    Lower HRV -> higher stress.
    """
    mask = df[hrv_col].notna() & df[stress_col].notna()
    if mask.sum() < 2:
        print("Not enough data for HRV vs stress scatter plot.")
        return

    x = df.loc[mask, hrv_col]
    y = df.loc[mask, stress_col]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.8)
    ax.set_xlabel(f"{hrv_col} (ms)")
    ax.set_ylabel("Stress score (0–100, higher = more stressed)")
    ax.set_title("HRV trend versus HRV-derived stress")
    plt.tight_layout()
    plt.show()


# ======================================================
# 2. PROCESS ALL PARTICIPANTS
# ======================================================

df_participants = pd.read_excel(PARTICIPANT_FILE)

df_participants = df_participants.copy()
df_participants["condition"] = df_participants["condition"].astype(str).str.strip()
df_participants["Bin file"] = df_participants["Bin file"].astype(str).str.strip()

results = []

for _, row in df_participants.iterrows():
    vp_id = row["VP_ID"]
    condition = row["condition"]
    bin_base = row["Bin file"]

    if not bin_base.endswith(".bin"):
        bin_base = bin_base + ".bin"

    file_path = os.path.join(DATA_DIR, bin_base)

    if not os.path.exists(file_path):
        print(f"WARNING: file not found for {vp_id}: {file_path}")
        mean_hr = np.nan
        max_inc_hr = np.nan
        sdnn_ms = np.nan
        rmssd_ms = np.nan
        stress_score = np.nan
    else:
        print(f"Processing {vp_id}: {file_path}")
        out = process_nilspod_file(file_path, warmup_s=WARMUP_S)
        mean_hr, max_inc_hr, sdnn_ms, rmssd_ms, stress_score = compute_hr_and_hrv_metrics(
            out["hr_time_s"], out["hr_values"]
        )

    results.append(
        {
            "VP_ID": vp_id,
            "condition": condition,
            "mean_HR": mean_hr,
            "max_increase_HR": max_inc_hr,
            "SDNN_ms": sdnn_ms,
            "RMSSD_ms": rmssd_ms,
            "Stress_Score": stress_score,
        }
    )

df_results = pd.DataFrame(results)

print("\n=== Per-participant HR + HRV + stress summary ===")
print(df_results)


# ======================================================
# 3. DESCRIPTIVE STATISTICS
# ======================================================

print("\n=== Descriptive stats by condition ===")
desc_by_cond = df_results.groupby("condition")[[
    "mean_HR", "max_increase_HR", "SDNN_ms", "RMSSD_ms", "Stress_Score"
]].agg(["mean", "std", "min", "max", "count"])
print(desc_by_cond)


# ======================================================
# 4. ANOVAS (condition only)
# ======================================================

df_anova = df_results.dropna(subset=["max_increase_HR", "SDNN_ms", "RMSSD_ms", "Stress_Score"]).copy()
df_anova["condition"] = df_anova["condition"].astype("category")

print("\n=== ANOVA: max_increase_HR ~ condition ===")
model_hr = smf.ols("max_increase_HR ~ C(condition)", data=df_anova).fit()
print(sm.stats.anova_lm(model_hr, typ=2))

print("\n=== ANOVA: SDNN_ms ~ condition ===")
model_sdnn = smf.ols("SDNN_ms ~ C(condition)", data=df_anova).fit()
print(sm.stats.anova_lm(model_sdnn, typ=2))

print("\n=== ANOVA: RMSSD_ms ~ condition ===")
model_rmssd = smf.ols("RMSSD_ms ~ C(condition)", data=df_anova).fit()
print(sm.stats.anova_lm(model_rmssd, typ=2))

print("\n=== ANOVA: Stress_Score ~ condition ===")
model_stress = smf.ols("Stress_Score ~ C(condition)", data=df_anova).fit()
print(sm.stats.anova_lm(model_stress, typ=2))


# ======================================================
# 5. PLOTS (including HRV trend vs stress)
# ======================================================

barplot_by_condition(
    df_results,
    "max_increase_HR",
    "Max increase HR (bpm)",
    "Max increase HR by Condition (speech vs math)"
)

barplot_by_condition(
    df_results,
    "mean_HR",
    "Mean HR (bpm)",
    "Mean HR by Condition (speech vs math)"
)

barplot_by_condition(
    df_results,
    "SDNN_ms",
    "SDNN (ms)",
    "SDNN by Condition (speech vs math)"
)

barplot_by_condition(
    df_results,
    "RMSSD_ms",
    "RMSSD (ms)",
    "RMSSD by Condition (speech vs math)"
)

# HRV-derived stress (this is your “HRV trend vs stress by condition” bar plot)
barplot_by_condition(
    df_results,
    "Stress_Score",
    "Stress score (0–100, higher = more stressed)",
    "HRV-derived stress score by Condition (speech vs math)"
)

# HRV trend versus stress across participants (scatter)
scatter_hrv_vs_stress(df_results, hrv_col="RMSSD_ms", stress_col="Stress_Score")


# ======================================================
# 6. SAVE RESULTS
# ======================================================

output_csv = os.path.join(DATA_DIR, "HR_HRV_stress_summary_by_participant.csv")
df_results.to_csv(output_csv, index=False)
print(f"\nSaved: {output_csv}")
