# Digital Health Psychology Project

## Project Structure

```
DigitalHealthPsychology/
├── ECG-Analysis/                       # ECG analysis
│   ├── ECGAnalysis.py         
│   ├── ECG_DescriptiveStatistics.py   
│   ├── ECG_HRV_Analysis.py             
│   ├── ECG_scaling_artifactRemoval.py  
├── PANAS-Analysis/                     # PANAS analysis 
│   ├── PANAS_Analysis.py               # Main analysis script
│   ├── data/                           # Raw dataset 
│   │   └── panas_raw_data.csv
│   ├── processed/                      # Cleaned & scored datasets
│   │   ├── panas_scored_wide_pre_post.csv
│   │   └── panas_long_format_for_mixed_models.csv
│   ├── figures/                        # Generated plots
│   │   ├── na_pre_post_by_condition.png
│   │   └── pa_pre_post_by_condition.png
│
├── README.md                                              
                      
```

---

## Part 1: PANAS Analysis

### Overview

This part of the project focuses on the **Positive and Negative Affect Schedule (PANAS)**. The analysis compares **Positive Affect (PA)** and **Negative Affect (NA)** scores **before and after** an intervention across two experimental conditions (e.g., *math* vs. *speech*).

The pipeline includes:

* Data cleaning and validation
* PANAS scoring with missing-data handling
* Baseline equivalence testing
* Mixed-effects modeling
* Effect size estimation
* Visualization of pre–post changes

---

### Data Loading

* Input data are read from a semicolon-separated CSV file.
* Participant IDs and group assignments are extracted.
* Pre- and post-intervention PANAS items are identified using column naming conventions (".1" suffix for post).

---

### PANAS Item Definitions

* **Positive Affect (PA)**: 10 items (e.g., *active, interested, excited, inspired*)
* **Negative Affect (NA)**: 10 items (e.g., *distressed, nervous, afraid, hostile*)

Each subscale ranges from **10 to 50**.

---

### Data Cleaning

* PANAS item values are range-checked (valid range: 1–5).
* Values outside this range are replaced with `NaN`.

---

### Scoring Procedure

PANAS subscales are scored according to standard guidelines:

* If **≤ 2 items are missing**, person-mean imputation is applied.
* If **≥ 3 items are missing**, the subscale score is set to missing.

Scores are computed separately for:

* PA_pre, NA_pre
* PA_post, NA_post

Change scores are calculated as:

* ΔPA = PA_post − PA_pre
* ΔNA = NA_post − NA_pre

---

### Output Files

Processed datasets are saved to:

* `processed/panas_scored_wide_pre_post.csv` → Wide-format dataset (one row per participant)

* `processed/panas_long_format_for_mixed_models.csv` → Long-format dataset (for statistical modeling)

---

### Descriptive Statistics

Descriptive statistics are computed for each group and time point, including:

* Mean (M)
* Standard deviation (SD)

Variables summarized:

* Positive Affect (PA_pre, PA_post)
* Negative Affect (NA_pre, NA_post)

These statistics provide an overview of the data distribution.

---

### Baseline Equivalence

Independent-samples t-tests (Welch correction) are conducted to verify that groups do **not differ at baseline** on:

* Positive Affect (PA_pre)
* Negative Affect (NA_pre)

---

### Mixed-Effects Models

To analyze intervention effects, **linear mixed-effects models** are fitted separately for PA and NA:

* Outcome: PANAS score
* Fixed effects: Time (Pre vs. Post), Group, and their interaction
* Random effect: Participant ID (random intercept)

This approach accounts for the repeated-measures structure of the data.

---

### Effect Sizes

Between-group effect sizes are computed using **Cohen’s d** based on change scores (ΔPA and ΔNA), allowing interpretation of the magnitude of intervention effects.

---

### Visualization

Pre–post changes are visualized using line plots with:

* Group-wise means
* Standard error of the mean (SEM) as error bars

Generated figures (saved in `figures/`):

* `na_pre_post_by_condition.png`
* `pa_pre_post_by_condition.png`

---

### How to Run the PANAS Analysis

```bash
cd PANAS-Analysis
python PANAS_Analysis.py
```

All outputs (CSV files, model summaries, and figures) will be generated automatically.

---

## Part 2: ECG-Analysis


# 🫀 ECG, HRV & Stress Analysis Pipeline (NilsPod)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)

A complete pipeline for processing **NilsPod ECG + IMU data** to extract:

* ❤️ Heart Rate (HR)
* 📉 Heart Rate Variability (HRV)
* 🚶 Motion artifacts (IMU-based)
* 😰 HRV-derived stress scores

Includes statistical analysis and visualization for comparing experimental conditions (e.g., **speech vs math stress tasks**).

---
---

## 📄 Input Data

### 1. NilsPod `.bin` files

Place all `.bin` files in the project directory (or update `DATA_DIR`).

### 2. Participant Metadata File

`Participant Data-2.xlsx` must include:

| Column      | Description                                     |
| ----------- | ----------------------------------------------- |
| `VP_ID`     | Participant ID                                  |
| `condition` | Experimental condition (e.g., `speech`, `math`) |
| `Bin file`  | Corresponding `.bin` filename                   |

---

## 🧠 Processing Pipeline

### 🔹 ECG Processing

* Signal cleaning
* R-peak detection
* Heart rate estimation

### 🔹 Signal Conditioning

* Warm-up removal (default: 4 seconds)
* Baseline drift removal
* Signal normalization

### 🔹 Motion Artifact Detection

* Based on IMU (acc + gyro)
* Rolling standard deviation
* Threshold-based detection
* Removes contaminated R-peaks

### 🔹 HRV Metrics

| Metric          | Description                        |
| --------------- | ---------------------------------- |
| Mean HR         | Average heart rate (bpm)           |
| Max Increase HR | Peak HR change                     |
| SDNN            | Standard deviation of RR intervals |
| RMSSD           | Short-term HRV                     |
| Stress Score    | Derived from RMSSD                 |

---

## 😰 Stress Score Model

A simple HRV-based stress proxy:

* RMSSD ↓ → Stress ↑
* Scaled to **0–100 range**

```text
Low RMSSD (~10 ms)  → High stress (~100)
High RMSSD (~100 ms) → Low stress (~0)
```

> ⚠️ This is a heuristic model, not a clinical metric.

---

## 📊 Output

### 1. CSV File

```
HR_HRV_stress_summary_by_participant.csv
```

Contains:

* HR metrics
* HRV metrics
* Stress scores
* Condition labels

---

### 2. Statistical Analysis

ANOVA tests:

```text
max_increase_HR ~ condition
SDNN_ms ~ condition
RMSSD_ms ~ condition
Stress_Score ~ condition
```

---

### 3. Visualizations

* 📊 Bar plots (mean ± SD)

  * Mean HR
  * Max HR increase
  * SDNN
  * RMSSD
  * Stress score

* 📉 Scatter plot:

  * RMSSD vs Stress Score
  * Demonstrates HRV–stress relationship

---

## 🔧 Configuration

Modify parameters in the script:

```python
DATA_DIR = "."
WARMUP_S = 4.0
FORCE_ECG_UP = True
```

Motion detection:

```python
window_s = 0.5
acc_factor = 2.0
gyr_factor = 2.0
```

---

## ⚠️ Limitations

* Requires sufficient R-peaks for HRV calculation
* Motion detection is threshold-based (not ML-based)
* Stress score is simplified (RMSSD only)
* No frequency-domain HRV (LF/HF)

---

## 🙌 Acknowledgments

* [`biopsykit`](https://github.com/mad-lab-fau/biopsykit)
* Scientific Python ecosystem (NumPy, SciPy, Statsmodels)

---


## Notes

* All File paths are relative
* Run the script from the project root
* Output folders are created automatically
