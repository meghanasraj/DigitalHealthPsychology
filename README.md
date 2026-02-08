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
│   ├── code.py                         # Main analysis script
│   ├── data_project_*.csv              # Raw dataset           
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

* PApre, NApre
* PApost, NApost

Change scores are calculated as:

* ΔPA = PApost − PApre
* ΔNA = NApost − NApre

The final scored dataset is saved as `scored_PANAS.csv`.

---

### Baseline Equivalence

Independent-samples t-tests (Welch correction) are conducted to verify that groups do **not differ at baseline** on:

* Positive Affect (PApre)
* Negative Affect (NApre)

---

### Statistical Modeling

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

Generated figures:

* `fig_NA_pre_post.png`
* `fig_PA_pre_post.png`

---

### How to Run the PANAS Analysis

```bash
python PANAS-Analysis/PANAS_Analysis.py
```

All outputs (CSV files, model summaries, and figures) will be generated automatically.

---

## Part 2: ECG-Analysis

---

## Notes

* File paths are relative; run the script from the project root.
