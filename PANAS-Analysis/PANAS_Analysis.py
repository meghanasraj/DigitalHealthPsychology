import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ==================================================
# 1. CREATE OUTPUT DIRECTORIES
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

FIG_DIR = BASE_DIR / "figures"
PROC_DIR = BASE_DIR / "processed"

FIG_DIR.mkdir(exist_ok=True)
PROC_DIR.mkdir(exist_ok=True)

# ==================================================
# 2. LOAD DATA
# ==================================================

FILE_PATH = BASE_DIR / "data" / "panas_raw_data.csv"

df_raw = pd.read_csv(
    FILE_PATH,
    sep=";",
    header=1,
    skiprows=[2],
    engine="python"
)

ID_COL = "Participant ID"
GROUP_COL = "Condition"


# ==================================================
# 3. PANAS ITEM DEFINITIONS
# ==================================================

PA_ITEMS = [
    "active", "interested", "excited", "strong", "enthusiastic",
    "proud", "alert", "inspired", "determined", "attentive"
]

NA_ITEMS = [
    "distressed", "upset", "guilty", "scared", "hostile",
    "irritable", "ashamed", "nervous", "jittery", "afraid"
]


# ==================================================
# 4. IDENTIFY PRE / POST COLUMNS
# ==================================================

PA_PRE_COLS  = [c for c in df_raw.columns if c in PA_ITEMS and not c.endswith(".1")]
NA_PRE_COLS  = [c for c in df_raw.columns if c in NA_ITEMS and not c.endswith(".1")]
PA_POST_COLS = [c for c in df_raw.columns if c.replace(".1", "") in PA_ITEMS and c.endswith(".1")]
NA_POST_COLS = [c for c in df_raw.columns if c.replace(".1", "") in NA_ITEMS and c.endswith(".1")]

assert len(PA_PRE_COLS) == len(NA_PRE_COLS) == 10
assert len(PA_POST_COLS) == len(NA_POST_COLS) == 10


# ==================================================
# 5. DATA CLEANING & SCORING FUNCTIONS
# ==================================================

def range_check(df, cols):
    """Replace PANAS values outside 1–5 with NaN."""
    df[cols] = df[cols].apply(lambda x: x.where((x >= 1) & (x <= 5), np.nan))
    return df


def score_subscale(df, cols):
    """
    PANAS scoring:
    ≤ 2 missing → person-mean imputation
    ≥ 3 missing → subscale missing
    """
    sub = df[cols]
    n_missing = sub.isna().sum(axis=1)
    row_means = sub.mean(axis=1)

    sub_imputed = sub.copy()
    for i in range(len(sub)):
        if n_missing.iloc[i] <= 2:
            sub_imputed.iloc[i] = sub.iloc[i].fillna(row_means.iloc[i])
        else:
            sub_imputed.iloc[i] = np.nan

    return sub_imputed.sum(axis=1)


# ==================================================
# 6. SCORE PANAS
# ==================================================

df = df_raw.copy()
df = range_check(df, PA_PRE_COLS + NA_PRE_COLS)
df = range_check(df, PA_POST_COLS + NA_POST_COLS)

df_scores = pd.DataFrame({
    "ID": df[ID_COL],
    "Group": df[GROUP_COL].astype("category"),
    "PA_pre":  score_subscale(df, PA_PRE_COLS),
    "NA_pre":  score_subscale(df, NA_PRE_COLS),
    "PA_post": score_subscale(df, PA_POST_COLS),
    "NA_post": score_subscale(df, NA_POST_COLS),
})

df_scores["ΔPA"] = df_scores["PA_post"] - df_scores["PA_pre"]
df_scores["ΔNA"] = df_scores["NA_post"] - df_scores["NA_pre"]

df_scores.to_csv(PROC_DIR / "panas_scored_wide_pre_post.csv", index=False)

# ==================================================
# 7. DESCRIPTIVE STATISTICS FOR TABLE
# ==================================================

desc = (
    df_scores
    .groupby("Group", observed=True)[["PA_pre", "PA_post", "NA_pre", "NA_post"]]
    .agg(["mean", "std"])
    .round(2)
)

print("\nDescriptive statistics (PANAS):")
print(desc)

# ==================================================
# 8. BASELINE EQUIVALENCE
# ==================================================

math   = df_scores[df_scores["Group"] == "math"]
speech = df_scores[df_scores["Group"] == "speech"]

print(
    "Baseline PA:",
    stats.ttest_ind(math["PA_pre"], speech["PA_pre"], equal_var=False, nan_policy="omit")
)

print(
    "Baseline NA:",
    stats.ttest_ind(math["NA_pre"], speech["NA_pre"], equal_var=False, nan_policy="omit")
)

# ==================================================
# 9. MIXED-EFFECTS MODELS
# ==================================================

df_long = pd.melt(
    df_scores,
    id_vars=["ID", "Group"],
    value_vars=["PA_pre", "PA_post", "NA_pre", "NA_post"],
    var_name="Measure",
    value_name="Score"
)

df_long["Time"]   = df_long["Measure"].str.split("_").str[1].str.capitalize()
df_long["Affect"] = df_long["Measure"].str.split("_").str[0]

df_long.to_csv(PROC_DIR / "panas_long_format_for_mixed_models.csv", index=False)

# Negative Affect
na_long = df_long[df_long["Affect"] == "NA"].dropna()

print(
    smf.mixedlm("Score ~ Time * Group", na_long, groups=na_long["ID"]).fit().summary()
)

# Positive Affect
pa_long = df_long[df_long["Affect"] == "PA"].dropna()

print(
    smf.mixedlm("Score ~ Time * Group", pa_long, groups=pa_long["ID"]).fit().summary()
)


# ==================================================
# 10. EFFECT SIZES
# ==================================================

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    )
    return (x.mean() - y.mean()) / pooled_sd

print("Cohen’s d (ΔNA):", cohens_d(math["ΔNA"].dropna(), speech["ΔNA"].dropna()))
print("Cohen’s d (ΔPA):", cohens_d(math["ΔPA"].dropna(), speech["ΔPA"].dropna()))


# ==================================================
# 11. PLOTS
# ==================================================

def plot_pre_post(long_df, ylabel, filename):
    summary = (
        long_df
        .groupby(["Group", "Time"], observed=True)["Score"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    # 🔹 Force correct left-to-right order
    summary["Time"] = pd.Categorical(
        summary["Time"],
        categories=["Pre", "Post"],
        ordered=True
    )

    summary = summary.sort_values("Time")

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")

    # APA-style lines (black & gray, different markers/linestyles)
    styles = {
        "math":   dict(color="black", marker="o", linestyle="-"),
        "speech": dict(color="gray",  marker="s", linestyle="--")
    }

    for g in summary["Group"].unique():
        d = summary[summary["Group"] == g]
        ax.errorbar(
            d["Time"],
            d["mean"],
            yerr=d["sem"],
            capsize=4,
            label=g,
            **styles.get(g, {})
        )
        
    # Labels (keep, APA)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    
    # Clean look (APA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    # Legend (no title)
    ax.legend(frameon=False)

    # Save
    plt.tight_layout()

    # ✅ SAVE FIGURE
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


plot_pre_post(na_long, 
              "Negative Affect",
              FIG_DIR / "na_pre_post_by_condition.png")

plot_pre_post(pa_long, 
              "Positive Affect", 
              FIG_DIR / "pa_pre_post_by_condition.png")

# ==================================================
# END OF SCRIPT
# ==================================================