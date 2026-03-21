import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

# ==================================================
# 1. CREATE OUTPUT DIRECTORIES
# ==================================================

os.makedirs("./PANAS-Analysis/figures", exist_ok=True)
os.makedirs("./PANAS-Analysis/processed", exist_ok=True)

# ==================================================
# 2. LOAD DATA
# ==================================================

FILE_PATH = "./PANAS-Analysis/data/data_project_1119684_2026_01_12.csv"

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

df_scores.to_csv("./PANAS-Analysis/processed/panas_scored_wide_pre_post.csv", index=False)


# ==================================================
# 7. BASELINE EQUIVALENCE
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
# 8. MIXED-EFFECTS MODELS
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

df_long.to_csv("./PANAS-Analysis/processed/panas_long_format_for_mixed_models.csv", index=False)

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
# 9. EFFECT SIZES
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
# 10. PLOTS
# ==================================================

def plot_pre_post(long_df, ylabel, title, filename):
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

    fig, ax = plt.subplots()

    for g in summary["Group"].unique():
        d = summary[summary["Group"] == g]
        ax.errorbar(
            d["Time"],
            d["mean"],
            yerr=d["sem"],
            marker="o",
            capsize=4,
            label=g
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Condition")

    # ✅ SAVE FIGURE
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


plot_pre_post(na_long, 
              "Negative Affect", 
              "Negative Affect Pre–Post by Condition", 
              "./PANAS-Analysis/figures/na_pre_post_by_condition.png")

plot_pre_post(pa_long, 
              "Positive Affect", 
              "Positive Affect Pre–Post by Condition", 
              "./PANAS-Analysis/figures/pa_pre_post_by_condition.png")

# ==================================================
# END OF SCRIPT
# ==================================================
