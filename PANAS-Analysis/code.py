import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


# ==================================================
# 1. LOAD DATA
# ==================================================

FILE_PATH = "./PANAS-Analysis/data_project_1119684_2026_01_12.csv"

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
# 2. PANAS ITEM DEFINITIONS
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
# 3. IDENTIFY PRE / POST COLUMNS
# ==================================================

PA_PRE_COLS  = [c for c in df_raw.columns if c in PA_ITEMS and not c.endswith(".1")]
NA_PRE_COLS  = [c for c in df_raw.columns if c in NA_ITEMS and not c.endswith(".1")]
PA_POST_COLS = [c for c in df_raw.columns if c.replace(".1", "") in PA_ITEMS and c.endswith(".1")]
NA_POST_COLS = [c for c in df_raw.columns if c.replace(".1", "") in NA_ITEMS and c.endswith(".1")]

assert len(PA_PRE_COLS) == len(NA_PRE_COLS) == 10
assert len(PA_POST_COLS) == len(NA_POST_COLS) == 10


# ==================================================
# 4. DATA CLEANING & SCORING FUNCTIONS
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
# 5. SCORE PANAS
# ==================================================

df = df_raw.copy()
df = range_check(df, PA_PRE_COLS + NA_PRE_COLS)
df = range_check(df, PA_POST_COLS + NA_POST_COLS)

df_scores = pd.DataFrame({
    "ID": df[ID_COL],
    "Group": df[GROUP_COL].astype("category"),
    "PApre":  score_subscale(df, PA_PRE_COLS),
    "NApre":  score_subscale(df, NA_PRE_COLS),
    "PApost": score_subscale(df, PA_POST_COLS),
    "NApost": score_subscale(df, NA_POST_COLS),
})

df_scores["ΔPA"] = df_scores["PApost"] - df_scores["PApre"]
df_scores["ΔNA"] = df_scores["NApost"] - df_scores["NApre"]

df_scores.to_csv("./PANAS-Analysis/scored_PANAS.csv", index=False)


# ==================================================
# 6. BASELINE EQUIVALENCE
# ==================================================

math   = df_scores[df_scores["Group"] == "math"]
speech = df_scores[df_scores["Group"] == "speech"]

print(
    "Baseline PA:",
    stats.ttest_ind(math["PApre"], speech["PApre"], equal_var=False, nan_policy="omit")
)

print(
    "Baseline NA:",
    stats.ttest_ind(math["NApre"], speech["NApre"], equal_var=False, nan_policy="omit")
)


# ==================================================
# 7. CHANGE SCORE ANALYSES
# ==================================================

print(
    "ΔNA Group Difference:",
    stats.ttest_ind(math["ΔNA"], speech["ΔNA"], equal_var=False, nan_policy="omit")
)

print(
    "ΔPA Group Difference:",
    stats.ttest_ind(math["ΔPA"], speech["ΔPA"], equal_var=False, nan_policy="omit")
)


# ==================================================
# 8. MIXED-EFFECTS MODELS
# ==================================================

df_long = pd.melt(
    df_scores,
    id_vars=["ID", "Group"],
    value_vars=["PApre", "PApost", "NApre", "NApost"],
    var_name="Measure",
    value_name="Score"
)

df_long["Time"]   = np.where(df_long["Measure"].str.contains("pre"), "Pre", "Post")
df_long["Affect"] = np.where(df_long["Measure"].str.contains("PA"), "PA", "NA")

df_long.to_csv("./PANAS-Analysis/df_long_data.csv", index=False)

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

def plot_pre_post(long_df, ylabel, title):
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

    plt.show()


plot_pre_post(na_long, "Negative Affect", "Negative Affect Pre–Post by Condition")
plot_pre_post(pa_long, "Positive Affect", "Positive Affect Pre–Post by Condition")

plt.boxplot([math["ΔNA"].dropna(), speech["ΔNA"].dropna()],
            tick_labels=["Math", "Speech"])
plt.ylabel("ΔNA (Post − Pre)")
plt.title("Change in Negative Affect")
plt.show()

plt.boxplot([math["ΔPA"].dropna(), speech["ΔPA"].dropna()],
            tick_labels=["Math", "Speech"])
plt.ylabel("ΔPA (Post − Pre)")
plt.title("Change in Positive Affect")
plt.show()
