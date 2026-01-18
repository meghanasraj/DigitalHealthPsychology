import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

FILE_PATH = "./PANAS-Analysis/data_project_1119684_2026_01_12.csv"

df_raw = pd.read_csv(
    FILE_PATH,
    sep=";",
    header=1,      # row 2 = column names
    skiprows=[2],  # skip row 3 (subheaders)
    engine="python"
)


# --------------------------------------------------
# 2. Key column names
# --------------------------------------------------

ID_COL = "Participant ID"
GROUP_COL = "Condition"


# --------------------------------------------------
# 3. PANAS item definitions
# --------------------------------------------------

PA_ITEMS = [
    "active", "interested", "excited", "strong", "enthusiastic",
    "proud", "alert", "inspired", "determined", "attentive"
]

NA_ITEMS = [
    "distressed", "upset", "guilty", "scared", "hostile",
    "irritable", "ashamed", "nervous", "jittery", "afraid"
]


# --------------------------------------------------
# 4. Identify PRE and POST PANAS columns
# --------------------------------------------------

# PRE (no .1 suffix)
PA_PRE_COLS = [
    c for c in df_raw.columns
    if (c in PA_ITEMS or c == "ethusiastic") and not c.endswith(".1")
]

NA_PRE_COLS = [
    c for c in df_raw.columns
    if c in NA_ITEMS and not c.endswith(".1")
]

# POST (.1 suffix)
PA_POST_COLS = [
    c for c in df_raw.columns
    if (c.replace(".1", "") in PA_ITEMS or c == "ethusiastic.1")
    and c.endswith(".1")
]

NA_POST_COLS = [
    c for c in df_raw.columns
    if c.replace(".1", "") in NA_ITEMS and c.endswith(".1")
]

# Sanity check
assert len(PA_PRE_COLS) == len(NA_PRE_COLS) == 10
assert len(PA_POST_COLS) == len(NA_POST_COLS) == 10


# --------------------------------------------------
# 5. Data cleaning utilities
# --------------------------------------------------

def range_check(df, cols):
    """Set values outside PANAS range (1–5) to NaN."""
    df[cols] = df[cols].apply(
        lambda x: x.where((x >= 1) & (x <= 5), np.nan)
    )
    return df


def score_subscale(df, cols):
    """
    Score a PANAS subscale.
    ≤ 2 missing items → person-mean imputation
    ≥ 3 missing items → subscale set to missing
    """
    sub = df[cols]
    n_missing = sub.isna().sum(axis=1)

    sub_imputed = sub.copy()
    row_means = sub.mean(axis=1)

    for i in range(len(sub)):
        if n_missing.iloc[i] <= 2:
            sub_imputed.iloc[i] = sub.iloc[i].fillna(row_means.iloc[i])
        else:
            sub_imputed.iloc[i] = np.nan

    return sub_imputed.sum(axis=1)


# --------------------------------------------------
# 6. Apply cleaning and score PANAS
# --------------------------------------------------

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


# --------------------------------------------------
# 7. Baseline equivalence tests
# --------------------------------------------------

math = df_scores[df_scores["Group"] == "math"]
speech = df_scores[df_scores["Group"] == "speech"]

t_PA, p_PA = stats.ttest_ind(
    math["PApre"].dropna(),
    speech["PApre"].dropna(),
    equal_var=False
)

t_NA, p_NA = stats.ttest_ind(
    math["NApre"].dropna(),
    speech["NApre"].dropna(),
    equal_var=False
)

print(f"Baseline PA: t = {t_PA:.3f}, p = {p_PA:.3f}")
print(f"Baseline NA: t = {t_NA:.3f}, p = {p_NA:.3f}")


# --------------------------------------------------
# 8. Primary stress analysis (ΔNA)
# --------------------------------------------------

t_dNA, p_dNA = stats.ttest_ind(
    math["ΔNA"].dropna(),
    speech["ΔNA"].dropna(),
    equal_var=False
)

print(f"ΔNA Group Difference: t = {t_dNA:.3f}, p = {p_dNA:.3f}")


# ==================================================
# 9. Mixed-effects model (Time × Group)
# ==================================================

df_long = pd.melt(
    df_scores,
    id_vars=["ID", "Group"],
    value_vars=["PApre", "PApost", "NApre", "NApost"],
    var_name="Measure",
    value_name="Score"
)

df_long["Time"] = df_long["Measure"].apply(
    lambda x: "Pre" if "pre" in x else "Post"
)

df_long["Affect"] = df_long["Measure"].apply(
    lambda x: "PA" if "PA" in x else "NA"
)

na_long = df_long[df_long["Affect"] == "NA"].dropna()

mixed_model = smf.mixedlm(
    "Score ~ Time * Group",
    data=na_long,
    groups=na_long["ID"]
)

mixed_result = mixed_model.fit()
print(mixed_result.summary())

# --------------------------------------------------
# 10. Effect size (Cohen's d)
# --------------------------------------------------

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1))
        / (nx + ny - 2)
    )
    return (x.mean() - y.mean()) / pooled_sd


d_dNA = cohens_d(
    math["ΔNA"].dropna(),
    speech["ΔNA"].dropna()
)

print(f"Cohen's d for ΔNA = {d_dNA:.2f}")

# --------------------------------------------------
# Plotting (optional)
# --------------------------------------------------    
import matplotlib.pyplot as plt

# 1
# Compute means and SEM
summary = (
    na_long
    .groupby(["Group", "Time"])["Score"]
    .agg(["mean", "sem"])
    .reset_index()
)

summary["Time"] = pd.Categorical(
    summary["Time"],
    categories=["Pre", "Post"],
    ordered=True
)

summary = summary.sort_values("Time")

fig, ax = plt.subplots()

for group in summary["Group"].unique():
    data = summary[summary["Group"] == group]
    ax.errorbar(
        data["Time"],
        data["mean"],
        yerr=data["sem"],
        marker="o",
        label=group,
        capsize=4
    )

ax.set_xlabel("Time")
ax.set_ylabel("Negative Affect")
ax.set_title("Negative Affect Pre–Post by Condition")
ax.legend(title="Condition")

plt.show()
# 2
fig, ax = plt.subplots()

ax.boxplot(
    [math["ΔNA"].dropna(), speech["ΔNA"].dropna()],
    labels=["Math", "Speech"]
)

ax.set_ylabel("ΔNA (Post − Pre)")
ax.set_title("Change in Negative Affect by Condition")

plt.show()



