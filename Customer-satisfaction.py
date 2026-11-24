# -*- coding: utf-8 -*-
"""
Customer Satisfaction Analysis
==============================
Dataset : dataset.csv (120 rows x 10 columns)
Columns : Customer_ID, Group, Satisfaction_Score, Age, Gender, Location,
          Purchase_History, Support_Contacted, Loyalty_Level, Satisfaction_Factor

How to run (Positron / VS Code / Jupyter)
-----------------------------------------
Every "# %%" line starts a runnable chunk (click "Run Cell").
Each chunk is self-contained and follows the same six steps:
    1. Library
    2. Data Connection / Data Source
    3. Data Cleaning
    4. Data Preparation
    5. Analysis
    6. Conclusion

Required packages:
    pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn

Optional: set the environment variable DATA_PATH to override the CSV path.
"""


# %% ==============================================================
# CHUNK: 1. DESCRIPTIVE STATISTICS
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:/Users/jonalyn.rapanot/OneDrive - Robinsons Retail Holdings Inc/01_CI Team/08_Github/04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level
num_cols = ['Satisfaction_Score', 'Age']
cat_cols = ['Group', 'Gender', 'Location', 'Purchase_History', 'Support_Contacted', 'Loyalty_Level', 'Satisfaction_Factor']

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
desc = df[num_cols].agg(["count", "mean", "median", "std", "min", "max"]).T
desc["mode"] = [df[c].mode().iloc[0] for c in num_cols]
desc["range"] = desc["max"] - desc["min"]
print("\n--- Numerical variables ---")
print(desc.round(2))

print("\n--- Categorical variables (frequency & %) ---")
for c in cat_cols:
    freq = df[c].value_counts(dropna=False)
    tbl = pd.DataFrame({"Count": freq, "Percent": (freq / freq.sum() * 100).round(1)})
    print(f"\n{c}\n{tbl}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(df["Satisfaction_Score"], bins=10, kde=True, ax=ax[0]); ax[0].set_title("Satisfaction Score")
sns.histplot(df["Age"], bins=10, kde=True, ax=ax[1]); ax[1].set_title("Age")
sns.countplot(data=df, x="Loyalty_Level", ax=ax[2]); ax[2].set_title("Loyalty Level")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "01_descriptive.png"), dpi=150); plt.show()

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
s = df["Satisfaction_Score"]
print(f"Average satisfaction is {s.mean():.2f}/10 (median {s.median():.1f}, SD {s.std():.2f}, range {s.min():.0f}-{s.max():.0f}).")
print(f"Average customer age is {df['Age'].mean():.1f} years (range {df['Age'].min():.0f}-{df['Age'].max():.0f}).")
print(f"Largest loyalty segment: {df['Loyalty_Level'].value_counts().idxmax()} "
      f"({df['Loyalty_Level'].value_counts(normalize=True).max()*100:.1f}% of customers).")
print(f"Most common satisfaction factor: {df['Satisfaction_Factor'].mode().iloc[0]}.")


# %% ==============================================================
# CHUNK: 2. TWO-SAMPLE (INDEPENDENT) t-TEST  -  Group A vs Group B
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level

a = df.loc[df["Group"] == "A", "Satisfaction_Score"]
b = df.loc[df["Group"] == "B", "Satisfaction_Score"]
print(f"Group A n={len(a)}, Group B n={len(b)}")

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
print("Shapiro-Wilk normality  A: p=%.4f | B: p=%.4f" % (stats.shapiro(a)[1], stats.shapiro(b)[1]))
lev_stat, lev_p = stats.levene(a, b)
equal_var = lev_p > ALPHA
print(f"Levene equal-variance test p={lev_p:.4f} -> {'equal' if equal_var else 'unequal'} variances (Welch used if unequal)")

t_stat, p_val = stats.ttest_ind(a, b, equal_var=equal_var)
pooled_sd = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
cohen_d = (a.mean() - b.mean()) / pooled_sd
print(f"Mean A={a.mean():.2f}, Mean B={b.mean():.2f}")
print(f"t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {cohen_d:.2f}")

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
if p_val < ALPHA:
    print(f"SIGNIFICANT: mean satisfaction differs between Group A and B (p={p_val:.4f}, d={cohen_d:.2f}).")
else:
    print(f"NOT significant: no evidence that Group A and B differ in mean satisfaction (p={p_val:.4f}).")


# %% ==============================================================
# CHUNK: 3. PAIRED t-TEST  -  Before vs After
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level

# The base dataset has NO before/after measurements. This test only runs if your file
# contains two related columns; rename below to match your data.
BEFORE_COL, AFTER_COL = "Satisfaction_Before", "Satisfaction_After"
has_pair = {BEFORE_COL, AFTER_COL}.issubset(df.columns)
if has_pair:
    pairs = df[[BEFORE_COL, AFTER_COL]].apply(pd.to_numeric, errors="coerce").dropna()
    diff = pairs[AFTER_COL] - pairs[BEFORE_COL]

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
if has_pair:
    print(f"n pairs = {len(pairs)}, mean before={pairs[BEFORE_COL].mean():.2f}, mean after={pairs[AFTER_COL].mean():.2f}")
    print("Shapiro-Wilk on differences: p=%.4f" % stats.shapiro(diff)[1])
    t_stat, p_val = stats.ttest_rel(pairs[AFTER_COL], pairs[BEFORE_COL])
    dz = diff.mean() / diff.std(ddof=1)
    ci = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))
    print(f"t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's dz = {dz:.2f}, 95% CI of mean change = ({ci[0]:.2f}, {ci[1]:.2f})")
    w_stat, w_p = stats.wilcoxon(pairs[AFTER_COL], pairs[BEFORE_COL])
    print(f"Wilcoxon signed-rank (non-parametric check): p = {w_p:.4f}")
else:
    print(f"SKIPPED: columns '{BEFORE_COL}' and '{AFTER_COL}' are not in the dataset.")

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
if has_pair:
    if p_val < ALPHA:
        print(f"SIGNIFICANT change in satisfaction (mean change {diff.mean():+.2f}, p={p_val:.4f}).")
    else:
        print(f"NOT significant: no evidence of change in satisfaction (p={p_val:.4f}).")
else:
    print("Paired t-test not applicable: the dataset has one satisfaction score per customer. "
          "Add a before/after column pair to enable it.")


# %% ==============================================================
# CHUNK: 4. ONE-WAY ANOVA  -  Satisfaction by Loyalty Level
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level

groups = [df.loc[df["Loyalty_Level"] == lv, "Satisfaction_Score"] for lv in LOYALTY_ORDER]
print({lv: len(g) for lv, g in zip(LOYALTY_ORDER, groups)})

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
print("Levene equal-variance p = %.4f" % stats.levene(*groups)[1])
f_stat, p_val = stats.f_oneway(*groups)
all_scores = df["Satisfaction_Score"]
ss_between = sum(len(g) * (g.mean() - all_scores.mean())**2 for g in groups)
eta_sq = ss_between / ((all_scores - all_scores.mean())**2).sum()
print(df.groupby("Loyalty_Level", observed=True)["Satisfaction_Score"].agg(["count", "mean", "std"]).round(2))
print(f"F = {f_stat:.3f}, p = {p_val:.4f}, eta-squared = {eta_sq:.3f}")

if p_val < ALPHA:
    print("\nTukey HSD post-hoc:")
    print(stats.tukey_hsd(*groups))      # order of groups: Low, Medium, High

sns.boxplot(data=df, x="Loyalty_Level", y="Satisfaction_Score", order=LOYALTY_ORDER)
plt.title("Satisfaction Score by Loyalty Level")
plt.savefig(os.path.join(OUTPUT_DIR, "04_anova_boxplot.png"), dpi=150); plt.show()

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
if p_val < ALPHA:
    print(f"SIGNIFICANT: mean satisfaction differs across loyalty levels (F={f_stat:.2f}, p={p_val:.4f}, eta2={eta_sq:.3f}). See Tukey table for which pairs differ.")
else:
    print(f"NOT significant: no evidence that mean satisfaction differs by loyalty level (p={p_val:.4f}).")


# %% ==============================================================
# CHUNK: 5. CHI-SQUARE TEST OF INDEPENDENCE
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level

pairs_to_test = [("Gender", "Purchase_History"), ("Loyalty_Level", "Support_Contacted")]

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
results = []
for v1, v2 in pairs_to_test:
    ct = pd.crosstab(df[v1], df[v2])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    low_exp = (expected < 5).mean() * 100
    print(f"\n=== {v1} vs {v2} ===")
    print(ct)
    print(f"chi2 = {chi2:.3f}, dof = {dof}, p = {p:.4f}, Cramer's V = {cramers_v:.3f}")
    print(f"Cells with expected count < 5: {low_exp:.0f}%")
    if ct.shape == (2, 2) and low_exp > 0:
        print(f"Fisher exact test p = {stats.fisher_exact(ct)[1]:.4f}  (preferred when expected counts are small)")
    elif low_exp > 20:
        print("WARNING: >20% of cells have expected count < 5; interpret chi-square with caution.")
    results.append((v1, v2, p, cramers_v))

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
for v1, v2, p, v in results:
    verdict = "SIGNIFICANT association" if p < ALPHA else "NO significant association"
    print(f"{v1} vs {v2}: {verdict} (p={p:.4f}, Cramer's V={v:.3f}).")


# %% ==============================================================
# CHUNK: 6. MANN-WHITNEY U TEST  -  Group A vs Group B
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level

a = df.loc[df["Group"] == "A", "Satisfaction_Score"]
b = df.loc[df["Group"] == "B", "Satisfaction_Score"]

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
print("Shapiro-Wilk  A: p=%.4f | B: p=%.4f  (p<0.05 suggests non-normal -> rank test appropriate)"
      % (stats.shapiro(a)[1], stats.shapiro(b)[1]))
u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
rank_biserial = 1 - (2 * u_stat) / (len(a) * len(b))
print(f"Median A={a.median():.1f}, Median B={b.median():.1f}")
print(f"U = {u_stat:.1f}, p = {p_val:.4f}, rank-biserial r = {rank_biserial:.3f}")

sns.boxplot(data=df, x="Group", y="Satisfaction_Score", order=["A", "B"])
plt.title("Satisfaction Score by Group")
plt.savefig(os.path.join(OUTPUT_DIR, "06_mannwhitney_boxplot.png"), dpi=150); plt.show()

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
if p_val < ALPHA:
    print(f"SIGNIFICANT: satisfaction distributions differ between Group A and B (p={p_val:.4f}, r={rank_biserial:.2f}).")
else:
    print(f"NOT significant: distributions of Group A and B do not differ (p={p_val:.4f}).")


# %% ==============================================================
# CHUNK: 7. KRUSKAL-WALLIS TEST
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level
factors_to_test = ['Loyalty_Level', 'Satisfaction_Factor']

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
summary = []
for col in factors_to_test:
    levels = [l for l in df[col].dropna().unique()]
    samples = [df.loc[df[col] == l, "Satisfaction_Score"] for l in levels]
    H, p = stats.kruskal(*samples)
    n, k = len(df), len(levels)
    eps_sq = max((H - k + 1) / (n - k), 0)
    print(f"\n=== Satisfaction_Score by {col} ===")
    print(df.groupby(col, observed=True)["Satisfaction_Score"].agg(["count", "median", "mean"]).round(2))
    print(f"H = {H:.3f}, p = {p:.4f}, epsilon-squared = {eps_sq:.3f}")
    if p < ALPHA:
        pairs = list(combinations(levels, 2))
        print("Pairwise Mann-Whitney (Bonferroni-adjusted):")
        for l1, l2 in pairs:
            pp = stats.mannwhitneyu(df.loc[df[col] == l1, "Satisfaction_Score"],
                                    df.loc[df[col] == l2, "Satisfaction_Score"])[1]
            print(f"  {l1} vs {l2}: adj. p = {min(pp * len(pairs), 1):.4f}")
    summary.append((col, H, p, eps_sq))

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
for col, H, p, e in summary:
    verdict = "SIGNIFICANT difference" if p < ALPHA else "NO significant difference"
    print(f"{col}: {verdict} in satisfaction (H={H:.2f}, p={p:.4f}, epsilon2={e:.3f}).")


# %% ==============================================================
# CHUNK: 8. SPEARMAN'S RANK CORRELATION
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level
pairs_to_test = [('Age', 'Satisfaction_Score'), ('Loyalty_Num', 'Satisfaction_Score'), ('Age', 'Loyalty_Num')]

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
rows = []
for x, y in pairs_to_test:
    rho, p = stats.spearmanr(df[x], df[y])
    rows.append({"Variable 1": x, "Variable 2": y, "rho": round(rho, 3), "p-value": round(p, 4),
                 "Significant": p < ALPHA})
spearman_tbl = pd.DataFrame(rows)
print(spearman_tbl)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.regplot(data=df, x="Age", y="Satisfaction_Score", scatter_kws={"alpha": .6}, ax=ax[0])
sns.stripplot(data=df, x="Loyalty_Level", y="Satisfaction_Score", order=LOYALTY_ORDER, ax=ax[1])
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "08_spearman.png"), dpi=150); plt.show()

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
for _, r in spearman_tbl.iterrows():
    strength = "weak" if abs(r["rho"]) < 0.3 else "moderate" if abs(r["rho"]) < 0.6 else "strong"
    direction = "positive" if r["rho"] > 0 else "negative"
    if r["Significant"]:
        print(f"{r['Variable 1']} vs {r['Variable 2']}: significant {strength} {direction} monotonic relationship (rho={r['rho']}, p={r['p-value']}).")
    else:
        print(f"{r['Variable 1']} vs {r['Variable 2']}: no significant monotonic relationship (rho={r['rho']}, p={r['p-value']}).")


# %% ==============================================================
# CHUNK: 9. LINEAR REGRESSION  -  Satisfaction Score
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level

df["Loyalty_Level"] = df["Loyalty_Level"].astype(str)    # formula API treats this as categorical
formula = "Satisfaction_Score ~ Age + C(Gender) + C(Loyalty_Level, Treatment('Low')) + C(Group) + C(Support_Contacted)"

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
ols = smf.ols(formula, data=df).fit()
print(ols.summary())
print(f"\nR-squared = {ols.rsquared:.3f}, Adj. R-squared = {ols.rsquared_adj:.3f}, model F-test p = {ols.f_pvalue:.4f}")
print("Shapiro-Wilk on residuals: p = %.4f" % stats.shapiro(ols.resid)[1])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.scatterplot(x=ols.fittedvalues, y=ols.resid, ax=ax[0]); ax[0].axhline(0, color="red"); ax[0].set_title("Residuals vs Fitted")
stats.probplot(ols.resid, plot=ax[1]); ax[1].set_title("Q-Q Plot of Residuals")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "09_ols_diagnostics.png"), dpi=150); plt.show()

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
sig = ols.pvalues.drop("Intercept")
sig = sig[sig < ALPHA]
print(f"The model explains {ols.rsquared*100:.1f}% of the variance in satisfaction "
      f"({'statistically significant' if ols.f_pvalue < ALPHA else 'not statistically significant'} overall, p={ols.f_pvalue:.4f}).")
if len(sig):
    for term in sig.index:
        print(f"  Significant predictor: {term} (coef={ols.params[term]:+.3f}, p={sig[term]:.4f})")
else:
    print("  No individual predictor is significant at the 5% level.")


# %% ==============================================================
# CHUNK: 10. LOGISTIC REGRESSION  -  Purchase History & Support Contacted
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, accuracy_score

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level

df["Loyalty_Level"] = df["Loyalty_Level"].astype(str)
targets = {
    "Purchase_Bin": "Purchase_Bin ~ Age + Satisfaction_Score + C(Gender) + C(Loyalty_Level, Treatment('Low'))",
    "Support_Bin":  "Support_Bin ~ Age + Satisfaction_Score + C(Gender) + C(Loyalty_Level, Treatment('Low'))",
}

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
logit_results = {}
for target, formula in targets.items():
    print(f"\n=== Logistic regression: {target} ===")
    print("Class balance:", df[target].value_counts().to_dict())
    try:
        m = smf.logit(formula, data=df).fit(disp=False)
    except Exception as e:                       # e.g. perfect separation / singular matrix
        print("Model failed to converge:", e)
        continue
    print(m.summary2().tables[1].round(4))
    odds = np.exp(pd.concat([m.params, m.conf_int()], axis=1))
    odds.columns = ["Odds Ratio", "CI 2.5%", "CI 97.5%"]
    print("\nOdds ratios:\n", odds.round(3))
    prob = m.predict(df)
    print(f"Accuracy = {accuracy_score(df[target], prob > 0.5):.3f}, AUC = {roc_auc_score(df[target], prob):.3f}, "
          f"McFadden pseudo-R2 = {m.prsquared:.3f}, LR test p = {m.llr_pvalue:.4f}")
    logit_results[target] = m

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
for target, m in logit_results.items():
    sig = m.pvalues.drop("Intercept")
    sig = sig[sig < ALPHA]
    print(f"{target}: model {'is' if m.llr_pvalue < ALPHA else 'is NOT'} significantly better than a null model (p={m.llr_pvalue:.4f}).")
    if len(sig):
        for term in sig.index:
            print(f"   Significant: {term} (odds ratio = {np.exp(m.params[term]):.2f}, p = {sig[term]:.4f})")
    else:
        print("   No individual predictor is significant at the 5% level.")


# %% ==============================================================
# CHUNK: 11. FACTOR ANALYSIS
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level

factor_dummies = pd.get_dummies(df["Satisfaction_Factor"], prefix="Factor").astype(int)
X = pd.concat([df[["Satisfaction_Score", "Age", "Loyalty_Num", "Purchase_Bin", "Support_Bin", "Gender_Bin", "Group_Bin"]],
               factor_dummies], axis=1)
X_scaled = StandardScaler().fit_transform(X)
corr = np.corrcoef(X_scaled, rowvar=False)
eigvals = np.sort(np.linalg.eigvalsh(corr))[::-1]
n_factors = int(max(1, min((eigvals > 1).sum(), 4)))           # Kaiser criterion (capped at 4)

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
print("Eigenvalues:", np.round(eigvals, 2))
print(f"Factors retained (eigenvalue > 1): {n_factors}")
fa = FactorAnalysis(n_components=n_factors, rotation="varimax", random_state=42).fit(X_scaled)
loadings = pd.DataFrame(fa.components_.T, index=X.columns, columns=[f"Factor{i+1}" for i in range(n_factors)])
print("\nVarimax-rotated loadings:\n", loadings.round(2))

sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Factor Loadings")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "11_factor_loadings.png"), dpi=150); plt.show()
plt.plot(range(1, len(eigvals)+1), eigvals, "o-"); plt.axhline(1, color="red", ls="--")
plt.title("Scree Plot"); plt.xlabel("Component"); plt.ylabel("Eigenvalue")
plt.savefig(os.path.join(OUTPUT_DIR, "11_scree.png"), dpi=150); plt.show()

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
for f in loadings.columns:
    top = loadings[f].abs().sort_values(ascending=False).head(3).index.tolist()
    print(f"{f}: driven mainly by {', '.join(top)}.")
print("Note: most inputs are binary/dummy variables, so treat this as exploratory grouping of variables, not a strict psychometric scale.")


# %% ==============================================================
# CHUNK: 12. CLUSTER ANALYSIS  -  K-Means & Hierarchical
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level

features = ["Satisfaction_Score", "Age", "Loyalty_Num", "Purchase_Bin", "Support_Bin"]
X_scaled = StandardScaler().fit_transform(df[features])

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
# choose k by silhouette score
sil = {k: silhouette_score(X_scaled, KMeans(k, n_init=10, random_state=42).fit_predict(X_scaled)) for k in range(2, 7)}
print("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
best_k = max(sil, key=sil.get)
print("Best k:", best_k)

df["KMeans_Cluster"] = KMeans(best_k, n_init=10, random_state=42).fit_predict(X_scaled) + 1
df["Hier_Cluster"] = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X_scaled) + 1
print("Agreement between K-Means and Hierarchical (ARI):", round(adjusted_rand_score(df["KMeans_Cluster"], df["Hier_Cluster"]), 3))

profile = df.groupby("KMeans_Cluster")[features].mean().round(2)
profile["Size"] = df["KMeans_Cluster"].value_counts().sort_index()
print("\nK-Means cluster profiles:\n", profile)

pcs = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["KMeans_Cluster"], palette="tab10", ax=ax[0])
ax[0].set_title("K-Means clusters (PCA view)")
dendrogram(linkage(X_scaled, "ward"), truncate_mode="lastp", p=20, ax=ax[1]); ax[1].set_title("Hierarchical dendrogram (Ward)")
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "12_clusters.png"), dpi=150); plt.show()

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
print(f"Customers split into {best_k} segments (silhouette = {sil[best_k]:.2f}; >0.5 strong, 0.25-0.5 moderate, <0.25 weak).")
for c, r in profile.iterrows():
    print(f"Cluster {c} (n={int(r['Size'])}): avg satisfaction {r['Satisfaction_Score']}, avg age {r['Age']}, "
          f"loyalty {r['Loyalty_Num']} (1=Low,3=High), {r['Purchase_Bin']*100:.0f}% purchased, {r['Support_Bin']*100:.0f}% contacted support.")


# %% ==============================================================
# CHUNK: 13. CONFIDENCE INTERVALS
# =================================================================

# ---------------------------------------------------------------
# 1. LIBRARY
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------
# 2. DATA CONNECTION / DATA SOURCE
# ---------------------------------------------------------------
DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\jonalyn.rapanot\OneDrive - Robinsons Retail Holdings Inc\01_CI Team\08_Github\04_Customer Satisfaction\dataset.csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(DATA_PATH), "outputs")   # charts are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
raw = pd.read_csv(DATA_PATH)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from: {DATA_PATH}")

# ---------------------------------------------------------------
# 3. DATA CLEANING
# ---------------------------------------------------------------
df = raw.copy()
df.columns = df.columns.str.strip().str.replace(" ", "_")          # tidy column names
for c in df.columns:                                               # trim text values
    if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype("string").str.strip()
df = df.drop_duplicates()
df["Satisfaction_Score"] = pd.to_numeric(df["Satisfaction_Score"], errors="coerce")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Group"] = df["Group"].str.upper()
for c in ["Gender", "Purchase_History", "Support_Contacted", "Loyalty_Level", "Satisfaction_Factor"]:
    df[c] = df[c].str.title()
df = df.dropna(subset=["Satisfaction_Score"])                      # score is the key metric
df = df[df["Satisfaction_Score"].between(1, 10)]                   # valid scale is 1-10
df["Age"] = df["Age"].fillna(df["Age"].median())                   # impute missing age
print("Rows after cleaning:", len(df))
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])

# ---------------------------------------------------------------
# 4. DATA PREPARATION
# ---------------------------------------------------------------
LOYALTY_ORDER = ["Low", "Medium", "High"]
df["Loyalty_Num"] = df["Loyalty_Level"].map({"Low": 1, "Medium": 2, "High": 3})   # ordinal code
df["Loyalty_Level"] = pd.Categorical(df["Loyalty_Level"], categories=LOYALTY_ORDER, ordered=True)
df["Purchase_Bin"] = (df["Purchase_History"] == "Yes").astype(int)
df["Support_Bin"] = (df["Support_Contacted"] == "Yes").astype(int)
df["Gender_Bin"] = (df["Gender"] == "Male").astype(int)
df["Group_Bin"] = (df["Group"] == "B").astype(int)
ALPHA = 0.05   # significance level

CONF = 0.95
rng = np.random.default_rng(42)
def mean_ci(x, conf=CONF):
    x = pd.Series(x).dropna()
    lo, hi = stats.t.interval(conf, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    return len(x), x.mean(), lo, hi

# ---------------------------------------------------------------
# 5. ANALYSIS
# ---------------------------------------------------------------
rows = []
for name, series in [("Satisfaction_Score (overall)", df["Satisfaction_Score"]), ("Age (overall)", df["Age"])]:
    rows.append((name, *mean_ci(series)))
for col in ["Group", "Gender", "Loyalty_Level"]:
    for lvl, sub in df.groupby(col, observed=True):
        rows.append((f"Satisfaction_Score | {col}={lvl}", *mean_ci(sub["Satisfaction_Score"])))
ci_tbl = pd.DataFrame(rows, columns=["Metric", "n", "Mean", "CI_Lower", "CI_Upper"]).round(2)
print(f"{int(CONF*100)}% t-based confidence intervals for the mean:\n", ci_tbl.to_string(index=False))

boot = [rng.choice(df["Satisfaction_Score"], len(df)).mean() for _ in range(5000)]
print("\nBootstrap 95% CI for mean satisfaction:", np.round(np.percentile(boot, [2.5, 97.5]), 2))
boot_med = [np.median(rng.choice(df["Satisfaction_Score"], len(df))) for _ in range(5000)]
print("Bootstrap 95% CI for median satisfaction:", np.round(np.percentile(boot_med, [2.5, 97.5]), 2))

for col in ["Purchase_Bin", "Support_Bin"]:
    ci = stats.binomtest(int(df[col].sum()), len(df)).proportion_ci(confidence_level=CONF, method="wilson")
    lo, hi = ci.low, ci.high
    print(f"{col}: {df[col].mean()*100:.1f}% (Wilson 95% CI {lo*100:.1f}% - {hi*100:.1f}%)")

# ---------------------------------------------------------------
# 6. CONCLUSION
# ---------------------------------------------------------------
overall = ci_tbl.iloc[0]
print(f"We are 95% confident the true average satisfaction score lies between {overall['CI_Lower']} and {overall['CI_Upper']} (sample mean {overall['Mean']}).")
print("Subgroups whose intervals do not overlap likely differ meaningfully; overlapping intervals do not prove they are the same.")