"""
Companion script for the RWDS article:
  "The Hidden Statistics Behind Credit Risk"

Covers all analysis sections of companion.ipynb:
  1. Data loading and preparation
  2. WoE and IV computation
  3. IV standard errors via the delta method
  4. PSI over time with confidence intervals
  5. Performance-fairness trade-off (Pareto frontier)

Figures are saved to ../images/.
"""

# %% [markdown]
# ## 1. Data loading and preparation

from __future__ import annotations

from pathlib import Path
from typing import Any

# %%
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES = PROJECT_ROOT / "images"
IMAGES.mkdir(exist_ok=True)

HF_DATASET = "hf://datasets/deburky/home-credit-credit-risk-model-stability/home_credit_processed.parquet"
LOCAL_DATASET = PROJECT_ROOT / "data" / "processed" / "home_credit_processed.parquet"

# -------------------------------------------------------------------------------
# Feature definitions
# -------------------------------------------------------------------------------
features_dpd: list[str] = [
    "maxdpdfrom6mto36m_3546853P",
    "maxdpdlast12m_727P",
    "maxdpdlast24m_143P",
    "maxdpdlast3m_392P",
    "maxdpdlast6m_474P",
]

features_user_profile: list[str] = [
    "numinstls_657L",
    "numinstlsallpaid_934L",
    "pctinstlsallpaidlat10d_839L",
    "totalsettled_863A",
    "totaldebt_9A",
    "currdebt_22A",
    "credamount_770A",
    "mobilephncnt_593L",
    "homephncnt_628L",
    "numactivecreds_622L",
    "applicationcnt_361L",
    "applications30d_658L",
    "applicationscnt_1086L",
    "applicationscnt_464L",
    "applicationscnt_629L",
    "avgdbddpdlast24m_3658932P",
    "amtinstpaidbefduel24m_4187115A",
    "maxdbddpdlast1m_3658939P",
    "maxdbddpdtollast12m_3658940P",
    "maxdbddpdtollast6m_4187119P",
    "numinstpaidlate1d_3546852L",
]

features_cb: list[str] = [
    "numberofqueries_373L",
    "days120_123L",
    "days180_256L",
    "days30_165L",
    "days90_310L",
    "days360_512L",
]

all_features: list[str] = features_dpd + features_cb + features_user_profile

# -------------------------------------------------------------------------------
# Load dataset (local first, then HuggingFace)
# -------------------------------------------------------------------------------
if LOCAL_DATASET.exists():
    df = pd.read_parquet(LOCAL_DATASET)
    print(f"Loaded from {LOCAL_DATASET}")
else:
    df = pd.read_parquet(HF_DATASET)
    print(f"Loaded from {HF_DATASET}")

print(f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Default rate: {df['target'].mean():.2%}")
print(f"\nSex distribution:\n{df['sex_738L'].value_counts(dropna=False)}")
print(f"\nAge summary:\n{df['age'].describe()}")

# -------------------------------------------------------------------------------
# Train/test split
# -------------------------------------------------------------------------------
X = df[all_features]
y = df["target"]
week = df["WEEK_NUM"]
sex = df["sex_738L"]
age = df["age"]

ix_train, ix_test = train_test_split(X.index, stratify=y, test_size=0.3, random_state=42)
print(f"Train: {len(ix_train):,}  Test: {len(ix_test):,}")


# %% [markdown]
# ## 2. WoE and IV computation
#
# WoE_j = ln(p_{b,j} / p_{g,j})  where b = bad (default), g = good
# IV    = sum_j (p_{b,j} - p_{g,j}) * WoE_j  =  Jeffreys divergence


# %%
def compute_woe_iv(
    feature_values: npt.ArrayLike,
    target: npt.ArrayLike,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """Compute WoE and IV for a single feature using quantile binning.

    Parameters
    ----------
    feature_values : array-like
        Feature column (may contain NaN — gets its own bin).
    target : array-like
        Binary target (1 = bad/default, 0 = good).
    n_bins : int
        Number of quantile bins for continuous features.

    Returns
    -------
    result : DataFrame with columns [bin, count, n_good, n_bad, p_g, p_b, woe, iv_j]
    iv_total : float
    """
    tmp = pd.DataFrame({"x": np.asarray(feature_values), "y": np.asarray(target)})

    # Separate NaN into its own bin
    mask_nan = tmp["x"].isna()
    tmp_valid = tmp[~mask_nan].copy()
    tmp_nan = tmp[mask_nan].copy()

    # Quantile-bin the non-missing values
    if tmp_valid["x"].nunique() <= n_bins:
        tmp_valid["bin"] = tmp_valid["x"].astype(str)
    else:
        tmp_valid["bin"] = pd.qcut(tmp_valid["x"], q=n_bins, duplicates="drop").astype(str)

    if len(tmp_nan) > 0:
        tmp_nan["bin"] = "Missing"
        tmp = pd.concat([tmp_valid, tmp_nan], ignore_index=True)
    else:
        tmp = tmp_valid

    total_good = (tmp["y"] == 0).sum()
    total_bad = (tmp["y"] == 1).sum()

    agg = tmp.groupby("bin")["y"].agg(["count", "sum"]).reset_index()
    agg.columns = ["bin", "count", "n_bad"]
    agg["n_good"] = agg["count"] - agg["n_bad"]

    # Distribution rates (clipped to avoid log(0))
    agg["p_g"] = np.clip(agg["n_good"] / total_good, 1e-15, None)
    agg["p_b"] = np.clip(agg["n_bad"] / total_bad, 1e-15, None)

    agg["woe"] = np.log(agg["p_b"] / agg["p_g"])
    agg["iv_j"] = (agg["p_b"] - agg["p_g"]) * agg["woe"]

    iv_total: float = float(agg["iv_j"].sum())
    return agg, iv_total


# Compute IV for all features on the training set
iv_results = {}
for feat in all_features:
    agg, iv_val = compute_woe_iv(X.loc[ix_train, feat], y.loc[ix_train])
    iv_results[feat] = {"table": agg, "iv": iv_val}

iv_series = pd.Series({f: v["iv"] for f, v in iv_results.items()}).sort_values(ascending=False)
print("\nInformation Value (all features):")
print(iv_series.to_string())


# %% [markdown]
# ## 2b. Why WoE inherits its SE from log odds (location invariance)
#
# WoE is log odds shifted by a constant — the prior log odds:
#
#   WoE_j = θ_j − θ_prior,   where θ_prior = ln(n_bad / n_good)
#
# A basic property of variance is that it is invariant to location shifts:
#
#   Var(X − K) = Var(X)   for any constant K
#
# This is the same property exploited by Welford's algorithm for numerically
# stable variance computation: shifting all values by a constant K avoids
# catastrophic cancellation without changing the result.
#
# Here, θ_prior plays the role of K.  Because it is fixed for the entire
# dataset, subtracting it from each bin's log odds does not alter the
# variance.  Therefore:
#
#   SE(WoE_j) = SE(θ_j) = sqrt(1/n_{j,g} + 1/n_{j,b})
#
# This single fact is the foundation of everything that follows: IV and PSI
# standard errors are just the delta-method propagation of WoE SEs, and
# those SEs come "for free" from log odds because WoE is a centered
# version of log odds.
#
# Reference: Burakov (2025), "Weight of Evidence (WOE), Log Odds, and
# Standard Errors", Technical Report.

# %%
# -------------------------------------------------------------------------------
# Empirical verification (reproduces Burakov 2025, Section 4)
# Simulate log odds and WoE for a single bin to confirm Var(WoE) = Var(θ)
# -------------------------------------------------------------------------------
rng = np.random.default_rng(42)

p_bin = 0.6  # true P(bad) in this bin
n_bin = 50  # bin sample size
prior_p = 0.4  # overall P(bad)
theta_prior = np.log(prior_p / (1 - prior_p))  # constant

n_sim = 10_000
log_odds_samples = []
woe_samples = []

for _ in range(n_sim):
    n_bad = rng.binomial(n_bin, p_bin)
    n_good = n_bin - n_bad
    if n_bad > 0 and n_good > 0:
        theta = np.log(n_bad / n_good)
        log_odds_samples.append(theta)
        woe_samples.append(theta - theta_prior)

var_log_odds = np.var(log_odds_samples, ddof=1)
var_woe = np.var(woe_samples, ddof=1)
theoretical_var = 1 / (n_bin * p_bin) + 1 / (n_bin * (1 - p_bin))

print(f"\nLocation-invariance check (n_sim={n_sim:,}):")
print(f"  Var(log odds) = {var_log_odds:.4f}")
print(f"  Var(WoE)      = {var_woe:.4f}")
print(f"  Theoretical   = {theoretical_var:.4f}")
print(f"  Difference    = {abs(var_log_odds - var_woe):.6f}")


# %% [markdown]
# ## 3. IV standard errors via the delta method
#
# SE(WoE_j) = sqrt(1/n_{j,g} + 1/n_{j,b})
# SE(IV)    ≈ sqrt( sum_j (p_{b,j} - p_{g,j})^2 * SE(WoE_j)^2 )


# %%
def iv_standard_error(
    bin_good: npt.ArrayLike,
    bin_bad: npt.ArrayLike,
    total_good: int,
    total_bad: int,
) -> tuple[float, float]:
    """Compute IV and its standard error from per-bin counts.

    Parameters
    ----------
    bin_good, bin_bad : array-like
        Number of good / bad observations in each bin.
    total_good, total_bad : int
        Total good / bad in the population.

    Returns
    -------
    iv : float
    se_iv : float
    """
    bg = np.asarray(bin_good, dtype=float)
    bb = np.asarray(bin_bad, dtype=float)

    p_g = bg / total_good
    p_b = bb / total_bad

    woe = np.log(np.clip(p_b, 1e-15, None) / np.clip(p_g, 1e-15, None))
    weight = p_b - p_g

    iv = float((weight * woe).sum())

    se_woe = np.sqrt(1.0 / np.clip(bg, 1, None) + 1.0 / np.clip(bb, 1, None))
    se_iv = float(np.sqrt((weight**2 * se_woe**2).sum()))

    return iv, se_iv


# Compute IV + SE for every feature
iv_se_records = []
total_good_train = (y.loc[ix_train] == 0).sum()
total_bad_train = (y.loc[ix_train] == 1).sum()

for feat in all_features:
    tbl = iv_results[feat]["table"]
    iv_val, se_val = iv_standard_error(
        tbl["n_good"].values, tbl["n_bad"].values, total_good_train, total_bad_train
    )
    z_score = iv_val / se_val if se_val > 0 else np.inf
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    iv_se_records.append(
        {
            "feature": feat,
            "iv": iv_val,
            "se": se_val,
            "ci_lower": max(0, iv_val - 1.96 * se_val),
            "ci_upper": iv_val + 1.96 * se_val,
            "z_score": z_score,
            "p_value": p_value,
        }
    )

iv_df = pd.DataFrame(iv_se_records).sort_values("iv", ascending=True).reset_index(drop=True)

print("\nIV with standard errors (sorted ascending):")
print(iv_df[["feature", "iv", "se", "ci_lower", "ci_upper", "z_score", "p_value"]].to_string())


# %%  Figure: IV bar chart with 95% confidence intervals
fig, ax = plt.subplots(figsize=(8, 8))

y_pos = np.arange(len(iv_df))
ax.barh(
    y_pos,
    iv_df["iv"],
    xerr=1.96 * iv_df["se"],
    height=0.6,
    color="#5b9bd5",
    edgecolor="white",
    capsize=3,
    error_kw={"linewidth": 1.0, "color": "#333333"},
)
ax.set_yticks(y_pos)
ax.set_yticklabels(iv_df["feature"], fontsize=8)
ax.set_xlabel("Information Value")
ax.set_title("IV with 95% Confidence Intervals (Delta Method)")

# Reference lines for IV interpretation thresholds
for threshold, label in [(0.02, "Weak"), (0.1, "Medium"), (0.3, "Strong")]:
    ax.axvline(threshold, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(threshold, len(iv_df) - 0.5, f" {label}", fontsize=7, color="grey", va="top")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(IMAGES / "iv_bar_chart.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {IMAGES / 'iv_bar_chart.png'}")


# %% [markdown]
# ## 4. PSI over time with confidence intervals
#
# PSI between reference P and comparison Q:
#   PSI = sum_j (q_j - p_j) * ln(q_j / p_j)
#
# This is the same Jeffreys divergence, now across time windows.


# %%
def compute_psi_with_se(
    ref_counts: npt.ArrayLike,
    comp_counts: npt.ArrayLike,
) -> tuple[float, float]:
    """Compute PSI and its delta-method SE from bin counts.

    Parameters
    ----------
    ref_counts, comp_counts : array-like
        Counts per bin for reference and comparison periods.

    Returns
    -------
    psi : float
    se_psi : float
    """
    ref = np.asarray(ref_counts, dtype=float)
    comp = np.asarray(comp_counts, dtype=float)

    # Proportions (clipped)
    p = np.clip(ref / ref.sum(), 1e-15, None)
    q = np.clip(comp / comp.sum(), 1e-15, None)

    ln_ratio = np.log(q / p)
    psi = float(((q - p) * ln_ratio).sum())

    # SE via delta method (same structure as IV SE)
    # Var(ln(q_j/p_j)) ≈ 1/ref_j + 1/comp_j
    se_ln = np.sqrt(1.0 / np.clip(ref, 1, None) + 1.0 / np.clip(comp, 1, None))
    se_psi = float(np.sqrt(((q - p) ** 2 * se_ln**2).sum()))

    return psi, se_psi


def bin_feature_for_psi(series: pd.Series, n_bins: int = 10) -> pd.Series:
    """Bin a feature into quantile categories, returning bin labels."""
    if series.nunique() <= n_bins:
        return series.fillna(-999).astype(str)
    return pd.qcut(series.rank(method="first"), q=n_bins, duplicates="drop").astype(str)


# Pick a top-IV feature for PSI illustration
top_feature = iv_df.iloc[-1]["feature"]
print(f"\nPSI analysis for top feature: {top_feature}")

# Define reference period: first 4 weeks
weeks_sorted = sorted(df["WEEK_NUM"].unique())
ref_weeks = weeks_sorted[:4]
ref_mask = df["WEEK_NUM"].isin(ref_weeks)

# Bin the feature on the reference period
ref_series = df.loc[ref_mask, top_feature]
all_series = df[top_feature]

# Create shared bins from the full dataset
n_psi_bins: int = 10


def _bin_by_value(s: pd.Series) -> pd.Series:
    return s.fillna(-999).astype(str)


def _make_cut_binner(edges: npt.NDArray[Any]) -> Any:
    def _bin(s: pd.Series) -> pd.Series:
        return pd.cut(s.fillna(-999), bins=edges).astype(str)
    return _bin


if all_series.nunique() <= n_psi_bins:
    bin_func = _bin_by_value
else:
    _, shared_edges = pd.qcut(all_series.dropna(), q=n_psi_bins, duplicates="drop", retbins=True)
    shared_edges[0] = -np.inf
    shared_edges[-1] = np.inf
    bin_func = _make_cut_binner(shared_edges)

# Reference bin counts
ref_binned = bin_func(ref_series)
ref_counts_map = ref_binned.value_counts()

# Compute PSI for each subsequent week
psi_records = []
for w in weeks_sorted:
    if w in ref_weeks:
        continue
    w_mask = df["WEEK_NUM"] == w
    w_series = df.loc[w_mask, top_feature]
    if len(w_series) < 50:
        continue

    w_binned = bin_func(w_series)
    w_counts_map = w_binned.value_counts()

    # Align bins
    all_bins = sorted(set(ref_counts_map.index) | set(w_counts_map.index))
    ref_c = np.array([ref_counts_map.get(b, 0) for b in all_bins], dtype=float)
    w_c = np.array([w_counts_map.get(b, 0) for b in all_bins], dtype=float)

    psi_val, se_val = compute_psi_with_se(ref_c, w_c)
    psi_records.append(
        {
            "week": w,
            "psi": psi_val,
            "se": se_val,
            "ci_lower": max(0, psi_val - 1.96 * se_val),
            "ci_upper": psi_val + 1.96 * se_val,
            "n_obs": len(w_series),
        }
    )

psi_df = pd.DataFrame(psi_records)
print(f"PSI computed for {len(psi_df)} weeks")
print(psi_df.head(10).to_string())

# %%  Figure: PSI over time with confidence bands
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(psi_df["week"], psi_df["psi"], "o-", color="#5b9bd5", markersize=4, linewidth=1.5)
ax.fill_between(
    psi_df["week"],
    psi_df["ci_lower"],
    psi_df["ci_upper"],
    alpha=0.2,
    color="#5b9bd5",
    label="95% CI",
)

# PSI interpretation thresholds
ax.axhline(0.1, color="orange", linestyle="--", linewidth=1, label="Minor shift (0.1)")
ax.axhline(0.25, color="red", linestyle="--", linewidth=1, label="Major shift (0.25)")

ax.set_xlabel("Week Number")
ax.set_ylabel("PSI")
ax.set_title(
    f"Population Stability Index Over Time — {top_feature}\n(Reference: weeks {ref_weeks})"
)
ax.legend(loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(IMAGES / "psi_over_time.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {IMAGES / 'psi_over_time.png'}")


# %% [markdown]
# ## 5. Performance-fairness trade-off
#
# The dual IV formulation:
# - **Maximise** IV w.r.t. target (predictive power)
# - **Minimise** IV w.r.t. protected attributes (fairness)
#
# With SE, hard thresholds become probabilistic:
#   P(IV_protected ≤ ε) ≥ 95%

# %%
# We simulate a fairness analysis by treating a feature group as a
# pseudo-protected attribute. For each feature, we compute:
#   - Predictive IV: IV w.r.t. the default target
#   - Its SE and 95% CI
#
# Then we show the confidence-bounded Pareto frontier.

# Build a feature-level summary with predictive IV and SE
pareto_df = iv_df[["feature", "iv", "se", "ci_lower", "ci_upper"]].copy()
pareto_df = pareto_df.rename(columns={"iv": "predictive_iv", "se": "predictive_se"})

# For the article, we show how the Pareto concept works:
# conservative performance = IV - 1.96*SE (lower bound)
pareto_df["conservative_iv"] = pareto_df["ci_lower"]

print("\nFeature summary for Pareto analysis:")
print(pareto_df.to_string())


# %%  Figure: Nominal vs conservative IV (Pareto concept illustration)
pareto_sorted = pareto_df.sort_values("predictive_iv", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 8))
y_pos = np.arange(len(pareto_sorted))

# Nominal IV
ax.barh(
    y_pos + 0.15,
    pareto_sorted["predictive_iv"],
    height=0.3,
    color="#5b9bd5",
    label="Nominal IV",
    edgecolor="white",
)
# Conservative IV (lower 95% bound)
ax.barh(
    y_pos - 0.15,
    pareto_sorted["conservative_iv"],
    height=0.3,
    color="#ed7d31",
    label="Conservative IV (95% lower)",
    edgecolor="white",
)

ax.set_yticks(y_pos)
ax.set_yticklabels(pareto_sorted["feature"], fontsize=8)
ax.set_xlabel("Information Value")
ax.set_title("Nominal vs. Conservative IV\n(Performance Lower Bound via Delta Method)")
ax.legend(loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(IMAGES / "pareto_nominal_vs_conservative.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {IMAGES / 'pareto_nominal_vs_conservative.png'}")


# %% Figure: IV hypothesis test — Z-scores
fig, ax = plt.subplots(figsize=(8, 6))

z_df = iv_df.sort_values("z_score", ascending=True).reset_index(drop=True)
y_pos = np.arange(len(z_df))
colors = ["#5b9bd5" if p < 0.05 else "#cccccc" for p in z_df["p_value"]]

ax.barh(y_pos, z_df["z_score"], color=colors, height=0.6, edgecolor="white")
ax.axvline(1.96, color="red", linestyle="--", linewidth=1, label="Z = 1.96 (p = 0.05)")
ax.set_yticks(y_pos)
ax.set_yticklabels(z_df["feature"], fontsize=8)
ax.set_xlabel("Z-score (IV / SE)")
ax.set_title("Statistical Significance of Feature IV\n(H₀: IV = 0, two-sided test)")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(IMAGES / "iv_z_scores.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {IMAGES / 'iv_z_scores.png'}")

# %%
print(f"\nDone. Figures saved to {IMAGES.resolve()}")
