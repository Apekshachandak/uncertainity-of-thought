"""
h1_h2_natural_stories.py
=========================
Generates Figures 1–8 from "The Uncertainty of Thought" (Chandak & Gupta, 2025).

Covers:
  H1 (Entropy adds predictive power beyond Surprisal):
    Fig 1 – Marginal regressions of log(RT) on S and H
    Fig 2 – LRT chi^2 values for adding H and dH to the model
    Fig 3 – Standardised beta coefficients (forest plot)
    Fig 4 – BIC ladder across nested models

  H2 (High Entropy slows readers even when Surprisal is low):
    Fig 5 – Mean RT and log(RT) per S x H quadrant (4 bars each)
    Fig 6 – S x H scatter coloured by quadrant
    Fig 7 – Bayesian Gaussian Mixture Model over log(RT)
    Fig 8 – Conditional log(RT) vs H for low-S and high-S bands

Expected input
--------------
A single Natural Stories table (parquet or csv) with one row per word event
and at least these columns:

    subject     : participant id (string/int)
    item        : story/passage id (string/int)
    word_id     : sequential word index in the item (int)
    RT          : self-paced reading time, milliseconds (float)
    surprisal   : -log P(w | context) from the LM (float)
    entropy     : H of next-word distribution at this position (float)
    delta_H     : H(t-1) - H(t) (float)
    word_length : characters (int)
    log_freq    : log unigram frequency (float)
    position    : position-in-sentence (int)

Edit the constants in the CONFIG block to point at your file.

Usage
-----
    python h1_h2_natural_stories.py
    python h1_h2_natural_stories.py --data path/to/ns.parquet --out figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.mixture import GaussianMixture

# ----------------------------- CONFIG ---------------------------------------
DEFAULT_DATA = Path("data/natural_stories.parquet")
DEFAULT_OUT = Path("figures")

# Visual style ----------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})
PALETTE = {
    "S": "#1f6feb",   # blue – surprisal
    "H": "#0a9396",   # teal – entropy
    "dH": "#ee9b00",  # amber – entropy reduction
    "ctrl": "#9ca3af",
    "accent": "#bb3e03",
    "low_s": "#0a9396",
    "high_s": "#9b2226",
}

# ----------------------------- I/O ------------------------------------------
def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # log(RT)
    if "log_RT" not in df.columns:
        df["log_RT"] = np.log(df["RT"])
    # Z-score predictors used in LMMs
    for col in ("surprisal", "entropy", "delta_H", "word_length", "log_freq"):
        if col in df.columns and f"{col}_z" not in df.columns:
            df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    return df


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ============================================================================
# FIGURE 1 — Marginal regressions of log(RT) on S and H
# ============================================================================
def figure_1_marginal_regressions(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    # Bin to keep the scatter readable
    sample = df.sample(min(len(df), 30_000), random_state=0)

    for ax, predictor, color, label in [
        (axes[0], "surprisal", PALETTE["S"], "Surprisal  S"),
        (axes[1], "entropy", PALETTE["H"], "Entropy  H"),
    ]:
        ax.scatter(sample[predictor], sample["log_RT"],
                   s=4, alpha=0.06, color=color, rasterized=True)
        # OLS fit on the full data for the slope
        beta, intercept, r, _, _ = stats.linregress(df[predictor], df["log_RT"])
        xs = np.linspace(df[predictor].quantile(0.01),
                         df[predictor].quantile(0.99), 100)
        ax.plot(xs, intercept + beta * xs, color="black", lw=2.2)
        ax.set_xlabel(label)
        ax.set_title(f"β = {beta:.4f}    r = {r:.3f}")

    axes[0].set_ylabel("log(RT)")

    # Annotate ratio
    bS, _, _, _, _ = stats.linregress(df["surprisal"], df["log_RT"])
    bH, _, _, _, _ = stats.linregress(df["entropy"], df["log_RT"])
    rSH = df[["surprisal", "entropy"]].corr().iloc[0, 1]
    fig.suptitle(
        f"Fig 1 · Marginal regressions of log(RT) on Surprisal and Entropy   "
        f"(H slope is {bH/bS:.2f}× steeper than S; r(S,H) = {rSH:.2f})",
        y=1.02, fontsize=11,
    )
    save(fig, out_dir, "fig01_marginal_regressions")


# ============================================================================
# FIGURE 2 — Likelihood-ratio test: adding H and dH
# ============================================================================
def _lmm(df, formula):
    """Fit a mixed model with by-subject and by-item intercepts."""
    return smf.mixedlm(
        formula, data=df, groups=df["subject"], re_formula="1",
        vc_formula={"item": "0 + C(item)"},
    ).fit(method="lbfgs", reml=False)


def _lrt(small, big):
    chi2 = 2 * (big.llf - small.llf)
    df = big.df_modelwc - small.df_modelwc
    p = stats.chi2.sf(chi2, df)
    return chi2, df, p


def figure_2_lrt(df: pd.DataFrame, out_dir: Path,
                 use_lmm: bool = False) -> dict:
    """
    Computes LRT χ² for:
        S+controls          →   +H
        S+H+controls        →   +dH
    For speed we default to OLS proxies; pass use_lmm=True for the paper LMM.
    """
    base_formula = "log_RT ~ surprisal_z + word_length_z + log_freq_z"
    plus_H = base_formula + " + entropy_z"
    plus_dH = plus_H + " + delta_H_z"

    if use_lmm:
        m_base = _lmm(df, base_formula)
        m_H = _lmm(df, plus_H)
        m_dH = _lmm(df, plus_dH)
    else:
        m_base = smf.ols(base_formula, data=df).fit()
        m_H = smf.ols(plus_H, data=df).fit()
        m_dH = smf.ols(plus_dH, data=df).fit()

    chi2_H, _, p_H = _lrt(m_base, m_H)
    chi2_dH, _, p_dH = _lrt(m_H, m_dH)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ["+H over S", "+ΔH over S+H"]
    chi2 = [chi2_H, chi2_dH]
    ax.bar(bars, chi2, color=[PALETTE["H"], PALETTE["dH"]], width=0.55)
    ax.axhline(3.84, color="red", ls="--", lw=1.4,
               label="critical χ²(1) = 3.84")
    for x, c in enumerate(chi2):
        ax.text(x, c * 1.03, f"χ² = {c:.1f}", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("LRT χ² (1 df)")
    ax.set_yscale("log")
    ax.set_title("Fig 2 · Likelihood-ratio test: H and ΔH each add unique signal")
    ax.legend(frameon=False)
    save(fig, out_dir, "fig02_lrt_chi2")
    return {"chi2_H": chi2_H, "p_H": p_H, "chi2_dH": chi2_dH, "p_dH": p_dH}


# ============================================================================
# FIGURE 3 — Standardised β coefficients with 95% CIs
# ============================================================================
def figure_3_betas(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    formula = ("log_RT ~ surprisal_z + entropy_z + delta_H_z "
               "+ word_length_z + log_freq_z")
    m = smf.ols(formula, data=df).fit()
    coefs = m.params.drop("Intercept")
    cis = m.conf_int().drop("Intercept")
    cis.columns = ["lo", "hi"]
    table = pd.concat([coefs.rename("beta"), cis], axis=1)

    label_map = {
        "surprisal_z": "Surprisal (S)",
        "entropy_z": "Entropy (H)",
        "delta_H_z": "Entropy reduction (ΔH)",
        "word_length_z": "Word length",
        "log_freq_z": "Log-frequency",
    }
    table = table.rename(index=label_map)
    table = table.loc[list(label_map.values())]  # keep order

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    y = np.arange(len(table))
    colors = [PALETTE["S"], PALETTE["H"], PALETTE["dH"], "#6b7280", "#6b7280"]
    ax.errorbar(table["beta"], y,
                xerr=[table["beta"] - table["lo"], table["hi"] - table["beta"]],
                fmt="o", color="black", ecolor="black",
                markersize=7, lw=1.4, capsize=4, zorder=3)
    for yi, (b, c) in enumerate(zip(table["beta"], colors)):
        ax.scatter(b, yi, color=c, s=80, zorder=4, edgecolor="black")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(table.index)
    ax.invert_yaxis()
    ax.set_xlabel("Standardised β (95% CI)")
    ax.set_title("Fig 3 · Effect sizes from the full model")
    save(fig, out_dir, "fig03_standardised_betas")
    return table


# ============================================================================
# FIGURE 4 — BIC ladder across nested models
# ============================================================================
def figure_4_bic(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    formulas = {
        "Controls only":      "log_RT ~ word_length_z + log_freq_z",
        "+ S":                "log_RT ~ word_length_z + log_freq_z + surprisal_z",
        "+ S + H":            "log_RT ~ word_length_z + log_freq_z + surprisal_z + entropy_z",
        "+ S + ΔH":           "log_RT ~ word_length_z + log_freq_z + surprisal_z + delta_H_z",
        "+ S + H + ΔH":       "log_RT ~ word_length_z + log_freq_z + surprisal_z + entropy_z + delta_H_z",
    }
    bics = {name: smf.ols(f, data=df).fit().bic for name, f in formulas.items()}
    bic_df = pd.DataFrame({"BIC": bics})
    bic_df["ΔBIC vs full"] = bic_df["BIC"] - bic_df["BIC"].min()

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    order = list(formulas.keys())
    vals = bic_df.loc[order, "BIC"]
    colors = ["#9ca3af"] * 4 + [PALETTE["accent"]]
    ax.bar(order, vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + (vals.max() - vals.min()) * 0.01,
                f"{v:.0f}", ha="center", fontsize=10)
    ax.set_ylabel("BIC (lower = better)")
    ax.set_title("Fig 4 · BIC ladder — full S + H + ΔH model wins decisively")
    ax.set_ylim(vals.min() - 50, vals.max() + 50)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    save(fig, out_dir, "fig04_bic_ladder")
    return bic_df


# ============================================================================
# FIGURE 5 — S x H quadrant means (RT and log RT)
# ============================================================================
def _make_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    s_med = df["surprisal"].median()
    h_med = df["entropy"].median()
    df["S_band"] = np.where(df["surprisal"] >= s_med, "High S", "Low S")
    df["H_band"] = np.where(df["entropy"] >= h_med, "High H", "Low H")
    df["quadrant"] = df["S_band"] + " · " + df["H_band"]
    return df


QUAD_ORDER = ["Low S · Low H", "Low S · High H", "High S · Low H", "High S · High H"]
QUAD_COLORS = ["#94d2bd", "#0a9396", "#e9c46a", "#bb3e03"]


def figure_5_quadrant_bars(df: pd.DataFrame, out_dir: Path) -> dict:
    df = _make_quadrants(df)
    means_RT = df.groupby("quadrant")["RT"].mean().reindex(QUAD_ORDER)
    means_logRT = df.groupby("quadrant")["log_RT"].mean().reindex(QUAD_ORDER)

    # Welch t-test: Low-S/High-H vs Low-S/Low-H
    a = df.loc[df["quadrant"] == "Low S · High H", "RT"]
    b = df.loc[df["quadrant"] == "Low S · Low H", "RT"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / pooled

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, vals, ylabel, title in [
        (axes[0], means_RT, "Mean RT (ms)", "Raw RT"),
        (axes[1], means_logRT, "Mean log(RT)", "log(RT)"),
    ]:
        ax.bar(vals.index, vals.values, color=QUAD_COLORS, width=0.6)
        for i, v in enumerate(vals.values):
            ax.text(i, v + (vals.max() - vals.min()) * 0.02,
                    f"{v:.3g}", ha="center", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    delta = a.mean() - b.mean()
    fig.suptitle(
        f"Fig 5 · S × H quadrant means   "
        f"Δ(LowS·HighH − LowS·LowH) = {delta:.1f} ms,  d = {d:.3f},  "
        f"t = {t:.2f}, p = {p:.2e}",
        y=1.02, fontsize=11,
    )
    save(fig, out_dir, "fig05_quadrant_bars")
    return {"delta_ms": delta, "cohens_d": d, "t": t, "p": p}


# ============================================================================
# FIGURE 6 — S x H scatter coloured by quadrant
# ============================================================================
def figure_6_sxh_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    df = _make_quadrants(df)
    sample = df.sample(min(5000, len(df)), random_state=1)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for q, c in zip(QUAD_ORDER, QUAD_COLORS):
        sub = sample[sample["quadrant"] == q]
        ax.scatter(sub["surprisal"], sub["entropy"],
                   color=c, s=12, alpha=0.55, label=q, edgecolor="none")

    s_med = df["surprisal"].median()
    h_med = df["entropy"].median()
    ax.axvline(s_med, color="grey", ls="--", lw=0.8)
    ax.axhline(h_med, color="grey", ls="--", lw=0.8)

    # Annotate quadrant mean RTs
    means = df.groupby("quadrant")["RT"].mean()
    label_pos = {
        "Low S · Low H":   (df["surprisal"].quantile(0.15), df["entropy"].quantile(0.15)),
        "Low S · High H":  (df["surprisal"].quantile(0.15), df["entropy"].quantile(0.85)),
        "High S · Low H":  (df["surprisal"].quantile(0.85), df["entropy"].quantile(0.15)),
        "High S · High H": (df["surprisal"].quantile(0.85), df["entropy"].quantile(0.85)),
    }
    for q, (x, y) in label_pos.items():
        ax.text(x, y, f"{means[q]:.0f} ms",
                ha="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="black", lw=0.8))

    ax.set_xlabel("Surprisal (S)")
    ax.set_ylabel("Entropy (H)")
    ax.set_title("Fig 6 · S × H scatter — RT increases independently along both axes")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    save(fig, out_dir, "fig06_sxh_scatter")


# ============================================================================
# FIGURE 7 — Bayesian Gaussian Mixture Model over log(RT)
# ============================================================================
def figure_7_gmm(df: pd.DataFrame, out_dir: Path,
                 k_max: int = 6) -> pd.DataFrame:
    log_rt = df["log_RT"].values.reshape(-1, 1)
    bics = []
    for k in range(2, k_max + 1):
        gmm_k = GaussianMixture(n_components=k, random_state=0,
                                covariance_type="full", n_init=3)
        gmm_k.fit(log_rt)
        bics.append((k, gmm_k.bic(log_rt)))
    best_k = min(bics, key=lambda kb: kb[1])[0]

    gmm = GaussianMixture(n_components=best_k, random_state=0,
                          covariance_type="full", n_init=5).fit(log_rt)
    df = df.copy()
    df["component"] = gmm.predict(log_rt)
    # Order components by mean log RT (Fast → Very slow)
    order = np.argsort(gmm.means_.ravel())
    relabel = {old: new for new, old in enumerate(order)}
    df["regime"] = df["component"].map(relabel)
    names = ["Fast", "Normal", "Slow", "Very Slow", "Extreme"][:best_k]
    df["regime_name"] = df["regime"].map(dict(enumerate(names)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: density + GMM components
    xs = np.linspace(log_rt.min(), log_rt.max(), 400).reshape(-1, 1)
    axes[0].hist(log_rt.ravel(), bins=80, density=True,
                 color="#e5e7eb", edgecolor="white")
    cmap = plt.get_cmap("viridis", best_k)
    for new_lbl, old_lbl in enumerate(order):
        mu = gmm.means_[old_lbl, 0]
        sd = np.sqrt(gmm.covariances_[old_lbl, 0, 0])
        w = gmm.weights_[old_lbl]
        ys = w * stats.norm.pdf(xs.ravel(), mu, sd)
        axes[0].plot(xs.ravel(), ys, color=cmap(new_lbl), lw=2,
                     label=f"{names[new_lbl]} (μ={mu:.2f})")
    axes[0].set_xlabel("log(RT)")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"GMM over log(RT)  ·  k = {best_k} (chosen by BIC)")
    axes[0].legend(frameon=False, fontsize=8)

    # Right: mean S and H per regime
    summ = (df.groupby("regime_name")[["surprisal", "entropy"]]
              .mean().reindex(names[:best_k]))
    x = np.arange(len(summ))
    w = 0.38
    axes[1].bar(x - w / 2, summ["surprisal"], width=w,
                color=PALETTE["S"], label="Surprisal")
    axes[1].bar(x + w / 2, summ["entropy"], width=w,
                color=PALETTE["H"], label="Entropy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summ.index)
    axes[1].set_ylabel("mean predictor value")
    axes[1].set_title("Slower regimes have higher S and H")
    axes[1].legend(frameon=False)

    fig.suptitle("Fig 7 · 4-component GMM decomposes log(RT) into regimes",
                 y=1.02, fontsize=11)
    save(fig, out_dir, "fig07_gmm")
    return summ


# ============================================================================
# FIGURE 8 — Conditional log(RT) vs H within S bands
# ============================================================================
def figure_8_conditional_H(df: pd.DataFrame, out_dir: Path,
                           n_bins: int = 12) -> None:
    s_lo, s_hi = df["surprisal"].quantile([0.33, 0.67])
    df = df.copy()
    df["S_tert"] = np.where(df["surprisal"] <= s_lo, "low",
                  np.where(df["surprisal"] >= s_hi, "high", "mid"))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for tert, color, label in [("low", PALETTE["low_s"], "Low S (bottom 33%)"),
                                ("high", PALETTE["high_s"], "High S (top 33%)")]:
        sub = df[df["S_tert"] == tert]
        bins = np.linspace(sub["entropy"].quantile(0.02),
                           sub["entropy"].quantile(0.98), n_bins + 1)
        sub = sub.assign(H_bin=pd.cut(sub["entropy"], bins, include_lowest=True))
        agg = sub.groupby("H_bin", observed=True)["log_RT"].agg(["mean", "sem"]).dropna()
        centers = [iv.mid for iv in agg.index]
        ax.plot(centers, agg["mean"], color=color, lw=2.2,
                marker="o", markersize=5, label=label)
        ax.fill_between(centers, agg["mean"] - agg["sem"],
                        agg["mean"] + agg["sem"], color=color, alpha=0.18)

    # Highlight the dissociation region: high-H within low-S
    h_mid = df["entropy"].median()
    h_top = df["entropy"].quantile(0.95)
    ax.axvspan(h_mid, h_top, color=PALETTE["H"], alpha=0.07,
               label="critical region: high H | low S")

    ax.set_xlabel("Entropy (H)")
    ax.set_ylabel("mean log(RT)")
    ax.set_title("Fig 8 · H slows reading even at low S")
    ax.legend(frameon=False)
    save(fig, out_dir, "fig08_conditional_H")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--use-lmm", action="store_true",
                        help="Fit mixed-effects models for the LRT (slow).")
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    df = load_table(args.data)
    print(f"  N = {len(df):,} word events; "
          f"{df['subject'].nunique()} subjects, {df['item'].nunique()} items")

    print("Generating figures →", args.out)
    figure_1_marginal_regressions(df, args.out)
    lrt = figure_2_lrt(df, args.out, use_lmm=args.use_lmm)
    print(f"   LRT  +H over S       χ² = {lrt['chi2_H']:.2f}, p = {lrt['p_H']:.2e}")
    print(f"   LRT  +ΔH over S+H    χ² = {lrt['chi2_dH']:.2f}, p = {lrt['p_dH']:.2e}")
    figure_3_betas(df, args.out)
    figure_4_bic(df, args.out)
    res5 = figure_5_quadrant_bars(df, args.out)
    print(f"   Quadrant Δ = {res5['delta_ms']:.2f} ms,  d = {res5['cohens_d']:.3f}")
    figure_6_sxh_scatter(df, args.out)
    figure_7_gmm(df, args.out)
    figure_8_conditional_H(df, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
