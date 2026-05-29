"""
dundee_replication.py
=====================
Generates Figures 14–16 from "The Uncertainty of Thought" (Chandak & Gupta, 2025),
covering the Dundee eye-tracking replication of H1 and H2.

  Fig 14 – Standardised β coefficients in Natural Stories vs Dundee
  Fig 15 – Quadrant means in both corpora + Cohen's d for each contrast
  Fig 16 – First-pass vs total fixation β in Dundee + cross-corpus β scatter

Expected input
--------------
TWO tables (parquet/csv). Both must be word-level, with these columns:

  Natural Stories:
    subject, item, RT, surprisal, entropy, delta_H, word_length, log_freq

  Dundee:
    subject, item, first_pass_RT, total_fixation_RT, surprisal, entropy,
    delta_H, word_length, log_freq

(`first_pass_RT` is used as the primary Dundee RT measure; `total_fixation_RT`
is used in the right panel of Fig 16.)

Usage
-----
    python dundee_replication.py
    python dundee_replication.py \\
        --ns data/natural_stories.parquet \\
        --dundee data/dundee.parquet \\
        --out figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# ----------------------------- CONFIG ---------------------------------------
DEFAULT_NS = Path("data/natural_stories.parquet")
DEFAULT_DUNDEE = Path("data/dundee.parquet")
DEFAULT_OUT = Path("figures")

PREDICTORS = ["surprisal", "entropy", "delta_H", "word_length", "log_freq"]
LABELS = {
    "surprisal": "Surprisal (S)",
    "entropy": "Entropy (H)",
    "delta_H": "Entropy reduction (ΔH)",
    "word_length": "Word length",
    "log_freq": "Log-frequency",
}

# Visual style ---------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})
COLOR_NS = "#1f6feb"      # Natural Stories – blue
COLOR_DUN = "#7c3aed"     # Dundee – purple
COLOR_FP = "#0a9396"      # first-pass – teal
COLOR_TF = "#ee9b00"      # total fixation – amber

QUAD_ORDER = ["Low S · Low H", "Low S · High H",
              "High S · Low H", "High S · High H"]
QUAD_COLORS = ["#94d2bd", "#0a9396", "#e9c46a", "#bb3e03"]


# ----------------------------- I/O ------------------------------------------
def _read(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def prep(df: pd.DataFrame, rt_col: str) -> pd.DataFrame:
    df = df.copy()
    df["log_RT"] = np.log(df[rt_col])
    for c in PREDICTORS:
        if c in df.columns:
            df[f"{c}_z"] = (df[c] - df[c].mean()) / df[c].std(ddof=0)
    return df


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------- model ---------------------------------------
def fit_full_model(df: pd.DataFrame) -> pd.DataFrame:
    """OLS with z-scored predictors. Returns a tidy DataFrame of β + 95% CI."""
    rhs = " + ".join(f"{c}_z" for c in PREDICTORS if f"{c}_z" in df.columns)
    m = smf.ols(f"log_RT ~ {rhs}", data=df).fit()
    coefs = m.params.drop("Intercept")
    ci = m.conf_int().drop("Intercept")
    out = pd.DataFrame({
        "predictor": [c.replace("_z", "") for c in coefs.index],
        "beta": coefs.values,
        "ci_lo": ci[0].values,
        "ci_hi": ci[1].values,
        "p": m.pvalues.drop("Intercept").values,
    })
    out["label"] = out["predictor"].map(LABELS)
    return out


# ----------------------------- LRT helpers ---------------------------------
def _lrt(small, big):
    chi2 = 2 * (big.llf - small.llf)
    p = stats.chi2.sf(chi2, big.df_modelwc - small.df_modelwc)
    return chi2, p


def lrt_stats(df: pd.DataFrame) -> dict:
    base = "log_RT ~ surprisal_z + word_length_z + log_freq_z"
    plus_H = base + " + entropy_z"
    plus_dH = plus_H + " + delta_H_z"
    m_b = smf.ols(base, data=df).fit()
    m_H = smf.ols(plus_H, data=df).fit()
    m_dH = smf.ols(plus_dH, data=df).fit()
    chi2_H, p_H = _lrt(m_b, m_H)
    chi2_dH, p_dH = _lrt(m_H, m_dH)
    return {"chi2_H": chi2_H, "p_H": p_H, "chi2_dH": chi2_dH, "p_dH": p_dH}


# ============================================================================
# FIGURE 14 — β coefficients NS vs Dundee
# ============================================================================
def figure_14_betas(ns_betas: pd.DataFrame, dun_betas: pd.DataFrame,
                    out_dir: Path) -> None:
    keep = ["surprisal", "entropy", "delta_H"]
    ns = ns_betas.set_index("predictor").loc[keep]
    dn = dun_betas.set_index("predictor").loc[keep]

    x = np.arange(len(keep))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - w / 2, ns["beta"], width=w,
           yerr=[ns["beta"] - ns["ci_lo"], ns["ci_hi"] - ns["beta"]],
           color=COLOR_NS, label="Natural Stories", capsize=4)
    ax.bar(x + w / 2, dn["beta"], width=w,
           yerr=[dn["beta"] - dn["ci_lo"], dn["ci_hi"] - dn["beta"]],
           color=COLOR_DUN, label="Dundee", capsize=4)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in keep])
    ax.set_ylabel("Standardised β")
    ax.set_title("Fig 14 · Effect sizes replicate across paradigms")
    ax.legend(frameon=False)
    save(fig, out_dir, "fig14_betas_ns_vs_dundee")


# ============================================================================
# FIGURE 15 — Quadrant means + Cohen's d in both corpora
# ============================================================================
def _quadrants(df: pd.DataFrame, rt_col: str) -> pd.DataFrame:
    df = df.copy()
    s_med = df["surprisal"].median()
    h_med = df["entropy"].median()
    df["S_band"] = np.where(df["surprisal"] >= s_med, "High S", "Low S")
    df["H_band"] = np.where(df["entropy"] >= h_med, "High H", "Low H")
    df["quadrant"] = df["S_band"] + " · " + df["H_band"]
    df["__rt"] = df[rt_col]
    return df


def _cohens_d(a, b):
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled


def figure_15_quadrant_replication(ns: pd.DataFrame, dun: pd.DataFrame,
                                    ns_rt: str, dun_rt: str,
                                    out_dir: Path) -> dict:
    ns_q = _quadrants(ns, ns_rt)
    dn_q = _quadrants(dun, dun_rt)
    ns_means = ns_q.groupby("quadrant")["__rt"].mean().reindex(QUAD_ORDER)
    dn_means = dn_q.groupby("quadrant")["__rt"].mean().reindex(QUAD_ORDER)

    # Cohen's d for the three contrasts
    contrasts = [
        ("Low S · High H", "Low S · Low H"),
        ("High S · High H", "High S · Low H"),
        ("High S · Low H", "Low S · Low H"),
    ]
    labels = ["LowS·HighH\nvs LowS·LowH",
              "HighS·HighH\nvs HighS·LowH",
              "HighS·LowH\nvs LowS·LowH"]
    d_ns, d_dun = [], []
    for hi, lo in contrasts:
        d_ns.append(_cohens_d(ns_q.loc[ns_q["quadrant"] == hi, "__rt"],
                              ns_q.loc[ns_q["quadrant"] == lo, "__rt"]))
        d_dun.append(_cohens_d(dn_q.loc[dn_q["quadrant"] == hi, "__rt"],
                               dn_q.loc[dn_q["quadrant"] == lo, "__rt"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: quadrant mean RT in both corpora (paired bars per quadrant)
    x = np.arange(len(QUAD_ORDER))
    w = 0.36
    # Normalize Dundee to NS scale so they can sit side by side?
    # Better: dual y-axis since units differ.
    ax1 = axes[0]
    ax1b = ax1.twinx()
    ax1.bar(x - w / 2, ns_means.values, width=w, color=COLOR_NS,
            label="Natural Stories (RT, ms)")
    ax1b.bar(x + w / 2, dn_means.values, width=w, color=COLOR_DUN,
             label="Dundee (first-pass, ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(QUAD_ORDER, rotation=15, ha="right")
    ax1.set_ylabel("NS mean RT (ms)", color=COLOR_NS)
    ax1b.set_ylabel("Dundee first-pass (ms)", color=COLOR_DUN)
    ax1.set_title("Quadrant means")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper left")

    # Right: Cohen's d in both corpora
    x = np.arange(len(labels))
    axes[1].bar(x - w / 2, d_ns, width=w, color=COLOR_NS, label="Natural Stories")
    axes[1].bar(x + w / 2, d_dun, width=w, color=COLOR_DUN, label="Dundee")
    axes[1].axhline(0, color="grey", lw=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Cohen's d")
    axes[1].set_title("Effect sizes per contrast")
    axes[1].legend(frameon=False)

    fig.suptitle("Fig 15 · Quadrant dissociation replicates in Dundee",
                 y=1.02, fontsize=11)
    save(fig, out_dir, "fig15_quadrant_replication")
    return {"d_ns": d_ns, "d_dundee": d_dun}


# ============================================================================
# FIGURE 16 — First-pass vs total fixation in Dundee + cross-corpus β scatter
# ============================================================================
def figure_16_first_vs_total(dun: pd.DataFrame, ns_betas: pd.DataFrame,
                              dun_betas_fp: pd.DataFrame,
                              out_dir: Path) -> dict:
    if "total_fixation_RT" not in dun.columns:
        print("  [Fig 16 left] skipped — no 'total_fixation_RT' in Dundee data")
        dun_betas_tf = None
    else:
        dun_tf = prep(dun, "total_fixation_RT")
        dun_betas_tf = fit_full_model(dun_tf)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Left: first-pass vs total fixation β (Dundee)
    keep = ["surprisal", "entropy", "delta_H"]
    fp = dun_betas_fp.set_index("predictor").loc[keep]
    if dun_betas_tf is not None:
        tf = dun_betas_tf.set_index("predictor").loc[keep]
        x = np.arange(len(keep))
        w = 0.36
        axes[0].bar(x - w / 2, fp["beta"], width=w,
                    yerr=[fp["beta"] - fp["ci_lo"], fp["ci_hi"] - fp["beta"]],
                    color=COLOR_FP, label="First-pass", capsize=4)
        axes[0].bar(x + w / 2, tf["beta"], width=w,
                    yerr=[tf["beta"] - tf["ci_lo"], tf["ci_hi"] - tf["beta"]],
                    color=COLOR_TF, label="Total fixation", capsize=4)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([LABELS[k] for k in keep])
        axes[0].set_ylabel("Standardised β")
        axes[0].axhline(0, color="grey", lw=0.7)
        axes[0].set_title("Dundee · first-pass vs total fixation")
        axes[0].legend(frameon=False)
    else:
        axes[0].set_visible(False)

    # Right: cross-corpus β scatter
    merged = (ns_betas.set_index("predictor")[["beta"]]
              .join(dun_betas_fp.set_index("predictor")[["beta"]],
                    lsuffix="_ns", rsuffix="_dun"))
    r, p = stats.pearsonr(merged["beta_ns"], merged["beta_dun"])
    axes[1].scatter(merged["beta_ns"], merged["beta_dun"],
                    s=110, color="#1f2937", zorder=3)
    for name, row in merged.iterrows():
        axes[1].annotate(LABELS.get(name, name),
                         (row["beta_ns"], row["beta_dun"]),
                         xytext=(6, 6), textcoords="offset points", fontsize=9)
    lo = min(merged["beta_ns"].min(), merged["beta_dun"].min()) - 0.005
    hi = max(merged["beta_ns"].max(), merged["beta_dun"].max()) + 0.005
    axes[1].plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1)
    axes[1].axvline(0, color="grey", lw=0.5)
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].set_xlabel("β · Natural Stories")
    axes[1].set_ylabel("β · Dundee (first-pass)")
    axes[1].set_title(f"Cross-corpus β   r = {r:.3f}, p = {p:.2e}")

    fig.suptitle("Fig 16 · Effects survive at first-pass and replicate across corpora",
                 y=1.02, fontsize=11)
    save(fig, out_dir, "fig16_first_pass_vs_total")
    return {"cross_corpus_r": r, "cross_corpus_p": p}


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=Path, default=DEFAULT_NS)
    parser.add_argument("--dundee", type=Path, default=DEFAULT_DUNDEE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ns-rt-col", default="RT")
    parser.add_argument("--dundee-rt-col", default="first_pass_RT")
    args = parser.parse_args()

    print(f"Loading {args.ns} ...")
    ns = prep(_read(args.ns), args.ns_rt_col)
    print(f"  N = {len(ns):,}")
    print(f"Loading {args.dundee} ...")
    dun = prep(_read(args.dundee), args.dundee_rt_col)
    print(f"  N = {len(dun):,}")

    print("Fitting full models …")
    ns_betas = fit_full_model(ns)
    dun_betas = fit_full_model(dun)
    print("\nNatural Stories β:")
    print(ns_betas[["predictor", "beta", "p"]].to_string(index=False))
    print("\nDundee β (first-pass):")
    print(dun_betas[["predictor", "beta", "p"]].to_string(index=False))

    print("\nDundee LRT statistics:")
    lrt_d = lrt_stats(dun)
    print(f"   +H over S       χ² = {lrt_d['chi2_H']:.2f},  p = {lrt_d['p_H']:.2e}")
    print(f"   +ΔH over S+H    χ² = {lrt_d['chi2_dH']:.2f},  p = {lrt_d['p_dH']:.2e}")

    print("\nGenerating figures →", args.out)
    figure_14_betas(ns_betas, dun_betas, args.out)
    figure_15_quadrant_replication(ns, dun,
                                    args.ns_rt_col, args.dundee_rt_col,
                                    args.out)
    figure_16_first_vs_total(dun, ns_betas, dun_betas, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
