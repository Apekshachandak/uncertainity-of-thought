"""
h3_garden_path.py
=================
Generates Figures 9–13 from "The Uncertainty of Thought" (Chandak & Gupta, 2025),
covering the garden-path analyses for H3 (Entropy spike precedes Surprisal spike).

  Fig 9  – H(t) and S(t) curves for 4 representative garden-path sentences
  Fig 10 – Lead-time distribution (GP vs control) + boxplots by construction
  Fig 11 – Grand-averaged H and S aligned to the disambiguating word
  Fig 12 – ΔH aligned to disambiguation: GP vs control
  Fig 13 – Full temporal cascade: H spike → RT elevation → S spike

Plus statistics: Wilcoxon signed-rank on peak alignment, and cross-lag
correlations of H(t-k) with S(t=0).

Expected input
--------------
A long-format table (parquet/csv) with one row per (item, position) covering
both garden-path items and matched controls. Required columns:

    item_id          : unique id for the sentence (str/int)
    is_garden_path   : bool/int (1 = GP, 0 = matched control)
    construction     : str ('object_RC', 'subject_RC', 'NPZ', 'main_verb', ...)
    position         : integer position-in-sentence (0-indexed)
    disambig_pos     : the integer position of the disambiguating word
                        for THIS item (constant within an item)
    word             : token (str)              [optional, used in Fig 9 labels]
    H                : entropy at this position
    S                : surprisal at this position
    delta_H          : H(t-1) - H(t)            [optional; computed if missing]
    RT               : mean human reading time at this position [optional, Fig 13]

A `representative_items` argument (4 GP item ids) controls Fig 9. If not
provided, the script picks 4 items — one per construction type — with the
clearest H→S lead.

Usage
-----
    python h3_garden_path.py
    python h3_garden_path.py --data data/garden_path.parquet --out figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------- CONFIG ---------------------------------------
DEFAULT_DATA = Path("data/garden_path.parquet")
DEFAULT_OUT = Path("figures")

ALIGN_WINDOW = (-5, 5)   # positions relative to disambiguation
CONSTR_ORDER = ["object_RC", "subject_RC", "NPZ", "main_verb"]
CONSTR_LABELS = {
    "object_RC": "Object-extracted RC",
    "subject_RC": "Subject-extracted RC",
    "NPZ": "NP/Z ambiguity",
    "main_verb": "Temp. ambiguous main verb",
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
COLOR_GP = "#0a9396"     # teal
COLOR_CTRL = "#ee9b00"   # gold
COLOR_H = "#0a9396"
COLOR_S = "#bb3e03"
BAND_H = "#94d2bd"
BAND_S = "#f4a261"


# ----------------------------- I/O ------------------------------------------
def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["is_garden_path"] = df["is_garden_path"].astype(int)
    if "delta_H" not in df.columns:
        df = df.sort_values(["item_id", "position"])
        # ΔH = H(t-1) - H(t)  ==  -diff(H)
        df["delta_H"] = -df.groupby("item_id")["H"].diff().fillna(0)
    df["rel_pos"] = df["position"] - df["disambig_pos"]
    return df


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ============================================================================
# FIGURE 9 — Four representative garden-path sentences
# ============================================================================
def figure_9_example_sentences(df: pd.DataFrame, out_dir: Path,
                               representative_items: list | None = None
                               ) -> list:
    gp = df[df["is_garden_path"] == 1].copy()

    if representative_items is None:
        # Pick one item per construction with the largest positive H→S lead
        chosen = []
        for c in CONSTR_ORDER:
            sub = gp[gp["construction"] == c]
            if sub.empty:
                continue
            leads = []
            for iid, g in sub.groupby("item_id"):
                g = g.sort_values("position")
                if g["H"].notna().sum() < 3:
                    continue
                lead = (g.loc[g["H"].idxmax(), "rel_pos"]
                        - g.loc[g["S"].idxmax(), "rel_pos"])
                # We want H to PRECEDE S, so lead should be negative (H pos − S pos < 0)
                leads.append((iid, -lead))  # bigger = stronger lead
            if leads:
                leads.sort(key=lambda x: x[1], reverse=True)
                chosen.append(leads[0][0])
        representative_items = chosen[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    axes = axes.ravel()
    for ax, iid in zip(axes, representative_items):
        g = gp[gp["item_id"] == iid].sort_values("position")
        if g.empty:
            ax.set_visible(False)
            continue
        constr = g["construction"].iloc[0]

        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax.plot(g["rel_pos"], g["H"], color=COLOR_H, lw=2.2,
                marker="o", markersize=5, label="H")
        ax2.plot(g["rel_pos"], g["S"], color=COLOR_S, lw=2.2,
                 marker="s", markersize=5, label="S")

        # Bands at the peak positions
        h_peak = g.loc[g["H"].idxmax(), "rel_pos"]
        s_peak = g.loc[g["S"].idxmax(), "rel_pos"]
        ax.axvspan(h_peak - 0.4, h_peak + 0.4, color=BAND_H, alpha=0.35,
                   label=f"H peak (Δ={h_peak:+d})")
        ax.axvspan(s_peak - 0.4, s_peak + 0.4, color=BAND_S, alpha=0.35,
                   label=f"S peak (Δ={s_peak:+d})")
        ax.axvline(0, color="black", lw=0.8, ls=":")

        if "word" in g.columns:
            ax.set_xticks(g["rel_pos"])
            ax.set_xticklabels(g["word"], rotation=40, ha="right", fontsize=8)

        ax.set_ylabel("H", color=COLOR_H)
        ax2.set_ylabel("S", color=COLOR_S)
        ax.set_title(f"{CONSTR_LABELS.get(constr, constr)}  ·  item {iid}")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax2.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle("Fig 9 · H(t) and S(t) for four representative garden-path items",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, out_dir, "fig09_example_sentences")
    return representative_items


# ============================================================================
# Helper: per-item lead time δ = (H peak position) − (S peak position) in REL coords
# ============================================================================
def _lead_times(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (iid, gp), g in df.groupby(["item_id", "is_garden_path"]):
        g = g.sort_values("position")
        if g["H"].notna().sum() == 0 or g["S"].notna().sum() == 0:
            continue
        h_pk = g.loc[g["H"].idxmax(), "rel_pos"]
        s_pk = g.loc[g["S"].idxmax(), "rel_pos"]
        rows.append({
            "item_id": iid,
            "is_garden_path": gp,
            "construction": g["construction"].iloc[0],
            "h_peak": h_pk,
            "s_peak": s_pk,
            "delta": s_pk - h_pk,   # positive = H leads S
        })
    return pd.DataFrame(rows)


# ============================================================================
# FIGURE 10 — Lead-time distributions
# ============================================================================
def figure_10_lead_times(df: pd.DataFrame, out_dir: Path) -> dict:
    leads = _lead_times(df)
    gp = leads[leads["is_garden_path"] == 1]
    ctrl = leads[leads["is_garden_path"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: histogram
    bins = np.arange(leads["delta"].min() - 0.5,
                     leads["delta"].max() + 1.5, 1)
    axes[0].hist(ctrl["delta"], bins=bins, color=COLOR_CTRL, alpha=0.7,
                 label=f"Controls (n={len(ctrl)})", edgecolor="white")
    axes[0].hist(gp["delta"], bins=bins, color=COLOR_GP, alpha=0.8,
                 label=f"Garden-path (n={len(gp)})", edgecolor="white")
    axes[0].axvline(0, color="black", lw=1, ls="--")
    axes[0].axvline(gp["delta"].mean(), color=COLOR_GP, lw=2,
                    label=f"GP mean δ = {gp['delta'].mean():+.2f}")
    t, p = stats.ttest_1samp(gp["delta"], 0)
    axes[0].set_xlabel("lead time δ = S peak − H peak (words)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Lead-time distribution  (t = {t:.2f}, p = {p:.2e})")
    axes[0].legend(frameon=False, fontsize=9)

    # Right: boxplot per construction
    constr_present = [c for c in CONSTR_ORDER if c in gp["construction"].unique()]
    data = [gp.loc[gp["construction"] == c, "delta"] for c in constr_present]
    bp = axes[1].boxplot(data, patch_artist=True,
                         tick_labels=[CONSTR_LABELS[c] for c in constr_present])
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_GP); patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.6)
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set_ylabel("lead time δ (words)")
    axes[1].set_title("Lead time by construction")
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Fig 10 · Garden-path items show a reliable positive lead",
                 y=1.02, fontsize=11)
    save(fig, out_dir, "fig10_lead_time_distribution")
    return {"mean_delta_gp": gp["delta"].mean(),
            "t": t, "p": p, "n_gp": len(gp), "n_ctrl": len(ctrl)}


# ============================================================================
# FIGURE 11 — Grand-averaged H and S aligned to disambiguation
# ============================================================================
def _aligned_means(df: pd.DataFrame, value: str,
                   gp_only: bool = True) -> pd.DataFrame:
    sub = df[df["is_garden_path"] == 1] if gp_only else df
    sub = sub[(sub["rel_pos"] >= ALIGN_WINDOW[0])
              & (sub["rel_pos"] <= ALIGN_WINDOW[1])]
    return sub.groupby("rel_pos")[value].agg(["mean", "sem"]).reset_index()


def figure_11_grand_average(df: pd.DataFrame, out_dir: Path) -> None:
    H = _aligned_means(df, "H")
    S = _aligned_means(df, "S")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)

    ax.plot(H["rel_pos"], H["mean"], color=COLOR_H, lw=2.4,
            marker="o", markersize=6, label="Entropy (H)")
    ax.fill_between(H["rel_pos"], H["mean"] - H["sem"], H["mean"] + H["sem"],
                    color=COLOR_H, alpha=0.18)

    ax2.plot(S["rel_pos"], S["mean"], color=COLOR_S, lw=2.4,
             marker="s", markersize=6, label="Surprisal (S)")
    ax2.fill_between(S["rel_pos"], S["mean"] - S["sem"], S["mean"] + S["sem"],
                     color=COLOR_S, alpha=0.18)

    h_peak = int(H.loc[H["mean"].idxmax(), "rel_pos"])
    s_peak = int(S.loc[S["mean"].idxmax(), "rel_pos"])
    ax.axvline(h_peak, color=COLOR_H, ls="--", lw=1.4, alpha=0.7)
    ax.axvline(s_peak, color=COLOR_S, ls="--", lw=1.4, alpha=0.7)
    ax.axvline(0, color="black", lw=0.8, ls=":")

    ax.set_xlabel("position relative to disambiguating word (0)")
    ax.set_ylabel("Entropy", color=COLOR_H)
    ax2.set_ylabel("Surprisal", color=COLOR_S)
    ax.set_title(
        f"Fig 11 · Grand-averaged temporal cascade   "
        f"(H peak at {h_peak:+d}, S peak at {s_peak:+d})"
    )
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
              frameon=False)
    save(fig, out_dir, "fig11_grand_average")


# ============================================================================
# FIGURE 12 — ΔH at disambiguation: GP vs control
# ============================================================================
def figure_12_delta_H(df: pd.DataFrame, out_dir: Path) -> None:
    gp = df[df["is_garden_path"] == 1]
    ctrl = df[df["is_garden_path"] == 0]
    gp_dH = (gp[(gp["rel_pos"] >= ALIGN_WINDOW[0])
                 & (gp["rel_pos"] <= ALIGN_WINDOW[1])]
             .groupby("rel_pos")["delta_H"].agg(["mean", "sem"]).reset_index())
    ctrl_dH = (ctrl[(ctrl["rel_pos"] >= ALIGN_WINDOW[0])
                     & (ctrl["rel_pos"] <= ALIGN_WINDOW[1])]
               .groupby("rel_pos")["delta_H"].agg(["mean", "sem"]).reset_index())

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(gp_dH["rel_pos"], gp_dH["mean"], color=COLOR_GP, lw=2.4,
            marker="o", markersize=6, label="Garden-path")
    ax.fill_between(gp_dH["rel_pos"], gp_dH["mean"] - gp_dH["sem"],
                    gp_dH["mean"] + gp_dH["sem"], color=COLOR_GP, alpha=0.18)
    ax.plot(ctrl_dH["rel_pos"], ctrl_dH["mean"], color=COLOR_CTRL, lw=2.4,
            marker="s", markersize=6, label="Control")
    ax.fill_between(ctrl_dH["rel_pos"], ctrl_dH["mean"] - ctrl_dH["sem"],
                    ctrl_dH["mean"] + ctrl_dH["sem"], color=COLOR_CTRL, alpha=0.18)
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.axhline(0, color="grey", lw=0.7)

    ax.set_xlabel("position relative to disambiguating word (0)")
    ax.set_ylabel("ΔH = H(t−1) − H(t)")
    ax.set_title("Fig 12 · Entropy reduction spikes at disambiguation in GP only")
    ax.legend(frameon=False)
    save(fig, out_dir, "fig12_delta_H")


# ============================================================================
# FIGURE 13 — Full cascade: H spike → RT elevation → S spike
# ============================================================================
def figure_13_full_cascade(df: pd.DataFrame, out_dir: Path) -> None:
    if "RT" not in df.columns:
        print("  [Fig 13] skipped — no 'RT' column in data")
        return
    H = _aligned_means(df, "H")
    S = _aligned_means(df, "S")
    RT = _aligned_means(df, "RT")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 7),
                                  sharex=True,
                                  gridspec_kw={"height_ratios": [1.3, 1]})
    a1b = a1.twinx()
    a1b.spines["top"].set_visible(False)
    a1.plot(H["rel_pos"], H["mean"], color=COLOR_H, lw=2.4,
            marker="o", markersize=6, label="H")
    a1.fill_between(H["rel_pos"], H["mean"] - H["sem"],
                    H["mean"] + H["sem"], color=COLOR_H, alpha=0.18)
    a1b.plot(S["rel_pos"], S["mean"], color=COLOR_S, lw=2.4,
             marker="s", markersize=6, label="S")
    a1b.fill_between(S["rel_pos"], S["mean"] - S["sem"],
                     S["mean"] + S["sem"], color=COLOR_S, alpha=0.18)
    a1.axvline(0, color="black", lw=0.8, ls=":")
    a1.set_ylabel("Entropy", color=COLOR_H)
    a1b.set_ylabel("Surprisal", color=COLOR_S)
    a1.set_title("Top: model features around disambiguation")

    a2.plot(RT["rel_pos"], RT["mean"], color="#5a189a", lw=2.4,
            marker="D", markersize=6, label="mean RT")
    a2.fill_between(RT["rel_pos"], RT["mean"] - RT["sem"],
                    RT["mean"] + RT["sem"], color="#5a189a", alpha=0.18)
    a2.axvline(0, color="black", lw=0.8, ls=":")
    a2.set_xlabel("position relative to disambiguating word (0)")
    a2.set_ylabel("mean RT (ms)")
    a2.set_title("Bottom: human reading time")
    a2.legend(frameon=False)

    fig.suptitle("Fig 13 · Full cascade  H spike (−1) → RT elevation (0) → S spike (+1)",
                 y=1.02, fontsize=11)
    save(fig, out_dir, "fig13_full_cascade")


# ============================================================================
# Statistics — Wilcoxon signed-rank and cross-lag correlations (§5.5)
# ============================================================================
def cascade_statistics(df: pd.DataFrame) -> dict:
    leads = _lead_times(df)
    gp = leads[leads["is_garden_path"] == 1]
    diffs = gp["h_peak"] - gp["s_peak"]   # negative = H precedes S

    nz = diffs[diffs != 0]
    if len(nz) > 0:
        w, p_w = stats.wilcoxon(nz, alternative="less")  # H peak < S peak
    else:
        w, p_w = float("nan"), float("nan")

    # Cross-lag correlations: H(t = lag) vs S(t = 0), per item
    out = {}
    gp_items = df[df["is_garden_path"] == 1]
    pivot_H = gp_items.pivot_table(index="item_id", columns="rel_pos",
                                    values="H", aggfunc="mean")
    pivot_S = gp_items.pivot_table(index="item_id", columns="rel_pos",
                                    values="S", aggfunc="mean")
    if 0 in pivot_S.columns:
        s0 = pivot_S[0]
        for lag in (-2, -1, 0):
            if lag in pivot_H.columns:
                joint = pd.concat([pivot_H[lag].rename("H"),
                                   s0.rename("S")], axis=1).dropna()
                r, p = stats.pearsonr(joint["H"], joint["S"])
                out[f"r_H(t={lag:+d})_S(0)"] = (r, p)

    return {"wilcoxon_W": w, "wilcoxon_p": p_w,
            "n_items": len(gp), "n_nonzero": len(nz),
            "cross_lag": out}


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--items", nargs="*", default=None,
                        help="Optional list of 4 item_ids for Fig 9")
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    df = load_table(args.data)
    print(f"  N rows = {len(df):,}; "
          f"items: {df.loc[df['is_garden_path']==1,'item_id'].nunique()} GP / "
          f"{df.loc[df['is_garden_path']==0,'item_id'].nunique()} controls")

    print("Generating figures →", args.out)
    figure_9_example_sentences(df, args.out, representative_items=args.items)
    res10 = figure_10_lead_times(df, args.out)
    print(f"   Lead time GP   mean δ = {res10['mean_delta_gp']:+.2f} words "
          f"(t = {res10['t']:.2f}, p = {res10['p']:.2e})")
    figure_11_grand_average(df, args.out)
    figure_12_delta_H(df, args.out)
    figure_13_full_cascade(df, args.out)

    print("\nCascade statistics (§5.5):")
    stats_d = cascade_statistics(df)
    print(f"   Wilcoxon W = {stats_d['wilcoxon_W']}, p = {stats_d['wilcoxon_p']:.2e}")
    for k, (r, p) in stats_d["cross_lag"].items():
        print(f"   {k}: r = {r:+.3f}, p = {p:.2e}")
    print("Done.")


if __name__ == "__main__":
    main()
