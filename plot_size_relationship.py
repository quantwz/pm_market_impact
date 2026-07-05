"""
market_impact/plot_size_relationship.py
────────────────────────────────────────
Polymarket Market Impact Analysis – Trade Size Relationship Plot

Generates Figure 8: A 1x3 panel plot showing the relationship between
absolute trade size (USD) and ADV-fraction size (S_ADV) across different
groupings:
1. Panel (a): By Market Category
2. Panel (b): By Prior Probability Level
3. Panel (c): By Time to Expiry

This directly visualizes the difference in baseline liquidity and typical ADV
across market segments (vertical shifts on the log-log scale indicate differences
in typical ADV).

Saved as PDF and PNG to market_impact/plots/fig8_size_relationship.pdf/.png.

Author: Market Impact Research Pipeline
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── SciencePlots style ───────────────────────────────────────────────────────
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except ImportError:
    pass

matplotlib.rcParams.update({
    "font.size":          9.5,
    "axes.labelsize":     10.5,
    "axes.titlesize":     10.5,
    "legend.fontsize":    8.5,
    "legend.title_fontsize": 9,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "axes.linewidth":     0.8,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.35,
    "lines.linewidth":    1.5,
    "lines.markersize":   3.8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Design tokens (consistent with plot_impact.py) ───────────────────────────
CAT_ORDER  = ["Macro", "Politics", "Geopolitics"]
CAT_COLORS = {"Macro": "#1b6ca8", "Politics": "#c44c3a", "Geopolitics": "#2a9d8f"}
CAT_LS     = {"Macro": "-",       "Politics": "--",       "Geopolitics": "-."}
CAT_MK     = {"Macro": "o",       "Politics": "s",        "Geopolitics": "^"}

PROB_KEYS    = ["p_lo", "p_midlo", "p_mid", "p_midhi", "p_hi"]
PROB_COLORS  = ["#023e8a", "#457b9d", "#a8dadc", "#e76f51", "#9b2226"]
PROB_LS      = ["-", "--", "-.", ":", (0, (3,1,1,1))]
PROB_MK      = ["o", "s", "^", "D", "v"]
PROB_DISPLAY = {
    "p_lo":    r"$p_{\,k-1} \in [0.0, 0.2)$",
    "p_midlo": r"$p_{\,k-1} \in [0.2, 0.4)$",
    "p_mid":   r"$p_{\,k-1} \in [0.4, 0.6)$",
    "p_midhi": r"$p_{\,k-1} \in [0.6, 0.8)$",
    "p_hi":    r"$p_{\,k-1} \in [0.8, 1.0]$",
}

EXP_KEYS    = ["tte_le7d", "tte_7_30d", "tte_30_90d", "tte_gt90d"]
EXP_COLORS  = ["#7f0000", "#e84a5f", "#f4a261", "#2ec4b6"]
EXP_LS      = ["-", "--", "-.", ":"]
EXP_MK      = ["o", "s", "^", "D"]
EXP_DISPLAY = {
    "tte_le7d":    r"$\tau \leq 7$ d",
    "tte_7_30d":   r"$7 < \tau \leq 30$ d",
    "tte_30_90d":  r"$30 < \tau \leq 90$ d",
    "tte_gt90d":   r"$\tau > 90$ d",
}


# ─── Formatters ───────────────────────────────────────────────────────────────
def _usd_formatter(x, _):
    if x >= 1e6:   return rf"$\${x/1e6:.0f}$M"
    if x >= 1e3:   return rf"$\${x/1e3:.0f}$K"
    if x >= 1:     return rf"$\${x:.0f}$"
    return rf"$\${x:.2g}$"


def _adv_formatter(x, _):
    if x <= 0:
        return "0"
    exp = int(np.floor(np.log10(x)))
    coef = x / 10 ** exp
    if abs(coef - 1.0) < 0.15:
        return f"$10^{{{exp}}}$" if exp != 0 else "$1$"
    return f"${coef:.1f}\\!\\times\\!10^{{{exp}}}$"


# ─── Binned Computation ───────────────────────────────────────────────────────
def compute_binned_sizes(df: pd.DataFrame, group_col: str, n_bins: int = 15) -> pd.DataFrame:
    """Bins trades by usd_amount and computes mean usd_amount and mean adv_ratio per bin."""
    rows = []
    groups = df[group_col].dropna().unique()
    
    for g in groups:
        sub = df[(df[group_col] == g) & (df["usd_amount"] > 0) & (df["adv_ratio"] > 0)].copy()
        if len(sub) < n_bins * 5:
            continue
            
        # Quantile binning on USD amount
        edges = np.nanpercentile(sub["usd_amount"], np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            continue
            
        sub["_bin"] = pd.cut(sub["usd_amount"], bins=edges, include_lowest=True)
        
        # Calculate mean USD size and mean ADV ratio per bin
        # We use geometric mean to represent the log-scale centers accurately
        agg = sub.groupby("_bin", observed=True).agg(
            usd_mean=("usd_amount", lambda x: np.exp(np.mean(np.log(x)))),
            adv_mean=("adv_ratio", lambda x: np.exp(np.mean(np.log(x)))),
            count=("usd_amount", "count")
        ).reset_index()
        
        agg["group"] = str(g)
        rows.append(agg.drop(columns=["_bin"]))
        
    return pd.concat(rows, ignore_index=True)


# ─── Main Plotting ────────────────────────────────────────────────────────────
def generate_relationship_plot() -> None:
    print("Loading trade-level data...")
    trade_path = os.path.join(DATA_DIR, "impact_trade_level.parquet")
    if not os.path.exists(trade_path):
        raise FileNotFoundError(f"Missing: {trade_path} — run compute_impact.py first.")

    df = pd.read_parquet(trade_path)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)

    # --- Panel (a): By Category ---
    print("Computing category sizes...")
    cat_df = compute_binned_sizes(df, "category")
    for i, key in enumerate(CAT_ORDER):
        grp = cat_df[cat_df["group"] == key].sort_values("usd_mean")
        ax1.plot(
            grp["usd_mean"], grp["adv_mean"],
            color=CAT_COLORS[key], linestyle=CAT_LS[key], marker=CAT_MK[key],
            label=key
        )
    ax1.set_title("By Market Category")
    ax1.legend(title="Category", loc="upper left")

    # --- Panel (b): By Prior Probability ---
    print("Computing probability sizes...")
    prob_df = compute_binned_sizes(df, "prob_bin")
    for i, key in enumerate(PROB_KEYS):
        grp = prob_df[prob_df["group"] == key].sort_values("usd_mean")
        ax2.plot(
            grp["usd_mean"], grp["adv_mean"],
            color=PROB_COLORS[i], linestyle=PROB_LS[i], marker=PROB_MK[i],
            label=PROB_DISPLAY[key]
        )
    ax2.set_title("By Prior Probability")
    ax2.legend(title=r"Prior Probability $p_{k-1}$", loc="upper left")

    # --- Panel (c): By Time to Expiry ---
    print("Computing expiry sizes...")
    exp_df = compute_binned_sizes(df, "expiry_bin")
    for i, key in enumerate(EXP_KEYS):
        grp = exp_df[exp_df["group"] == key].sort_values("usd_mean")
        ax3.plot(
            grp["usd_mean"], grp["adv_mean"],
            color=EXP_COLORS[i], linestyle=EXP_LS[i], marker=EXP_MK[i],
            label=EXP_DISPLAY[key]
        )
    ax3.set_title("By Time to Expiry")
    ax3.legend(title=r"Time to Expiry $\tau$", loc="upper left")

    # Format all subplots
    for idx, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_adv_formatter))
        ax.set_xlabel("Trade Size $S$ (USD)", labelpad=4)
        if idx == 0:
            ax.set_ylabel(r"Normalized Trade Size $S_{\mathrm{ADV}}$ (fraction of ADV)", labelpad=4)
        ax.grid(True, which="both", axis="both", alpha=0.2)
        
        # Panel labels
        ax.text(0.025, 0.95, f"({chr(97+idx)})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    fig.suptitle("Trade Size in USD vs. Normalized Trade Size (Fraction of ADV)", fontsize=11.5, y=1.03)
    fig.tight_layout()

    # Save
    for ext in ("pdf", "png"):
        path = os.path.join(PLOTS_DIR, f"fig8_size_relationship.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close(fig)


if __name__ == "__main__":
    generate_relationship_plot()
