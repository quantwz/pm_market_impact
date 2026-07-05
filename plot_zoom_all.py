"""
market_impact/plot_zoom_all.py
──────────────────────────────
Polymarket Market Impact Analysis – Zoom-in Microstructure Plots (All Cases)

Generates four separate figures:
1. Figure 4: Macro full sample
2. Figure 5: Extreme low prior probability level ([0, 0.2)) across all categories
3. Figure 6: Extreme high prior probability level ([0.8, 1.0]) across all categories
4. Figure 7: Close to expiry (<7 days) across all categories

Each figure consists of a 1x2 panel (a: raw price impact dp, b: logit impact dx)
with:
- Background scatter of individual trades (downsampled to 30,000 points)
- Log-spaced boxplots (showing median, IQR, and 10-90% whiskers)
- Binned mean line with standard error bands

All figures saved to market_impact/plots/ as PDF and PNG.

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
import matplotlib.patches as mpatches

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


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _usd_formatter(x, _):
    if x >= 1e6:   return rf"$\${x/1e6:.0f}$M"
    if x >= 1e3:   return rf"$\${x/1e3:.0f}$K"
    if x >= 1:     return rf"$\${x:.0f}$"
    return rf"$\${x:.2g}$"


def geom_mid(l, r) -> float:
    if l <= 0:
        l = 0.01
    return np.sqrt(l * r)


# ─── Plotting Function ────────────────────────────────────────────────────────
def plot_case(df: pd.DataFrame, case_info: dict) -> None:
    print(f"Generating plot for case: {case_info['name']} ...")
    
    # Apply filter
    sub = case_info["filter"](df).dropna(subset=["usd_amount", "dp", "dx"])
    n_trades = len(sub)
    print(f"  Trades: {n_trades:,}")
    if n_trades == 0:
        print(f"  Warning: No trades found for {case_info['name']}.")
        return

    # Log-spaced bin edges for trade size
    bin_edges = [0.01, 1.0, 10.0, 100.0, 1000.0, 10000.0, float(sub["usd_amount"].max())]
    bin_centers = [geom_mid(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]

    # Assign bin groups
    sub = sub.copy()
    sub["size_bin"] = pd.cut(sub["usd_amount"], bins=bin_edges, include_lowest=True, labels=False)

    # Downsample for scatter background
    scatter_sample = sub.sample(n=min(30000, len(sub)), random_state=42)

    # Calculate dynamic y-limits using the 1st and 99th percentiles (plus a 10% padding)
    q01_dp, q99_dp = np.percentile(sub["dp"], [1, 99])
    q01_dx, q99_dx = np.percentile(sub["dx"], [1, 99])
    
    # Ensure some minimal range to avoid divide by zero or squeezed plots
    dp_range = max(q99_dp - q01_dp, 0.005)
    dx_range = max(q99_dx - q01_dx, 0.1)
    
    dp_ylim = (q01_dp - dp_range * 0.15, q99_dp + dp_range * 0.15)
    dx_ylim = (q01_dx - dx_range * 0.15, q99_dx + dx_range * 0.15)

    # Initialize Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2))

    # Styling properties for boxplots
    box_props = dict(facecolor="#264653", color="#264653", alpha=0.8, linewidth=1.0)
    whisker_props = dict(color="#264653", linewidth=0.8, linestyle="--")
    capprops = dict(color="#264653", linewidth=0.8)
    median_props = dict(color="#e9c46a", linewidth=1.5)

    for ax, col, ylim, label in [(ax1, "dp", dp_ylim, r"Signed Price Impact $\Delta p$"),
                                 (ax2, "dx", dx_ylim, r"Signed Logit Impact $\Delta x$")]:
        # 1. Background Scatter Plot
        ax.scatter(
            scatter_sample["usd_amount"],
            scatter_sample[col],
            color="#2a9d8f",
            alpha=0.06,
            s=1.5,
            edgecolors="none",
            label="Individual Trades (Sample)",
            zorder=1
        )

        # Prepare boxplots per bin
        box_data = []
        box_positions = []
        box_widths = []
        mean_x = []
        mean_y = []
        se_y = []

        for i in range(len(bin_edges)-1):
            bin_trades = sub[sub["size_bin"] == i][col].dropna()
            if len(bin_trades) < 10:
                continue
            
            # Clip display values to the boxplot range to avoid extreme outliers messing up box scaling
            clipped = np.clip(bin_trades.values, ylim[0]*2, ylim[1]*2)
            box_data.append(clipped)
            
            pos = bin_centers[i]
            box_positions.append(pos)
            
            # Width in log scale (proportional to box location)
            width = pos * 0.45
            box_widths.append(width)

            # Calculate mean and standard error
            mean_x.append(pos)
            mean_y.append(bin_trades.mean())
            se_y.append(bin_trades.std() / np.sqrt(len(bin_trades)))

        # 2. Draw Boxplots
        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=box_widths,
            patch_artist=True,
            showfliers=False,
            whis=[10, 90],  # Whiskers represent 10th and 90th percentiles
            manage_ticks=False,
            boxprops=box_props,
            whiskerprops=whisker_props,
            capprops=capprops,
            medianprops=median_props,
            zorder=3
        )

        # 3. Draw Mean Line and standard error bands
        mean_x = np.array(mean_x)
        mean_y = np.array(mean_y)
        se_y = np.array(se_y)
        
        ax.plot(
            mean_x,
            mean_y,
            color="#e76f51",
            linestyle="-",
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="Mean Price Impact",
            zorder=4
        )
        ax.fill_between(
            mean_x,
            mean_y - 2*se_y,
            mean_y + 2*se_y,
            color="#e76f51",
            alpha=0.15,
            linewidth=0,
            zorder=2
        )

        # Zero reference line
        ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5, zorder=2)

        # Scale and Axis formatting
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_usd_formatter))
        ax.set_xlabel("Trade Size $S$ (USD)", labelpad=4)
        ax.set_ylabel(label, labelpad=4)
        ax.set_ylim(ylim)
        ax.set_xlim(0.01, 100000.0)
        ax.grid(True, which="both", axis="both", alpha=0.2)

    # Legends & Titles
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#2a9d8f", alpha=0.4, markersize=5, label="Individual Trades"),
        mpatches.Patch(facecolor="#264653", edgecolor="#264653", alpha=0.8, label="Distribution (IQR, 10-90% whisker)"),
        plt.Line2D([0], [0], color="#e76f51", marker="o", markersize=4, label="Mean Impact (w/ $\pm 2$ SE band)")
    ]
    ax1.legend(handles=legend_handles, loc="upper left", frameon=True, framealpha=0.9)
    
    # Subplot labels
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes, fontsize=10, fontweight="bold", va="top")
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes, fontsize=10, fontweight="bold", va="top")

    fig.suptitle(case_info["title"], fontsize=11.0, y=1.01)
    fig.tight_layout()

    # Save outputs
    for ext in ("pdf", "png"):
        path = os.path.join(PLOTS_DIR, f"{case_info['filename']}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Market Impact Analysis: Zoom-In Microstructure Plots ===")
    trade_path = os.path.join(DATA_DIR, "impact_trade_level.parquet")
    if not os.path.exists(trade_path):
        raise FileNotFoundError(f"Missing: {trade_path} — run compute_impact.py first.")

    print("Loading trade-level dataset...")
    df = pd.read_parquet(trade_path)

    # Definition of the 4 cases requested by the user
    cases = [
        {
            "name": "macro",
            "title": "Zoom-In Microstructure Analysis: Macro Markets (All Trades)",
            "filter": lambda x: x[x["category"] == "Macro"],
            "filename": "fig4_zoom_macro"
        },
        {
            "name": "p_lo",
            "title": r"Zoom-In Microstructure Analysis: Extreme Low Prior Probability ($p_{k-1} < 0.2$, All Categories)",
            "filter": lambda x: x[x["prob_bin"] == "p_lo"],
            "filename": "fig5_zoom_p_lo"
        },
        {
            "name": "p_hi",
            "title": r"Zoom-In Microstructure Analysis: Extreme High Prior Probability ($p_{k-1} \geq 0.8$, All Categories)",
            "filter": lambda x: x[x["prob_bin"] == "p_hi"],
            "filename": "fig6_zoom_p_hi"
        },
        {
            "name": "tte_le7d",
            "title": r"Zoom-In Microstructure Analysis: Close to Expiry ($\tau \leq 7$ Days, All Categories)",
            "filter": lambda x: x[x["expiry_bin"] == "tte_le7d"],
            "filename": "fig7_zoom_tte_le7d"
        }
    ]

    for c in cases:
        plot_case(df, c)

    print("=== All Zoom-In Microstructure Plots Generated ===")


if __name__ == "__main__":
    main()
