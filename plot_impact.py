"""
market_impact/plot_impact.py
────────────────────────────
Polymarket Market Impact Analysis – Publication-Quality Figures

Generates three figures using SciencePlots styling and LaTeX-compatible
math labels, intended for presentation at the Oxford Man Institute.

Figure 1 – Market price impact vs trade size, by market category
Figure 2 – Conditional price impact vs trade size, by prior probability level
Figure 3 – Conditional price impact vs trade size, by time to expiry

All figures saved as PDF (vector) and PNG (raster) to market_impact/plots/.

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
import matplotlib.patheffects as pe

# ─── SciencePlots style ───────────────────────────────────────────────────────
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except ImportError:
    pass   # graceful fallback to default style

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


# ─── Design tokens ────────────────────────────────────────────────────────────
# Category palette
CAT_ORDER  = ["Macro", "Politics", "Geopolitics"]
CAT_COLORS = {"Macro": "#1b6ca8", "Politics": "#c44c3a", "Geopolitics": "#2a9d8f"}
CAT_LS     = {"Macro": "-",       "Politics": "--",       "Geopolitics": "-."}
CAT_MK     = {"Macro": "o",       "Politics": "s",        "Geopolitics": "^"}
CAT_LABEL  = {"Macro": "Macro",   "Politics": "Politics", "Geopolitics": "Geopolitics"}

# Probability level palette (5 bins: [0,.2) … [.8,1])
PROB_KEYS    = ["p_lo", "p_midlo", "p_mid", "p_midhi", "p_hi"]
PROB_COLORS  = ["#023e8a", "#457b9d", "#a8dadc", "#e76f51", "#9b2226"]
PROB_LS      = ["-", "--", "-.", ":", (0, (3,1,1,1))]
PROB_MK      = ["o", "s", "^", "D", "v"]
PROB_DISPLAY = {
    "p_lo":    r"$p_{\,k-1} \in [0.0,\;0.2)$",
    "p_midlo": r"$p_{\,k-1} \in [0.2,\;0.4)$",
    "p_mid":   r"$p_{\,k-1} \in [0.4,\;0.6)$",
    "p_midhi": r"$p_{\,k-1} \in [0.6,\;0.8)$",
    "p_hi":    r"$p_{\,k-1} \in [0.8,\;1.0]$",
}

# Time-to-expiry palette (4 bins)
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

# Axis labels (mathtext)
SIZE_XLABELS = {
    "usd": r"Trade Size $S$ (USD)",
    "adv": r"Trade Size $S_{\mathrm{ADV}}$ (fraction of ADV)",
}
IMPACT_YLABELS = {
    "dp": r"Signed Price Impact $\Delta p$",
    "dx": r"Signed Logit Impact $\Delta x$",
}

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _load(cond: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{cond}_impacts.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}  — run compute_impact.py first.")
    return pd.read_parquet(path)


def _usd_formatter(x, _):
    """Human-readable USD tick labels."""
    if x >= 1e6:   return rf"$\${x/1e6:.0f}$M"
    if x >= 1e3:   return rf"$\${x/1e3:.0f}$K"
    if x >= 1:     return rf"$\${x:.0f}$"
    return rf"$\${x:.2g}$"


def _adv_formatter(x, _):
    """Log-scale fraction-of-ADV tick labels."""
    if x <= 0:
        return "0"
    exp = int(np.floor(np.log10(x)))
    coef = x / 10 ** exp
    if abs(coef - 1.0) < 0.15:
        return f"$10^{{{exp}}}$" if exp != 0 else "$1$"
    return f"${coef:.1f}\\!\\times\\!10^{{{exp}}}$"


SIZE_FORMATTERS = {
    "usd": mticker.FuncFormatter(_usd_formatter),
    "adv": mticker.FuncFormatter(_adv_formatter),
}


def _draw_panel(
    ax: plt.Axes,
    tbl: pd.DataFrame,
    group_keys: list[str],
    colors:     list[str],
    linestyles: list[str],
    markers:    list[str],
    display:    dict[str, str],
    size_type:  str,
    impact_type: str,
    show_band:  bool = True,
    alpha_band: float = 0.10,
) -> list:
    """Draw one 2D impact-vs-size panel. Returns legend handles."""
    handles = []
    subset = tbl[
        (tbl["size_type"]   == size_type) &
        (tbl["impact_type"] == impact_type)
    ]

    for i, key in enumerate(group_keys):
        grp = subset[subset["group"] == key].dropna(
            subset=["size_bin_center", "mean", "se"]
        )
        if grp.empty:
            continue
        grp = grp.sort_values("size_bin_center")
        x    = grp["size_bin_center"].values
        y    = grp["mean"].values
        # 95% confidence band for the mean (very tight for large n, shows precision)
        y_lo = (grp["mean"] - 2.0 * grp["se"]).values
        y_hi = (grp["mean"] + 2.0 * grp["se"]).values

        c  = colors[i]
        ls = linestyles[i]
        mk = markers[i]
        lb = display.get(key, key)

        line, = ax.plot(x, y, color=c, ls=ls, marker=mk,
                        label=lb, zorder=4, clip_on=True)
        if show_band:
            ax.fill_between(x, y_lo, y_hi, color=c, alpha=alpha_band,
                            zorder=2, linewidth=0)
        handles.append(line)

    # Zero reference
    ax.axhline(0, color="black", lw=0.7, ls="--", zorder=3, alpha=0.6)

    # Axes formatting
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(SIZE_FORMATTERS[size_type])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(SIZE_XLABELS[size_type], labelpad=4)
    ax.set_ylabel(IMPACT_YLABELS[impact_type], labelpad=4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
    ax.grid(True, which="both", axis="both")
    ax.grid(True, which="minor", alpha=0.15)

    return handles


def _stamp_panels(axes: np.ndarray) -> None:
    for ax, lbl in zip(axes.flat, PANEL_LABELS):
        ax.text(0.025, 0.975, lbl, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", ha="left",
                color="black", zorder=10)


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(PLOTS_DIR, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  Saved → {path}")
    plt.close(fig)


# ─── Figure 1: by category ───────────────────────────────────────────────────
def fig1_by_category() -> None:
    """2×2 grid: rows = impact type (Δp, Δx), cols = size type (USD, ADV ratio).
       Lines coloured by market category."""
    print("Plotting Figure 1 — Price impact by category …")
    tbl = _load("cat")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    combos = [
        ("usd", "dp", axes[0, 0]),
        ("adv", "dp", axes[0, 1]),
        ("usd", "dx", axes[1, 0]),
        ("adv", "dx", axes[1, 1]),
    ]
    all_handles = None
    for sc, ic, ax in combos:
        h = _draw_panel(
            ax, tbl,
            group_keys  = CAT_ORDER,
            colors      = [CAT_COLORS[k] for k in CAT_ORDER],
            linestyles  = [CAT_LS[k]     for k in CAT_ORDER],
            markers     = [CAT_MK[k]     for k in CAT_ORDER],
            display     = {k: k for k in CAT_ORDER},
            size_type   = sc,
            impact_type = ic,
        )
        if all_handles is None and h:
            all_handles = h

    _stamp_panels(axes)

    # Shared legend above the grid
    if all_handles:
        labels = CAT_ORDER
        fig.legend(
            all_handles, labels,
            loc="upper center", ncol=3,
            bbox_to_anchor=(0.5, 1.03),
            frameon=True, framealpha=0.95,
            title="Market Category",
        )

    fig.suptitle(
        "Market Price Impact vs. Trade Size by Category",
        fontsize=11.5, y=1.07,
    )
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    _save(fig, "fig1_impact_by_category")


# ─── Figure 2: conditional on prior probability level ────────────────────────
def fig2_by_prob_level() -> None:
    """2×2 grid: rows = impact type, cols = size type.
       Lines coloured by prior YES probability bin."""
    print("Plotting Figure 2 — Conditional on probability level …")
    tbl = _load("prob")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    combos = [
        ("usd", "dp", axes[0, 0]),
        ("adv", "dp", axes[0, 1]),
        ("usd", "dx", axes[1, 0]),
        ("adv", "dx", axes[1, 1]),
    ]
    all_handles, all_labels = None, None
    for sc, ic, ax in combos:
        h = _draw_panel(
            ax, tbl,
            group_keys  = PROB_KEYS,
            colors      = PROB_COLORS,
            linestyles  = PROB_LS,
            markers     = PROB_MK,
            display     = PROB_DISPLAY,
            size_type   = sc,
            impact_type = ic,
            alpha_band  = 0.08,
        )
        if all_handles is None and h:
            all_handles = h
            all_labels  = [PROB_DISPLAY.get(k, k) for k in PROB_KEYS
                           if any(tbl["group"] == k)]

    _stamp_panels(axes)

    if all_handles and all_labels:
        fig.legend(
            all_handles, all_labels,
            loc="upper center", ncol=3,
            bbox_to_anchor=(0.5, 1.03),
            frameon=True, framealpha=0.95,
            title=r"Prior Probability $p_{k-1}$",
        )

    fig.suptitle(
        r"Market Impact Conditional on Prior Probability Level $p_{k-1}$",
        fontsize=11.5, y=1.07,
    )
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    _save(fig, "fig2_impact_by_prob_level")


# ─── Figure 3: conditional on time to expiry ─────────────────────────────────
def fig3_by_expiry() -> None:
    """2×2 grid: rows = impact type, cols = size type.
       Lines coloured by time to expiry."""
    print("Plotting Figure 3 — Conditional on time to expiry …")
    tbl = _load("expiry")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    combos = [
        ("usd", "dp", axes[0, 0]),
        ("adv", "dp", axes[0, 1]),
        ("usd", "dx", axes[1, 0]),
        ("adv", "dx", axes[1, 1]),
    ]
    all_handles, all_labels = None, None
    for sc, ic, ax in combos:
        h = _draw_panel(
            ax, tbl,
            group_keys  = EXP_KEYS,
            colors      = EXP_COLORS,
            linestyles  = EXP_LS,
            markers     = EXP_MK,
            display     = EXP_DISPLAY,
            size_type   = sc,
            impact_type = ic,
            alpha_band  = 0.08,
        )
        if all_handles is None and h:
            all_handles = h
            all_labels  = [EXP_DISPLAY.get(k, k) for k in EXP_KEYS
                           if any(tbl["group"] == k)]

    _stamp_panels(axes)

    if all_handles and all_labels:
        fig.legend(
            all_handles, all_labels,
            loc="upper center", ncol=4,
            bbox_to_anchor=(0.5, 1.03),
            frameon=True, framealpha=0.95,
            title=r"Time to Expiry $\tau$",
        )

    fig.suptitle(
        r"Market Impact Conditional on Time to Expiry $\tau$",
        fontsize=11.5, y=1.07,
    )
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    _save(fig, "fig3_impact_by_expiry")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Market Impact Analysis: Plotting ===")
    fig1_by_category()
    fig2_by_prob_level()
    fig3_by_expiry()
    print(f"=== All figures saved to: {PLOTS_DIR} ===")


if __name__ == "__main__":
    main()
