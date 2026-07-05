"""
plot_impact_filtered.py
────────────────────────
Polymarket Market Impact Analysis – Filtered Publication-Quality Figures (Trade Size >= $100)

Loads trade-level data, filters for trade size >= $100, aggregates,
and generates three figures using SciencePlots styling and LaTeX-compatible labels.

All figures saved as PDF (vector) and PNG (raster) to plots/.
"""

import os
import gc
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Constants & Styling Design Tokens ────────────────────────────────────────
CAT_ORDER  = ["Macro", "Politics", "Geopolitics"]
CAT_COLORS = {"Macro": "#1b6ca8", "Politics": "#c44c3a", "Geopolitics": "#2a9d8f"}
CAT_LS     = {"Macro": "-",       "Politics": "--",       "Geopolitics": "-."}
CAT_MK     = {"Macro": "o",       "Politics": "s",        "Geopolitics": "^"}

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

SIZE_XLABELS = {
    "usd": r"Trade Size $S$ (USD)",
    "adv": r"Trade Size $S_{\mathrm{ADV}}$ (fraction of ADV)",
}
IMPACT_YLABELS = {
    "dp": r"Signed Price Impact $\Delta p$",
    "dx": r"Signed Logit Impact $\Delta x$",
}

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

N_BINS_MAIN = 20
N_BINS_COND = 14
MIN_OBS_PER_BIN = 5

# ─── Formatting Helpers ───────────────────────────────────────────────────────
def geom_mid(interval) -> float:
    l, r = interval.left, interval.right
    if l <= 0:
        l = r * 0.01
    return np.sqrt(l * r)

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

SIZE_FORMATTERS = {
    "usd": mticker.FuncFormatter(_usd_formatter),
    "adv": mticker.FuncFormatter(_adv_formatter),
}

# ─── Aggregation ──────────────────────────────────────────────────────────────
def _quantile_aggregate(
    df: pd.DataFrame,
    size_col: str,
    impact_col: str,
    group_col: str,
    n_bins: int,
) -> pd.DataFrame:
    sub_all = df[df[size_col] > 0].copy()
    
    # Winsorise impact globally at 1st/99th percentile
    lo, hi = np.nanpercentile(sub_all[impact_col].dropna(), [1, 99])
    sub_all[impact_col] = sub_all[impact_col].clip(lo, hi)
    
    rows = []
    groups = sorted(sub_all[group_col].dropna().unique().tolist(), key=str)
    
    for g in groups:
        sub = sub_all[sub_all[group_col] == g].dropna(subset=[size_col, impact_col])
        if len(sub) < n_bins * MIN_OBS_PER_BIN:
            log.warning(f"   Skipping {group_col}={g}: only {len(sub)} obs")
            continue
            
        edges = np.nanpercentile(sub[size_col], np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            continue
            
        sub = sub.copy()
        sub["_bin"] = pd.cut(sub[size_col], bins=edges, include_lowest=True)
        
        agg = (sub.groupby("_bin", observed=True)[impact_col]
               .agg(
                   n="count",
                   q10=lambda x: np.percentile(x, 10),
                   q25=lambda x: np.percentile(x, 25),
                   q50="median",
                   q75=lambda x: np.percentile(x, 75),
                   q90=lambda x: np.percentile(x, 90),
                   mean="mean",
                   se=lambda x: x.std(ddof=1) / np.sqrt(len(x)),
               )
               .reset_index())
               
        agg["size_bin_center"] = agg["_bin"].apply(geom_mid)
        agg["group"]           = str(g)
        rows.append(agg.drop(columns=["_bin"]))
        
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def aggregate_all(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    size_map   = [("usd_amount", "usd"), ("adv_ratio", "adv")]
    impact_map = [("dp", "dp"),          ("dx", "dx")]
    cond_map   = [
        ("category",   "cat",    N_BINS_MAIN),
        ("prob_bin",   "prob",   N_BINS_COND),
        ("expiry_bin", "expiry", N_BINS_COND),
    ]
    
    results: dict[str, list[pd.DataFrame]] = {c: [] for _, c, _ in cond_map}
    
    for (gcol, gkey, n_bins) in cond_map:
        for (sc, sl) in size_map:
            for (ic, il) in impact_map:
                log.info(f"   Aggregating cond={gkey} size={sl} impact={il} ...")
                tbl = _quantile_aggregate(df, sc, ic, gcol, n_bins)
                if not tbl.empty:
                    tbl["size_type"]   = sl
                    tbl["impact_type"] = il
                    results[gkey].append(tbl)
                    
    return {k: pd.concat(v, ignore_index=True) for k, v in results.items() if v}

# ─── Drawing Helper ───────────────────────────────────────────────────────────
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
        
    ax.axhline(0, color="black", lw=0.7, ls="--", zorder=3, alpha=0.6)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(SIZE_FORMATTERS[size_type])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(SIZE_XLABELS[size_type], labelpad=4)
    if size_type == "usd":
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
        log.info(f"  Saved -> {path}")
    plt.close(fig)

# ─── Plot Figures ─────────────────────────────────────────────────────────────
def plot_fig1(tbl: pd.DataFrame) -> None:
    log.info("Plotting Figure 1 — Price impact by category (Filtered) ...")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey="row")
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
        r"Mean Market Price Impact vs. Trade Size by Category (Trade Size $\geq$ \$100)",
        fontsize=11.5, y=1.07,
    )
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    _save(fig, "fig1_impact_by_category_filtered")

def plot_fig2(tbl: pd.DataFrame) -> None:
    log.info("Plotting Figure 2 — Conditional on probability level (Filtered) ...")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey="row")
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
            all_labels  = [PROB_DISPLAY.get(k, k) for k in PROB_KEYS if any(tbl["group"] == k)]
            
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
        r"Mean Market Impact Conditional on Prior Probability Level $p_{k-1}$ (Trade Size $\geq$ \$100)",
        fontsize=11.5, y=1.07,
    )
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    _save(fig, "fig2_impact_by_prob_level_filtered")

def plot_fig3(tbl: pd.DataFrame) -> None:
    log.info("Plotting Figure 3 — Conditional on time to expiry (Filtered) ...")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey="row")
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
            all_labels  = [EXP_DISPLAY.get(k, k) for k in EXP_KEYS if any(tbl["group"] == k)]
            
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
        r"Mean Market Impact Conditional on Time to Expiry $\tau$ (Trade Size $\geq$ \$100)",
        fontsize=11.5, y=1.07,
    )
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    _save(fig, "fig3_impact_by_expiry_filtered")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("═══ Filtered Market Impact Pipeline (Trade Size >= $100) ═══")
    
    trade_level_path = os.path.join(DATA_DIR, "impact_trade_level.parquet")
    if not os.path.exists(trade_level_path):
        raise FileNotFoundError(f"Missing trade-level dataset: {trade_level_path}")
        
    log.info("━━ Step 1 ─ Loading trade-level dataset ━━")
    cols = ["category", "prob_bin", "expiry_bin", "usd_amount", "adv_ratio", "dp", "dx"]
    df = pd.read_parquet(trade_level_path, columns=cols)
    log.info(f"   Loaded {len(df):,} rows")
    
    log.info("━━ Step 2 ─ Filtering trade size >= $100 ━━")
    df_filtered = df[df["usd_amount"] >= 100].copy()
    log.info(f"   Filtered dataset size: {len(df_filtered):,} rows ({len(df_filtered)/len(df)*100:.2f}% of original)")
    
    # Clear memory of original df
    del df; gc.collect()
    
    log.info("━━ Step 3 ─ Aggregating (all conditions × sizes × impacts) ━━")
    tables = aggregate_all(df_filtered)
    
    log.info("━━ Step 4 ─ Writing filtered aggregation outputs ━━")
    for name, tbl in tables.items():
        path = os.path.join(DATA_DIR, f"{name}_impacts_filtered.parquet")
        tbl.to_parquet(path, index=False)
        log.info(f"   {name}_impacts_filtered.parquet  ->  {len(tbl):,} rows")
        
    log.info("━━ Step 5 ─ Generating figures ━━")
    plot_fig1(tables["cat"])
    plot_fig2(tables["prob"])
    plot_fig3(tables["expiry"])
    
    log.info("═══ Filtered Pipeline complete ═══")

if __name__ == "__main__":
    main()
