"""
build_july_fomc_replay_notebook.py
===================================
Constructs the Jupyter Notebook `fomc_july_maker_taker_orderbook_replay.ipynb`
with full July 1–29 date coverage and SciencePlots styling:
plt.style.use(['science', 'no-latex'])
"""

import os
import sys
import nbformat as nbf

NOTEBOOK_PATH = "fomc_july_maker_taker_orderbook_replay.ipynb"

def build_notebook():
    print(f"Constructing Jupyter Notebook across full July 1–29 window: {NOTEBOOK_PATH}...")
    nb = nbf.v4.new_notebook()
    nb.cells = []
    
    # Title & Markdown
    nb.cells.append(nbf.v4.new_markdown_cell("""# High-Frequency PMXT Orderbook & Maker/Taker Replay: FOMC July 2026

## Mission Overview
This notebook presents a synchronized, high-frequency reconstruction and replay of the **FOMC July 2026** contract (`fed-decision-in-july-181`) across its top 3 outcomes:
1. **No Change**
2. **Increase 25 bps**
3. **Decrease 25 bps**

### Pure-Source Data Architecture
All orderbook states ($L2$ snapshots and deltas) and trade execution ticks (`last_trade_price`) were reconstructed strictly from raw **PMXT** archives for the full **July 1 to July 29, 2026** trading window (**86,682 microsecond events**).

### Key Features
- **Full July 1–29 Timeline**: Full month coverage plus high-resolution microsecond focus on key macro shock windows.
- **Maker vs Taker Jump Driver Tagging**: Disentangling price jumps initiated by aggressive **Takers** (lifting ask / hitting bid) vs **Market Maker** quote cancellations / re-quoting.
- **Cross-Outcome Latency ($\tau_{MM}$)**: Measuring Market Maker reaction latency and Taker latency arbitrage volume ($V_{\\text{arb}}$) during cross-outcome news shocks.
- **Interactive Animated Replay Dashboard**: Equipped with time slider, progress bar, play/pause controls, zoom buttons, and driver callouts.
"""))

    # Cell 1: Setup and SciencePlots Import
    nb.cells.append(nbf.v4.new_code_cell("""# Environment & SciencePlots Setup
import os
import sys
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Apply SciencePlots styling as required by project standards
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
    print("Loaded SciencePlots style successfully.")
except Exception as e:
    print("SciencePlots fallback:", e)
    plt.style.use('default')

os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)
"""))

    # Cell 2: Data Loading & Verification
    nb.cells.append(nbf.v4.new_code_cell("""# Load Full July 1–29 Replay Dataset from PMXT DuckDB Database
from fomc_july_replay_engine import load_replay_data, classify_event_drivers, analyze_cross_outcome_shocks

print("Loading synchronized PMXT L2 orderbook states and trade executions...")
df_l2, df_trades = load_replay_data()

print(f"Total L2 Orderbook Ticks: {len(df_l2):,}")
print(f"Total Trade Execution Ticks: {len(df_trades):,}")

# Summary table per outcome
summary_df = df_l2.groupby("outcome_label").agg(
    l2_ticks=("mid_price", "count"),
    min_time=("timestamp_utc", "min"),
    max_time=("timestamp_utc", "max"),
    avg_mid=("mid_price", "mean"),
    avg_spread_cents=("spread_cents", "mean")
).reset_index()

print("\\nJuly 1–29 Outcome Summary Table:")
print(summary_df)
"""))

    # Cell 3: Synchronized Replay Engine Execution
    nb.cells.append(nbf.v4.new_code_cell("""# Run Synchronized Microsecond Event Stream & Driver Tagging
df_events = classify_event_drivers(df_l2, df_trades)
df_shocks = analyze_cross_outcome_shocks(df_events, jump_threshold_cents=1.0)

print(f"Unified Event Stream Total Count (July 1 - July 29): {len(df_events):,}")
print(f"Identified Cross-Outcome Shock Events: {len(df_shocks):,}")
"""))

    # Cell 4: SciencePlots Figure 1 - Full July 1-29 Overview & July 29 Shock Inset
    nb.cells.append(nbf.v4.new_code_cell("""# SciencePlots Plot 1: Full July 1–29 Implied Probabilities & Spreads
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)

colors = {"No Change": "#1f77b4", "Increase 25 bps": "#d62728", "Decrease 25 bps": "#2ca02c"}

for outcome, color in colors.items():
    sub_o = df_events[df_events["outcome_label"] == outcome]
    if len(sub_o) > 0:
        ax1.plot(sub_o["timestamp_utc"], sub_o["mid_price"], label=outcome, color=color, lw=1.0)
        ax2.plot(sub_o["timestamp_utc"], sub_o["spread_cents"], label=outcome, color=color, lw=0.8, alpha=0.7)

ax1.set_ylabel(r"Implied Prob ($P_{\text{mid}}$)")
ax1.set_title(r"\textbf{July 1--29, 2026 FOMC Contract Replay (Top 3 Outcomes)}")
ax1.legend(loc="upper left", frameon=True)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.set_ylabel(r"Spread (Cents)")
ax2.set_xlabel(r"UTC Timestamp (July 2026)")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plot_path_1 = "plots/fomc_july_full_month_overview.png"
plt.savefig(plot_path_1)
plt.close()
print(f"Saved SciencePlots full month overview to {plot_path_1}")
"""))

    # Cell 5: SciencePlots Figure 2 - Market Maker Reaction Latency CDF
    nb.cells.append(nbf.v4.new_code_cell("""# SciencePlots Plot 2: Market Maker Reaction Latency CDF & Taker Arbitrage
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

if len(df_shocks) > 0 and "mm_latency_ms" in df_shocks.columns:
    valid_lat = df_shocks["mm_latency_ms"].dropna()
    valid_lat = valid_lat[valid_lat > 0]
    
    sorted_lat = np.sort(valid_lat)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
    
    ax.plot(sorted_lat, cdf, color="#1f77b4", lw=1.5, label=r"MM Reaction Latency ($\tau_{\text{MM}}$)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Market Maker Reaction Latency $\tau_{\text{MM}}$ (ms)")
    ax.set_ylabel(r"Cumulative Probability")
    ax.set_title(r"\textbf{Cross-Outcome Market Maker Quote Adjustment Speed}")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

plt.tight_layout()
plot_path_2 = "plots/fomc_july_mm_latency_cdf.png"
plt.savefig(plot_path_2)
plt.close()
print(f"Saved SciencePlots MM latency CDF figure to {plot_path_2}")
"""))

    # Cell 6: SciencePlots Figure 3 - Maker vs Taker Signed Price Impact
    nb.cells.append(nbf.v4.new_code_cell("""# SciencePlots Plot 3: Taker Trade Size vs Price Impact
df_trades_all = df_events[df_events["is_trade"]].copy()

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

if len(df_trades_all) > 0:
    for driver, color, lbl in [("TAKER_BUY_YES", "#2ca02c", "Taker Buy YES"), ("TAKER_SELL_YES", "#d62728", "Taker Sell YES")]:
        sub_d = df_trades_all[df_trades_all["event_driver"] == driver]
        if len(sub_d) > 10:
            sizes = sub_d["usd_amount"]
            prices = sub_d["price"]
            ax.scatter(sizes, prices, color=color, alpha=0.4, s=15, label=lbl)

    ax.set_xscale("log")
    ax.set_xlabel(r"Taker Trade Size (\$USD)")
    ax.set_ylabel(r"Execution Price ($P$)")
    ax.set_title(r"\textbf{Taker Execution Size vs Price Impact (July 1--29)}")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

plt.tight_layout()
plot_path_3 = "plots/fomc_july_taker_impact_size.png"
plt.savefig(plot_path_3)
plt.close()
print(f"Saved SciencePlots Taker Impact figure to {plot_path_3}")
"""))

    # Cell 7: Interactive Animated HTML Dashboard Generation
    nb.cells.append(nbf.v4.new_code_cell("""# Generate Interactive Animated HTML Replay Dashboard with Time Slider & Driver Callouts
from build_interactive_dashboard_v2 import main as generate_dashboard_v2

generate_dashboard_v2()
print("Interactive Animated HTML Replay Dashboard generated successfully.")
"""))

    # Cell 8: Display HTML Dashboard
    nb.cells.append(nbf.v4.new_code_cell("""# Embed Interactive Animated HTML Dashboard
from IPython.display import IFrame
IFrame(src="data/july_fomc_interactive_animated_replay.html", width="100%", height=750)
"""))

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Successfully created notebook {NOTEBOOK_PATH}!")

if __name__ == "__main__":
    build_notebook()
