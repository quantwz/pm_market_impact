"""
execute_fomc_july_notebook.py
==============================
Executes full July 1–29 analysis, generates SciencePlots figures,
and creates the interactive animated HTML replay dashboard.
"""

import sys
import os
import shutil
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# SciencePlots setup
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
    print("Loaded SciencePlots style successfully.")
except Exception as e:
    print("SciencePlots fallback:", e)
    plt.style.use('default')

os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

import build_full_orderbook_replay
import build_july_fomc_replay_notebook

def main():
    print("1. Rebuilding full orderbook DOM ladders and HTML dashboard...", flush=True)
    build_full_orderbook_replay.main()
    
    print("2. Generating SciencePlots figures across full July 1–29 dataset...", flush=True)
    con = duckdb.connect("data/polymarket_july_fomc_pmxt_replay.db")
    df_l2 = con.execute("SELECT outcome_label, timestamp_utc, mid_price, spread_cents FROM pmxt_july_fomc_l2_ticks ORDER BY timestamp_utc ASC").df()
    df_trades = con.execute("SELECT outcome_label, timestamp_utc, price, usd_amount, 'BUY' as side FROM pmxt_july_fomc_trade_ticks ORDER BY timestamp_utc ASC").df()
    con.close()
    
    df_l2["timestamp_utc"] = pd.to_datetime(df_l2["timestamp_utc"], utc=True)
    df_trades["timestamp_utc"] = pd.to_datetime(df_trades["timestamp_utc"], utc=True)
    
    # Plot 1: Full Month Overview
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)
    colors = {"No Change": "#1f77b4", "Increase 25 bps": "#d62728", "Decrease 25 bps": "#2ca02c"}

    for outcome, color in colors.items():
        sub_o = df_l2[df_l2["outcome_label"] == outcome]
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
    print(f"Saved {plot_path_1}", flush=True)

    # Plot 2: Taker Execution Flow & Size
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    if len(df_trades) > 0:
        ax.scatter(df_trades["usd_amount"], df_trades["price"], color="#1f77b4", alpha=0.4, s=15, label="Trade Execution")
        ax.set_xscale("log")
        ax.set_xlabel(r"Trade Size (\$USD)")
        ax.set_ylabel(r"Execution Price ($P$)")
        ax.set_title(r"\textbf{Taker Execution Size vs Price (July 1--29)}")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")

    plt.tight_layout()
    plot_path_3 = "plots/fomc_july_taker_impact_size.png"
    plt.savefig(plot_path_3)
    plt.close()
    print(f"Saved {plot_path_3}", flush=True)

    # Build notebook
    build_july_fomc_replay_notebook.build_notebook()
    print("\n=== All Scripts & Replay Artifacts Generated Successfully! ===", flush=True)

if __name__ == "__main__":
    main()
