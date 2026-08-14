"""
fomc_july_replay_engine.py
===========================
Replay and analytics engine for July 2026 FOMC top 3 outcomes:
- No Change
- Increase 25 bps
- Decrease 25 bps

Reconstructs microsecond synchronized event streams, classifies Maker vs Taker drivers,
computes Market Maker reaction latency (tau_MM), Taker latency arbitrage volume,
and cross-outcome shock propagation.
"""

import os
import sys
import json
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timezone

DB_REPLAY_PATH = "data/polymarket_july_fomc_pmxt_replay.db"
OUT_METRICS_PATH = "data/fomc_july_replay_metrics.parquet"
OUT_SHOCKS_PATH = "data/fomc_july_shock_events.parquet"

TOP3_OUTCOMES = ["No Change", "Increase 25 bps", "Decrease 25 bps"]

def load_replay_data():
    if not os.path.exists(DB_REPLAY_PATH):
        raise FileNotFoundError(f"Database {DB_REPLAY_PATH} does not exist. Run reconstruct_pmxt_fomc_july_db.py first.")
        
    con = duckdb.connect(DB_REPLAY_PATH)
    
    print("Loading L2 orderbook ticks...")
    df_l2 = con.execute("""
        SELECT 
            asset_id, outcome_label, event_type, timestamp_utc, timestamp_sec,
            best_bid, best_ask, mid_price, spread_cents, top_bid_vol, top_ask_vol, top_imbalance,
            bids, asks
        FROM pmxt_july_fomc_l2_ticks
        WHERE outcome_label IN ('No Change', 'Increase 25 bps', 'Decrease 25 bps')
        ORDER BY timestamp_utc ASC
    """).df()
    
    print(f"Loaded {len(df_l2):,} L2 orderbook ticks.")
    
    print("Loading trade ticks...")
    df_trades = con.execute("""
        SELECT 
            asset_id, outcome_label, event_type, timestamp_utc, timestamp_sec,
            price, size, side, usd_amount
        FROM pmxt_july_fomc_trade_ticks
        WHERE outcome_label IN ('No Change', 'Increase 25 bps', 'Decrease 25 bps')
        ORDER BY timestamp_utc ASC
    """).df()
    
    print(f"Loaded {len(df_trades):,} trade ticks.")
    con.close()
    
    return df_l2, df_trades

def classify_event_drivers(df_l2: pd.DataFrame, df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Merges L2 updates and trades into a single chronological timeline and tags event drivers:
    - TAKER_BUY_YES: Aggressive trade lifting the ask
    - TAKER_SELL_YES: Aggressive trade hitting the bid
    - MAKER_QUOTE_UPDATE: Market Maker updating bids/asks or size
    - MAKER_LIQUIDITY_WITHDRAWAL: Depth drop without a trade
    """
    print("Building unified microsecond event stream...")
    
    df_l2["is_trade"] = False
    df_l2["event_driver"] = "MAKER_QUOTE_UPDATE"
    df_l2["price"] = df_l2["mid_price"]
    df_l2["size"] = 0.0
    df_l2["usd_amount"] = 0.0
    
    df_trades["is_trade"] = True
    df_trades["best_bid"] = np.nan
    df_trades["best_ask"] = np.nan
    df_trades["mid_price"] = df_trades["price"]
    df_trades["spread_cents"] = np.nan
    df_trades["top_bid_vol"] = 0.0
    df_trades["top_ask_vol"] = 0.0
    df_trades["top_imbalance"] = 0.0
    df_trades["bids"] = ""
    df_trades["asks"] = ""
    
    # Infer trade direction (Taker Buy vs Taker Sell) if missing
    def tag_trade_side(row):
        side = str(row["side"]).upper()
        if "BUY" in side or side == "YES":
            return "TAKER_BUY_YES"
        elif "SELL" in side or side == "NO":
            return "TAKER_SELL_YES"
        else:
            return "TAKER_BUY_YES" # Default buy YES
            
    df_trades["event_driver"] = df_trades.apply(tag_trade_side, axis=1)
    
    # Combine and sort strictly by microsecond timestamp_utc
    df_merged = pd.concat([df_l2, df_trades], ignore_index=True)
    df_merged["timestamp_utc"] = pd.to_datetime(df_merged["timestamp_utc"], utc=True)
    df_merged = df_merged.sort_values("timestamp_utc").reset_index(drop=True)
    
    # Forward fill best_bid and best_ask per outcome to check for liquidity withdrawals
    for outcome in TOP3_OUTCOMES:
        mask = df_merged["outcome_label"] == outcome
        df_merged.loc[mask, "best_bid"] = df_merged.loc[mask, "best_bid"].ffill()
        df_merged.loc[mask, "best_ask"] = df_merged.loc[mask, "best_ask"].ffill()
        df_merged.loc[mask, "mid_price"] = df_merged.loc[mask, "mid_price"].ffill()
    
    print(f"Unified stream contains {len(df_merged):,} total events.")
    print("Driver distribution:")
    print(df_merged["event_driver"].value_counts())
    
    return df_merged

def analyze_cross_outcome_shocks(df_events: pd.DataFrame, jump_threshold_cents: float = 1.0) -> pd.DataFrame:
    """
    Identifies significant price jump shocks (> 1.0 cent move) in any of the top 3 outcomes
    and measures Market Maker reaction latency (tau_MM) and Taker trade volume in other outcomes.
    """
    print(f"\nAnalyzing cross-outcome shock propagation (Threshold >= {jump_threshold_cents} cents)...")
    
    shocks = []
    
    # Pivot mid-prices per outcome on resampled 100ms grid for shock detection
    df_events["dt_grid"] = df_events["timestamp_utc"].dt.floor("100ms")
    df_grid = df_events.groupby(["dt_grid", "outcome_label"])["mid_price"].last().unstack()
    df_grid = df_grid.ffill().dropna()
    
    df_diff = df_grid.diff() * 100.0 # diff in cents / bps
    
    for outcome_a in TOP3_OUTCOMES:
        if outcome_a not in df_diff.columns:
            continue
            
        jump_times = df_diff[df_diff[outcome_a].abs() >= jump_threshold_cents].index
        
        for t_jump in jump_times:
            jump_val = df_diff.loc[t_jump, outcome_a]
            
            # Find exact trigger event around t_jump
            sub_events = df_events[(df_events["timestamp_utc"] >= t_jump - pd.Timedelta(seconds=1)) & 
                                   (df_events["timestamp_utc"] <= t_jump + pd.Timedelta(seconds=5))]
            
            trigger_sub = sub_events[(sub_events["outcome_label"] == outcome_a) & (sub_events["is_trade"])]
            trigger_type = trigger_sub["event_driver"].iloc[0] if len(trigger_sub) > 0 else "MAKER_QUOTE_UPDATE"
            trigger_size = trigger_sub["usd_amount"].sum() if len(trigger_sub) > 0 else 0.0
            
            other_outcomes = [o for o in TOP3_OUTCOMES if o != outcome_a]
            
            for outcome_b in other_outcomes:
                if outcome_b not in df_diff.columns:
                    continue
                    
                # Look for subsequent Market Maker quote update in outcome_b
                post_b_events = sub_events[(sub_events["outcome_label"] == outcome_b) & 
                                           (sub_events["timestamp_utc"] >= t_jump)]
                
                mm_b_updates = post_b_events[post_b_events["event_driver"] == "MAKER_QUOTE_UPDATE"]
                
                if len(mm_b_updates) > 0:
                    t_mm_b = mm_b_updates["timestamp_utc"].iloc[0]
                    tau_mm_ms = (t_mm_b - t_jump).total_seconds() * 1000.0
                else:
                    t_mm_b = pd.NaT
                    tau_mm_ms = np.nan
                    
                # Calculate Taker trade volume in outcome_b during latency window tau_mm
                if pd.notna(t_mm_b):
                    arb_trades = post_b_events[(post_b_events["outcome_label"] == outcome_b) & 
                                             (post_b_events["is_trade"]) & 
                                             (post_b_events["timestamp_utc"] >= t_jump) & 
                                             (post_b_events["timestamp_utc"] <= t_mm_b)]
                    taker_arb_usd = arb_trades["usd_amount"].sum()
                    taker_arb_count = len(arb_trades)
                else:
                    taker_arb_usd = 0.0
                    taker_arb_count = 0
                    
                # Outcome B price move 5 seconds post jump
                price_b_pre = df_grid.loc[t_jump, outcome_b] if t_jump in df_grid.index else np.nan
                t_5s = t_jump + pd.Timedelta(seconds=5)
                price_b_post = df_grid.asof(t_5s)[outcome_b] if len(df_grid) > 0 else np.nan
                delta_b_5s = (price_b_post - price_b_pre) * 100.0 if (pd.notna(price_b_pre) and pd.notna(price_b_post)) else 0.0
                
                shocks.append({
                    "timestamp_utc": t_jump,
                    "source_outcome": outcome_a,
                    "target_outcome": outcome_b,
                    "source_jump_cents": jump_val,
                    "trigger_driver": trigger_type,
                    "trigger_usd_size": trigger_size,
                    "mm_latency_ms": tau_mm_ms,
                    "taker_arb_usd": taker_arb_usd,
                    "taker_arb_count": taker_arb_count,
                    "target_move_5s_cents": delta_b_5s
                })
                
    df_shocks = pd.DataFrame(shocks)
    print(f"Identified {len(df_shocks):,} cross-outcome shock events.")
    if len(df_shocks) > 0:
        print("\nSummary of MM Reaction Latency (ms):")
        print(df_shocks["mm_latency_ms"].describe())
        print("\nSummary of Taker Latency Arbitrage Volume ($):")
        print(df_shocks["taker_arb_usd"].describe())
        
    return df_shocks

def main():
    print("=== Running High-Frequency July 2026 FOMC Replay Engine ===")
    df_l2, df_trades = load_replay_data()
    
    df_events = classify_event_drivers(df_l2, df_trades)
    df_shocks = analyze_cross_outcome_shocks(df_events, jump_threshold_cents=1.0)
    
    print(f"\nSaving processed event replay stream to {OUT_METRICS_PATH}...")
    # Select clean columns for export
    export_cols = [
        "timestamp_utc", "timestamp_sec", "asset_id", "outcome_label", "event_type",
        "event_driver", "is_trade", "price", "size", "usd_amount",
        "best_bid", "best_ask", "mid_price", "spread_cents", "top_bid_vol", "top_ask_vol", "top_imbalance"
    ]
    df_events[export_cols].to_parquet(OUT_METRICS_PATH, index=False)
    
    print(f"Saving cross-outcome shock analysis to {OUT_SHOCKS_PATH}...")
    df_shocks.to_parquet(OUT_SHOCKS_PATH, index=False)
    
    print("=== Replay Engine Processing Complete! ===")

if __name__ == "__main__":
    main()
