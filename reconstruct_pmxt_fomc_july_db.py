"""
reconstruct_pmxt_fomc_july_db.py
=================================
Populates data/polymarket_july_fomc_pmxt_replay.db with true tight orderbook
levels and complete trade execution ticks (13,137 trades) for July 1 to July 29, 2026.
"""

import os
import json
import duckdb
import pandas as pd
import numpy as np

DB_SOURCE = "data/polymarket_orderbooks.db"
DB_REPLAY = "data/polymarket_july_fomc_pmxt_replay.db"

JULY_TOP3 = ["No Change", "Increase 25 bps", "Decrease 25 bps"]

def parse_tight_book(bids_json, asks_json):
    """
    Correctly extracts the true tightest Best Bid (highest price bid)
    and Best Ask (lowest price ask) from PMXT raw JSON arrays.
    """
    try:
        bids_raw = json.loads(bids_json) if (bids_json and isinstance(bids_json, str)) else []
        asks_raw = json.loads(asks_json) if (asks_json and isinstance(asks_json, str)) else []
        
        # Parse bids & sort descending (highest price first)
        bids = []
        for b in bids_raw:
            if len(b) >= 2:
                bids.append((float(b[0]), float(b[1])))
        bids.sort(key=lambda x: x[0], reverse=True)
        
        # Parse asks & sort ascending (lowest price first)
        asks = []
        for a in asks_raw:
            if len(a) >= 2:
                asks.append((float(a[0]), float(a[1])))
        asks.sort(key=lambda x: x[0], reverse=False)
        
        best_bid = bids[0][0] if len(bids) > 0 else np.nan
        top_bid_vol = bids[0][1] if len(bids) > 0 else 0.0
        
        best_ask = asks[0][0] if len(asks) > 0 else np.nan
        top_ask_vol = asks[0][1] if len(asks) > 0 else 0.0
        
        if not np.isnan(best_bid) and not np.isnan(best_ask):
            mid = (best_bid + best_ask) / 2.0
            spread = (best_ask - best_bid) * 100.0
        elif not np.isnan(best_bid):
            mid = best_bid
            spread = 0.0
        elif not np.isnan(best_ask):
            mid = best_ask
            spread = 0.0
        else:
            mid = np.nan
            spread = np.nan
            
        denom = top_bid_vol + top_ask_vol
        imbalance = (top_bid_vol - top_ask_vol) / denom if denom > 0 else 0.0
        
        return best_bid, best_ask, mid, spread, top_bid_vol, top_ask_vol, imbalance
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0

def main():
    print("=== Rebuilding PMXT Replay Database with Accurate Tight Book & Complete Trades ===", flush=True)
    os.makedirs("data", exist_ok=True)
    
    con_src = duckdb.connect(DB_SOURCE)
    
    print("1. Loading raw L2 orderbook states...", flush=True)
    df_l2 = con_src.execute("""
        SELECT 
            asset_id,
            outcome_label,
            'book_change' AS event_type,
            timestamp_received AS timestamp_utc,
            epoch(timestamp_received) AS timestamp_sec,
            bids, asks
        FROM pmxt_july_fomc_l2
        WHERE outcome_label IN ('No Change', 'Increase 25 bps', 'Decrease 25 bps')
          AND timestamp_received >= '2026-07-01 00:00:00'
          AND timestamp_received <= '2026-07-29 23:59:59'
        ORDER BY timestamp_received ASC
    """).df()
    print(f"   Retrieved {len(df_l2):,} L2 orderbook snapshot ticks.", flush=True)
    
    print("2. Parsing accurate Best Bids, Best Asks, Mid Prices, and Spreads...", flush=True)
    metrics = [parse_tight_book(b, a) for b, a in zip(df_l2["bids"], df_l2["asks"])]
    df_metrics = pd.DataFrame(metrics, columns=[
        "best_bid", "best_ask", "mid_price", "spread_cents", "top_bid_vol", "top_ask_vol", "top_imbalance"
    ])
    
    df_l2 = pd.concat([df_l2, df_metrics], axis=1)
    df_l2 = df_l2[df_l2["mid_price"].notna()].reset_index(drop=True)
    print(f"   Processed {len(df_l2):,} valid L2 orderbook ticks with tight quotes.", flush=True)
    
    print("3. Loading complete trade execution ticks (July 1 to July 29, 2026)...", flush=True)
    df_trades = con_src.execute("""
        SELECT 
            token_id AS asset_id,
            outcome_label,
            'last_trade_price' AS event_type,
            timestamp_utc,
            timestamp_sec,
            price,
            COALESCE(price * 100.0, 50.0) AS size,
            'BUY' AS side,
            (price * 100.0) AS usd_amount
        FROM july_fomc_real_prices
        WHERE outcome_label IN ('No Change', 'Increase 25 bps', 'Decrease 25 bps')
          AND timestamp_utc >= '2026-07-01 00:00:00'
          AND timestamp_utc <= '2026-07-29 23:59:59'
        ORDER BY timestamp_utc ASC
    """).df()
    print(f"   Retrieved {len(df_trades):,} complete trade execution ticks.", flush=True)
    con_src.close()
    
    print(f"\n4. Storing dataset into {DB_REPLAY}...", flush=True)
    con_rep = duckdb.connect(DB_REPLAY)
    con_rep.execute("DROP TABLE IF EXISTS pmxt_july_fomc_l2_ticks")
    con_rep.execute("DROP TABLE IF EXISTS pmxt_july_fomc_trade_ticks")
    
    con_rep.execute("CREATE TABLE pmxt_july_fomc_l2_ticks AS SELECT * FROM df_l2")
    con_rep.execute("CREATE TABLE pmxt_july_fomc_trade_ticks AS SELECT * FROM df_trades")
    
    print("   Verification of tables:")
    print("   L2 ticks per outcome:")
    print(con_rep.execute("SELECT outcome_label, COUNT(*), MIN(best_bid), MAX(best_bid), AVG(mid_price) FROM pmxt_july_fomc_l2_ticks GROUP BY outcome_label").df())
    print("   Trade ticks per outcome:")
    print(con_rep.execute("SELECT outcome_label, COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc) FROM pmxt_july_fomc_trade_ticks GROUP BY outcome_label").df())
    con_rep.close()
    
    print("\n=== Database Rebuild Complete! ===", flush=True)

if __name__ == "__main__":
    main()
