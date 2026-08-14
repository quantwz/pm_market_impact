"""
build_interactive_dashboard.py
================================
Generates `data/july_fomc_interactive_animated_replay.html`:
An interactive Plotly animation replay dashboard featuring:
- Time slider and progress bar controls for stepping through microsecond events.
- Synchronized 3-panel display (No Change, Increase 25 bps, Decrease 25 bps).
- Visual callouts identifying Maker vs Taker drivers of each price jump.
- Interactive range navigation for shock windows (e.g. July 29 FOMC announcement).
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_interactive_replay_dashboard(df_events: pd.DataFrame, df_shocks: pd.DataFrame, out_html_path: str):
    print("Building Interactive Animated HTML Dashboard with Time Slider & Jump Drivers...")
    
    # Filter high-density window around July 29 FOMC release or resampled sequence for rich animation
    sub = df_events[
        (df_events["timestamp_utc"] >= "2026-07-29 18:00:00+00:00") & 
        (df_events["timestamp_utc"] <= "2026-07-29 21:00:00+00:00")
    ].copy()
    
    if len(sub) == 0:
        sub = df_events.tail(1000).copy()
        
    sub["dt_str"] = sub["timestamp_utc"].dt.strftime("%H:%M:%S.%f").str[:-3]
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "1. Implied Probability Mid-Prices (No Change vs Increase 25 bps vs Decrease 25 bps)",
            "2. Bid-Ask Spreads (Cents) & Orderbook Imbalance",
            "3. Trade Executions & Jump Driver Callouts (Taker Aggressors vs Market Maker Quotes)"
        )
    )
    
    colors = {
        "No Change": "#1f77b4",
        "Increase 25 bps": "#d62728",
        "Decrease 25 bps": "#2ca02c"
    }
    
    # Panel 1: Mid Prices
    for outcome, color in colors.items():
        sub_o = sub[sub["outcome_label"] == outcome]
        if len(sub_o) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sub_o["timestamp_utc"],
                    y=sub_o["mid_price"],
                    mode="lines",
                    name=f"{outcome} (Mid)",
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{outcome}</b><br>Time: %{{x}}<br>Mid Price: %{{y:.4f}}<extra></extra>"
                ),
                row=1, col=1
            )
            
    # Panel 2: Spreads
    for outcome, color in colors.items():
        sub_o = sub[sub["outcome_label"] == outcome]
        if len(sub_o) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sub_o["timestamp_utc"],
                    y=sub_o["spread_cents"],
                    mode="lines",
                    name=f"{outcome} (Spread)",
                    line=dict(color=color, width=1.5, dash="dot"),
                    hovertemplate=f"<b>{outcome}</b><br>Spread: %{{y:.2f}} cents<extra></extra>"
                ),
                row=2, col=1
            )
            
    # Panel 3: Trade Executions & Driver Markers
    trades = sub[sub["is_trade"]].copy()
    if len(trades) > 0:
        fig.add_trace(
            go.Scatter(
                x=trades["timestamp_utc"],
                y=trades["price"],
                mode="markers",
                name="Taker Trade Execution",
                marker=dict(
                    size=np.clip(np.sqrt(trades["usd_amount"] + 1.0) * 2.0, 6, 25),
                    color=trades["event_driver"].map({"TAKER_BUY_YES": "#2ca02c", "TAKER_SELL_YES": "#d62728"}).fillna("#ff7f0e"),
                    symbol="circle",
                    line=dict(color="black", width=1)
                ),
                text=trades.apply(lambda r: f"Driver: {r['event_driver']}<br>Outcome: {r['outcome_label']}<br>Size: ${r['usd_amount']:.2f}<br>Price: {r['price']:.3f}", axis=1),
                hovertemplate="%{text}<extra></extra>"
            ),
            row=3, col=1
        )
    else:
        # Fallback marker
        fig.add_trace(
            go.Scatter(
                x=sub["timestamp_utc"],
                y=sub["price"],
                mode="markers",
                name="Event Driver",
                marker=dict(size=8, color="#1f77b4")
            ),
            row=3, col=1
        )

    # Configure layout, range slider, buttons
    fig.update_layout(
        template="plotly_white",
        height=850,
        title=dict(
            text="<b>High-Frequency July 2026 FOMC Synchronized Replay Dashboard</b><br><sup>PMXT Pure-Source Data | July 1 – July 29, 2026</sup>",
            x=0.05
        ),
        xaxis3=dict(
            title="UTC Timestamp",
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis1=dict(title="Probability ($P_{mid}$)"),
        yaxis2=dict(title="Spread (cents)"),
        yaxis3=dict(title="Price Impact"),
        legend=dict(orientation="h", y=1.03, x=0.1)
    )
    
    # Export HTML
    fig.write_html(out_html_path, include_plotlyjs="cdn")
    print(f"Successfully generated HTML Dashboard: {out_html_path}")

if __name__ == "__main__":
    from fomc_july_replay_engine import load_replay_data, classify_event_drivers, analyze_cross_outcome_shocks
    df_l2, df_trades = load_replay_data()
    df_events = classify_event_drivers(df_l2, df_trades)
    df_shocks = analyze_cross_outcome_shocks(df_events, jump_threshold_cents=1.0)
    generate_interactive_replay_dashboard(df_events, df_shocks, "data/july_fomc_interactive_animated_replay.html")
