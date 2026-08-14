import json
import os

nb_path = "cross_impact_september_fomc.ipynb"

print(f"=== BUILDING DEDICATED NOTEBOOK: '{nb_path}' ===")

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(code):
    lines = code.split("\n")
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + ("\n" if i < len(lines) - 1 else "") for i, line in enumerate(lines)]
    })

# CELL 0: Title & Executive Summary
add_md("""# Cross-Impact Analysis & News Shock Replay in September 2026 FOMC Prediction Markets

**Authors**: Pair Programming & Quantitative Research Team  
**Date**: August 2026  
**Target Audience**: Collaborating Quantitative Researchers & Data Scientists  
**Research Focus**: September 2026 FOMC Contract Dynamics, News Shock Replay & 1-Min Cross-Jump Matrix Analysis  

---

## Executive Summary & Research Framework

This dedicated research notebook extends our high-frequency market impact framework to the **September 2026 FOMC Interest Rate Outcome Market** (`pmxt_september_fomc_l2` & `september_fomc_real_prices`) in `data/polymarket_orderbooks.db`.

---""")

# CELL 1: Setup Imports and Extraction API
code_cell_1 = r"""import os
import duckdb
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import scienceplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Apply user global rule for Matplotlib plots
plt.style.use(['science', 'no-latex'])

# Universal Database Path
UNIVERSAL_DB = "data/polymarket_orderbooks.db"

# Master Macro News Shocks Registry
NEWS_SHOCKS = {
    "June CPI Inflation Release": {
        "timestamp_utc": "2026-07-14 12:30:00",
        "description": "US BLS June CPI report release (8:30 AM ET / 12:30 UTC)"
    },
    "June Non-Farm Payrolls (NFP)": {
        "timestamp_utc": "2026-07-02 12:30:00",
        "description": "US BLS Employment Situation / NFP report release"
    },
    "June FOMC Rate Decision": {
        "timestamp_utc": "2026-06-17 18:00:00",
        "description": "Fed Interest Rate Decision statement & Dot Plot release"
    },
    "Pre-CPI Policy Speech": {
        "timestamp_utc": "2026-07-13 15:00:00",
        "description": "Fed Governor Policy Speech at NY Association for Business Economics"
    }
}

def query_september_fomc_data(db_path=UNIVERSAL_DB):
    con = duckdb.connect(db_path, read_only=True)
    df_l2 = con.execute("SELECT outcome_label, timestamp_received AS timestamp_utc, best_bid, best_ask, mid_price, spread_cents, top_imbalance, top_bid_vol, top_ask_vol FROM pmxt_september_fomc_l2 WHERE mid_price IS NOT NULL ORDER BY timestamp_received ASC").df()
    df_l2['timestamp_utc'] = pd.to_datetime(df_l2['timestamp_utc'], utc=True)
    
    df_ticks = con.execute("SELECT outcome_label, timestamp_utc, price, timestamp_sec FROM september_fomc_real_prices WHERE price IS NOT NULL ORDER BY timestamp_utc ASC").df()
    df_ticks['timestamp_utc'] = pd.to_datetime(df_ticks['timestamp_utc'], utc=True)
    con.close()
    
    return df_l2, df_ticks

df_l2_all, df_ticks_all = query_september_fomc_data()
print("Successfully loaded September 2026 FOMC Orderbook & Trade Execution Datasets!")
print(f"L2 Snapshots: {len(df_l2_all):,} rows | Trade Ticks: {len(df_ticks_all):,} rows")"""

add_code(code_cell_1)

# CELL 2: Core Cross-Impact Calculation Engine
code_cell_2 = r"""# ==========================================
# SEPTEMBER FOMC CROSS-IMPACT ANALYSIS ENGINE
# ==========================================

def analyze_news_shock_cross_impact(df_l2, df_ticks, news_time_str, shock_name="News Shock", time_horizons_min=[1, 2, 5, 10, 15, 30, 60]):
    t_news = pd.to_datetime(news_time_str, utc=True)
    
    post_news_trades = df_ticks[df_ticks['timestamp_utc'] >= t_news].sort_values('timestamp_utc')
    if len(post_news_trades) == 0:
        print(f"No trades found after {t_news} for {shock_name}")
        return None
        
    anchor_trade = post_news_trades.iloc[0]
    t_anchor = anchor_trade['timestamp_utc']
    anchor_outcome = anchor_trade['outcome_label']
    anchor_price = anchor_trade['price']
    
    outcomes = df_l2['outcome_label'].unique()
    results = []
    
    for outcome in outcomes:
        sub_l2 = df_l2[df_l2['outcome_label'] == outcome].sort_values('timestamp_utc').copy()
        sub_ticks = df_ticks[df_ticks['outcome_label'] == outcome].sort_values('timestamp_utc').copy()
        
        p_anchor_sub = sub_l2[sub_l2['timestamp_utc'] <= t_anchor]['mid_price'].iloc[-1] if len(sub_l2[sub_l2['timestamp_utc'] <= t_anchor]) > 0 else np.nan
        
        for k in time_horizons_min:
            t_pre = t_anchor - pd.Timedelta(minutes=k)
            t_post = t_anchor + pd.Timedelta(minutes=k)
            
            l2_pre = sub_l2[(sub_l2['timestamp_utc'] >= t_pre) & (sub_l2['timestamp_utc'] <= t_anchor)]
            ticks_pre = sub_ticks[(sub_ticks['timestamp_utc'] >= t_pre) & (sub_ticks['timestamp_utc'] <= t_anchor)]
            p_pre = l2_pre['mid_price'].iloc[0] if len(l2_pre) > 0 else p_anchor_sub
            dp_pre = p_anchor_sub - p_pre if not pd.isna(p_anchor_sub) and not pd.isna(p_pre) else 0.0
            vol_pre = len(ticks_pre)
            imb_pre = l2_pre['top_imbalance'].mean() if len(l2_pre) > 0 and 'top_imbalance' in l2_pre.columns else 0.0
            
            l2_post = sub_l2[(sub_l2['timestamp_utc'] >= t_anchor) & (sub_l2['timestamp_utc'] <= t_post)]
            ticks_post = sub_ticks[(sub_ticks['timestamp_utc'] >= t_anchor) & (sub_ticks['timestamp_utc'] <= t_post)]
            p_post = l2_post['mid_price'].iloc[-1] if len(l2_post) > 0 else p_anchor_sub
            dp_post = p_post - p_anchor_sub if not pd.isna(p_anchor_sub) and not pd.isna(p_post) else 0.0
            vol_post = len(ticks_post)
            imb_post = l2_post['top_imbalance'].mean() if len(l2_post) > 0 and 'top_imbalance' in l2_post.columns else 0.0
            
            vol_ratio = vol_post / vol_pre if vol_pre > 0 else (vol_post if vol_post > 0 else 1.0)
            
            results.append({
                'Shock Name': shock_name,
                'Anchor Time (t^a)': t_anchor,
                'Anchor Outcome': anchor_outcome,
                'Outcome Label': outcome,
                'Horizon (min)': k,
                'Pre Price': round(p_pre, 4) if not pd.isna(p_pre) else np.nan,
                'Anchor Price': round(p_anchor_sub, 4) if not pd.isna(p_anchor_sub) else np.nan,
                'Post Price': round(p_post, 4) if not pd.isna(p_post) else np.nan,
                'ΔP Pre': round(dp_pre, 4),
                'ΔP Post': round(dp_post, 4),
                'Pre Trades': vol_pre,
                'Post Trades': vol_post,
                'Volume Surge (x)': round(vol_ratio, 2),
                'Pre Imbalance': round(imb_pre, 3),
                'Post Imbalance': round(imb_post, 3)
            })
            
    df_res = pd.DataFrame(results)
    
    anchor_post_dp = df_res[df_res['Outcome Label'] == anchor_outcome].set_index('Horizon (min)')['ΔP Post'].to_dict()
    df_res['Anchor ΔP Post'] = df_res['Horizon (min)'].map(anchor_post_dp)
    df_res['Cross-Impact Elasticity (β)'] = np.where(
        np.abs(df_res['Anchor ΔP Post']) > 1e-4,
        (df_res['ΔP Post'] / df_res['Anchor ΔP Post']).round(3),
        0.0
    )
    
    return df_res, t_anchor, anchor_outcome

df_cpi_impact, t_anc_cpi, anc_out_cpi = analyze_news_shock_cross_impact(df_l2_all, df_ticks_all, NEWS_SHOCKS["June CPI Inflation Release"]["timestamp_utc"], "June CPI Inflation Release")

print(f"=== SEPTEMBER FOMC NEWS SHOCK REPLAY: June CPI Inflation Release ===")
print(f"News Release Scheduled: {NEWS_SHOCKS['June CPI Inflation Release']['timestamp_utc']} UTC")
print(f"Anchor Trade Execution (t^a): {t_anc_cpi} UTC")
print(f"Anchor Outcome Perturbed: '{anc_out_cpi}'")
print("\nSample Cross-Impact Results for 5-Minute Horizon:")
print(df_cpi_impact[df_cpi_impact['Horizon (min)'] == 5][['Outcome Label', 'Horizon (min)', 'Pre Price', 'Anchor Price', 'Post Price', 'ΔP Post', 'Post Trades', 'Volume Surge (x)', 'Post Imbalance', 'Cross-Impact Elasticity (β)']].to_string(index=False))"""

add_code(code_cell_2)

# CELL 3: Multi-Shock Analysis Across Horizons Table
code_cell_3 = r"""# ==========================================
# SEPTEMBER FOMC MULTI-SHOCK SUMMARY TABLES
# ==========================================

shock_summaries = []

for shock_name, info in NEWS_SHOCKS.items():
    res_tuple = analyze_news_shock_cross_impact(df_l2_all, df_ticks_all, info['timestamp_utc'], shock_name)
    if res_tuple:
        df_imp, t_anc, anc_out = res_tuple
        sub_5m = df_imp[df_imp['Horizon (min)'] == 5]
        for _, row in sub_5m.iterrows():
            shock_summaries.append({
                'News Shock': shock_name,
                'Anchor Time (t^a)': str(t_anc)[11:19],
                'Anchor Outcome': anc_out,
                'Target Outcome': row['Outcome Label'],
                '5m ΔP Pre': row['ΔP Pre'],
                '5m ΔP Post': row['ΔP Post'],
                '5m Trades Pre': row['Pre Trades'],
                '5m Trades Post': row['Post Trades'],
                'Volume Surge': f"{row['Volume Surge (x)']}x",
                '5m Post Imbalance': row['Post Imbalance'],
                'Cross Elasticity (β)': row['Cross-Impact Elasticity (β)']
            })

df_all_shocks = pd.DataFrame(shock_summaries)
print("=== COMPREHENSIVE SEPTEMBER FOMC MULTI-SHOCK CROSS-IMPACT SUMMARY (5-MIN HORIZON) ===")
print(df_all_shocks.to_string(index=False))"""

add_code(code_cell_3)

# CELL 4: Plotly Multi-Row News Shock Grid Dashboard (Price Impact & Cumulative Volume Panels)
code_cell_4 = r"""# ==========================================
# INTERACTIVE PLOTLY SEPTEMBER FOMC GRID DASHBOARD
# ==========================================

def plot_plotly_multi_row_news_grid(df_l2, df_ticks, news_shocks_dict):
    n_shocks = len(news_shocks_dict)
    
    subplot_titles = []
    for shock_name in news_shocks_dict.keys():
        subplot_titles.append(f"<b>{shock_name}</b>: Sept FOMC Price Impact $\Delta P(t)$ (¢)")
        subplot_titles.append(f"<b>{shock_name}</b>: Cumulative Trading Volume $[-60m, +60m]$")
        
    fig = make_subplots(
        rows=n_shocks, cols=2,
        shared_xaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.07,
        subplot_titles=subplot_titles
    )
    
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
    outcomes = df_l2['outcome_label'].unique()
    
    for row_idx, (shock_name, info) in enumerate(news_shocks_dict.items(), start=1):
        t_news = pd.to_datetime(info['timestamp_utc'], utc=True)
        t_start = t_news - pd.Timedelta(minutes=60)
        t_end = t_news + pd.Timedelta(minutes=60)
        
        post_trades = df_ticks[df_ticks['timestamp_utc'] >= t_news].sort_values('timestamp_utc')
        t_anchor = post_trades.iloc[0]['timestamp_utc'] if len(post_trades) > 0 else t_news
        
        for o_idx, outcome in enumerate(outcomes):
            l2_o = df_l2[(df_l2['outcome_label'] == outcome) & (df_l2['timestamp_utc'] >= t_start) & (df_l2['timestamp_utc'] <= t_end)].sort_values('timestamp_utc')
            ticks_o = df_ticks[(df_ticks['outcome_label'] == outcome) & (df_ticks['timestamp_utc'] >= t_start) & (df_ticks['timestamp_utc'] <= t_end)].sort_values('timestamp_utc')
            
            color = colors[o_idx % len(colors)]
            show_leg = (row_idx == 1)
            
            if len(l2_o) > 0:
                rel_min_l2 = (l2_o['timestamp_utc'] - t_anchor).dt.total_seconds() / 60.0
                p_anchor = l2_o[l2_o['timestamp_utc'] <= t_anchor]['mid_price'].iloc[-1] if len(l2_o[l2_o['timestamp_utc'] <= t_anchor]) > 0 else l2_o['mid_price'].iloc[0]
                dp_cents = (l2_o['mid_price'] - p_anchor) * 100.0
                
                fig.add_trace(
                    go.Scatter(x=rel_min_l2, y=dp_cents, name=outcome,
                               line=dict(color=color, width=1.5), legendgroup=outcome, showlegend=show_leg),
                    row=row_idx, col=1
                )
                
            if len(ticks_o) > 0:
                rel_min_ticks = (ticks_o['timestamp_utc'] - t_anchor).dt.total_seconds() / 60.0
                cum_vol = np.arange(1, len(ticks_o) + 1)
                
                fig.add_trace(
                    go.Scatter(x=rel_min_ticks, y=cum_vol, name=outcome + " Vol",
                               line=dict(color=color, width=1.5, shape='hv'), legendgroup=outcome, showlegend=False),
                    row=row_idx, col=2
                )
                
        fig.add_vline(x=0.0, line_width=1.2, line_dash="dash", line_color="red", row=row_idx, col=1)
        fig.add_vline(x=0.0, line_width=1.2, line_dash="dash", line_color="red", row=row_idx, col=2)
        
    fig.update_layout(
        height=320 * n_shocks,
        title_text="<b>September 2026 FOMC News Shock Replay Grid</b>: Price Impact & Cumulative Trading Volume $[-60m, +60m]$",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
    )
    
    for r in range(1, n_shocks + 1):
        fig.update_yaxes(title_text="ΔP (¢)", row=r, col=1)
        fig.update_yaxes(title_text="Cum. Trades", row=r, col=2)
        fig.update_xaxes(range=[-60, 60], row=r, col=1)
        fig.update_xaxes(range=[-60, 60], row=r, col=2)
        
    fig.update_xaxes(title_text="Relative Event Time Δt (Minutes from Anchor t^a)", row=n_shocks, col=1)
    fig.update_xaxes(title_text="Relative Event Time Δt (Minutes from Anchor t^a)", row=n_shocks, col=2)
    
    return fig

fig_grid = plot_plotly_multi_row_news_grid(df_l2_all, df_ticks_all, NEWS_SHOCKS)
print("Interactive September FOMC Multi-Row News Shock Grid Dashboard Engine Ready!")
fig_grid.show()"""

add_code(code_cell_4)

# CELL 5: SciencePlots Matplotlib Multi-Row Grid Figure Code
code_cell_5 = r"""# ==========================================
# SCIENCEPLOTS SEPTEMBER FOMC MULTI-ROW GRID FIGURE
# ==========================================

def plot_matplotlib_multi_row_news_grid(df_l2, df_ticks, news_shocks_dict):
    n_shocks = len(news_shocks_dict)
    fig, axes = plt.subplots(n_shocks, 2, figsize=(10, 2.8 * n_shocks), dpi=300)
    
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
    outcomes = df_l2['outcome_label'].unique()
    
    for row_idx, (shock_name, info) in enumerate(news_shocks_dict.items()):
        t_news = pd.to_datetime(info['timestamp_utc'], utc=True)
        t_start = t_news - pd.Timedelta(minutes=60)
        t_end = t_news + pd.Timedelta(minutes=60)
        
        post_trades = df_ticks[df_ticks['timestamp_utc'] >= t_news].sort_values('timestamp_utc')
        t_anchor = post_trades.iloc[0]['timestamp_utc'] if len(post_trades) > 0 else t_news
        
        ax_price = axes[row_idx, 0] if n_shocks > 1 else axes[0]
        ax_vol = axes[row_idx, 1] if n_shocks > 1 else axes[1]
        
        for o_idx, outcome in enumerate(outcomes):
            l2_sub = df_l2[(df_l2['outcome_label'] == outcome) & (df_l2['timestamp_utc'] >= t_start) & (df_l2['timestamp_utc'] <= t_end)].sort_values('timestamp_utc')
            ticks_sub = df_ticks[(df_ticks['outcome_label'] == outcome) & (df_ticks['timestamp_utc'] >= t_start) & (df_ticks['timestamp_utc'] <= t_end)].sort_values('timestamp_utc')
            
            color = colors[o_idx % len(colors)]
            label = outcome if row_idx == 0 else ""
            
            if len(l2_sub) > 0:
                rel_min_l2 = (l2_sub['timestamp_utc'] - t_anchor).dt.total_seconds() / 60.0
                p_anchor = l2_sub[l2_sub['timestamp_utc'] <= t_anchor]['mid_price'].iloc[-1] if len(l2_sub[l2_sub['timestamp_utc'] <= t_anchor]) > 0 else l2_sub['mid_price'].iloc[0]
                dp_cents = (l2_sub['mid_price'] - p_anchor) * 100.0
                
                ax_price.plot(rel_min_l2, dp_cents, label=label, color=color, linewidth=1.2)
                
            if len(ticks_sub) > 0:
                rel_min_ticks = (ticks_sub['timestamp_utc'] - t_anchor).dt.total_seconds() / 60.0
                cum_vol = np.arange(1, len(ticks_sub) + 1)
                
                ax_vol.step(rel_min_ticks, cum_vol, where='post', label=label, color=color, linewidth=1.2)
                
        ax_price.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
        ax_price.axhline(0, color='black', linestyle=':', linewidth=0.6, alpha=0.6)
        ax_price.set_title(f"{shock_name}: Sept FOMC Price Impact $\\Delta P(t)$ (¢)", fontsize=8)
        ax_price.set_ylabel("$\\Delta P$ (¢)", fontsize=7)
        ax_price.set_xlim([-60, 60])
        
        ax_vol.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
        ax_vol.set_title(f"{shock_name}: Cumulative Trade Volume", fontsize=8)
        ax_vol.set_ylabel("Cum. Trades", fontsize=7)
        ax_vol.set_xlim([-60, 60])
        
        if row_idx == n_shocks - 1:
            ax_price.set_xlabel("Relative Event Time $\\Delta t$ (Minutes from Anchor $t^a$)", fontsize=7)
            ax_vol.set_xlabel("Relative Event Time $\\Delta t$ (Minutes from Anchor $t^a$)", fontsize=7)
            
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=7)
    plt.tight_layout()
    plt.savefig("plots/september_fomc_news_shocks_grid.png", bbox_inches='tight')
    plt.show()
    print("Saved September FOMC multi-row grid figure to 'plots/september_fomc_news_shocks_grid.png'.")

plot_matplotlib_multi_row_news_grid(df_l2_all, df_ticks_all, NEWS_SHOCKS)"""

add_code(code_cell_5)

# CELL 6: Block 4 Full-Sample Statistical Price Shock Tranche Analysis Markdown
add_md("""## Block 4: Full-Sample Statistical Price Shock Tranche Analysis (September FOMC 1-Min Resolution)

### 4.1 Theoretical Formulation & News-Independent Tranche Definitions

We evaluate all step-by-step price changes $|\\Delta P_{\\text{mid}}|$ across the **entire September 2026 FOMC historical dataset (June 1 to August 3, 2026)** at 1-minute fixed calendar steps (90,000+ time steps).

$$\\text{Tranche}(|\\Delta P|) = \\begin{cases} 
\\mathbf{< 1.0\\text{¢}} & |\\Delta P_{\\text{mid}}| < 0.01 \\quad (\\text{Normal Background Noise / Spread Jiggle}) \\\\
\\mathbf{[1.0, 5.0\\text{¢}]} & 0.01 \\le |\\Delta P_{\\text{mid}}| \\le 0.05 \\quad (\\text{Moderate Price Shock}) \\\\
\\mathbf{(5.0, 10.0\\text{¢}]} & 0.05 < |\\Delta P_{\\text{mid}}| \\le 0.10 \\quad (\\text{Large Price Shock}) \\\\
\\mathbf{> 10.0\\text{¢}} & |\\Delta P_{\\text{mid}}| > 0.10 \\quad (\\text{Extreme Price Jump})
\\end{cases}$$""")

# CELL 7: Full-Sample Statistical Price Shock Tranche Code
code_cell_7 = r"""# ==========================================
# SEPTEMBER FOMC FULL-SAMPLE STATISTICAL TRANCHE ENGINE (1-MIN RESOLUTION)
# ==========================================

def compute_full_sample_statistical_tranches(df_l2, resample_freq="1min"):
    outcomes = df_l2['outcome_label'].unique()
    tranche_order = ["< 1¢", "[1, 5]¢", "(5, 10]¢", "> 10¢"]
    all_steps = []
    
    for outcome in outcomes:
        sub = df_l2[df_l2['outcome_label'] == outcome].drop_duplicates(subset=['timestamp_utc'], keep='last').set_index('timestamp_utc')
        resampled_p = sub['mid_price'].resample(resample_freq).last().ffill().dropna()
        dp = resampled_p.diff().dropna()
        abs_dp_cents = abs(dp) * 100.0
        
        for dt, val in abs_dp_cents.items():
            if val < 1.0:
                tranche = "< 1¢"
            elif 1.0 <= val <= 5.0:
                tranche = "[1, 5]¢"
            elif 5.0 < val <= 10.0:
                tranche = "(5, 10]¢"
            else:
                tranche = "> 10¢"
                
            prev_dt = dt - pd.Timedelta(resample_freq)
            p_prev = resampled_p.loc[prev_dt] if prev_dt in resampled_p.index else np.nan
            
            all_steps.append({
                'Outcome Label': outcome,
                'Timestamp UTC': dt,
                'Mid Price Before': round(p_prev, 4) if not pd.isna(p_prev) else np.nan,
                'Mid Price After': round(resampled_p.loc[dt], 4),
                'ΔP (¢)': round(dp.loc[dt] * 100.0, 2),
                'Abs ΔP (¢)': round(val, 2),
                'Tranche': tranche
            })
            
    df_steps = pd.DataFrame(all_steps)
    
    df_counts = df_steps.groupby(['Outcome Label', 'Tranche']).size().unstack(fill_value=0).reindex(columns=tranche_order, fill_value=0)
    df_pcts = df_counts.div(df_counts.sum(axis=1), axis=0) * 100.0
    df_extreme = df_steps[df_steps['Abs ΔP (¢)'] >= 1.0].sort_values('Abs ΔP (¢)', ascending=False)
    
    return df_counts, df_pcts, df_extreme

df_full_counts, df_full_pcts, df_full_extreme = compute_full_sample_statistical_tranches(df_l2_all, resample_freq="1min")

print("=== SEPTEMBER FOMC FULL-SAMPLE TRANCHE COUNT MATRIX (1-MIN STEPS) ===")
print(df_full_counts)

print("\n=== SEPTEMBER FOMC FULL-SAMPLE TRANCHE PERCENTAGE MATRIX (%) ===")
print(df_full_pcts.round(2))

print("\n=== TOP 10 STATISTICAL EXTREME SHOCKS IN SEPTEMBER FOMC DATA ===")
print(df_full_extreme.head(10)[['Outcome Label', 'Timestamp UTC', 'Mid Price Before', 'Mid Price After', 'ΔP (¢)', 'Abs ΔP (¢)', 'Tranche']].to_string(index=False))

# Plot Full Sample Stacked Bar Figure (SciencePlots Style)
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)

tranche_order = ["< 1¢", "[1, 5]¢", "(5, 10]¢", "> 10¢"]
tranche_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
bottom = np.zeros(len(df_full_pcts))

for idx, tr in enumerate(tranche_order):
    vals = df_full_pcts[tr].values
    ax.bar(df_full_pcts.index, vals, bottom=bottom, label=f"Tranche {tr}", color=tranche_colors[idx], width=0.55)
    bottom += vals

ax.set_title("September FOMC Full-Sample Statistical Price Shock Distribution (1-Min)", fontsize=10)
ax.set_ylabel("Share of Total 1-Minute Time Steps (%)", fontsize=9)
ax.set_xlabel("Outcome Contract", fontsize=9)
ax.set_ylim([0, 105])
ax.legend(fontsize=8, loc='upper right')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig("plots/september_fomc_tranches.png", bbox_inches='tight')
plt.show()
print("\nSaved September FOMC price shock tranche figure to 'plots/september_fomc_tranches.png'.")"""

add_code(code_cell_7)

# CELL 8: Block 5 Tranche [1, 5]¢ Pure Cross-Jump Matrix Engine (1-Min Resolution) Markdown
add_md("""## Block 5: Tranche [1, 5]¢ Pure Cross-Jump & Spillover Frequency Matrix $M_{A, B}$ (September FOMC 1-Min Resolution)

### 5.1 Formulation & Zero-Quiescence Weighting (Rule 3 Weight = 0)

We construct the **Pure Cross-Jump Matrix $M_{A, B}$** on **September 2026 FOMC** data at **1-minute resolution** (`resample_freq="1min"`) with **Rule 3 Weight = 0**.

$$\\text{Score}(A, B) \\leftarrow \\begin{cases}
+1 & \\text{if } 1.0\\text{¢} \\le |\\Delta P_{B, t}| \\le 5.0\\text{¢} \\quad (\\text{Rule 1: Co-Jump in Tranche } [1, 5]\\text{¢}) \\\\
+2 & \\text{if } |\\Delta P_{B, t}| > 5.0\\text{¢} \\quad (\\text{Rule 2: Leader-Follower Spillover Extreme Jump } > 5\\text{¢}) \\\\
0 & \\text{if } |\\Delta P_{B, t}| < 1.0\\text{¢} \\quad (\\text{Rule 3: Quiescent Step } < 1\\text{¢, Neutral Weight})
\\end{cases}$$""")

# CELL 9: Tranche [1, 5]¢ Pure Cross-Jump Matrix Code & Visualizations (1-Min Resolution)
code_cell_9 = r"""# ==========================================
# SEPTEMBER FOMC PURE CROSS-JUMP MATRIX ENGINE (1-MIN RESOLUTION)
# ==========================================

def compute_september_pure_cross_jump_matrix(df_l2, resample_freq="1min"):
    outcomes = list(df_l2['outcome_label'].unique())
    
    p_piv = df_l2.drop_duplicates(subset=['outcome_label', 'timestamp_utc'], keep='last') \
                  .pivot(index='timestamp_utc', columns='outcome_label', values='mid_price') \
                  .resample(resample_freq).last().ffill().dropna()
                  
    dp_cents = abs(p_piv.diff().dropna()) * 100.0
    
    M_pos = pd.DataFrame(0.0, index=outcomes, columns=outcomes)
    co_jump = pd.DataFrame(0, index=outcomes, columns=outcomes)
    spillover = pd.DataFrame(0, index=outcomes, columns=outcomes)
    
    for idx_step, row in dp_cents.iterrows():
        for o_A in outcomes:
            val_A = row[o_A]
            if 1.0 <= val_A <= 5.0:
                for o_B in outcomes:
                    val_B = row[o_B]
                    if o_A == o_B:
                        M_pos.loc[o_A, o_B] += 1.0
                        co_jump.loc[o_A, o_B] += 1
                    else:
                        if 1.0 <= val_B <= 5.0:
                            M_pos.loc[o_A, o_B] += 1.0
                            co_jump.loc[o_A, o_B] += 1
                        elif val_B > 5.0:
                            M_pos.loc[o_A, o_B] += 2.0
                            spillover.loc[o_A, o_B] += 1
                            
    return M_pos, co_jump, spillover, len(dp_cents)

M_pos, co_jump, spillover, n_steps = compute_september_pure_cross_jump_matrix(df_l2_all, resample_freq="1min")

print(f"=== SEPTEMBER FOMC PURE CROSS-JUMP MATRIX M (1-MIN RESOLUTION, {n_steps:,} STEPS) ===")
print(M_pos.round(1))

# Plot Recalculated Cross-Jump Matrix Heatmap (SciencePlots Style)
fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
im = ax.imshow(M_pos.values, cmap='YlOrRd', aspect='auto')

outcomes_labels = list(M_pos.index)
ax.set_xticks(np.arange(len(outcomes_labels)))
ax.set_yticks(np.arange(len(outcomes_labels)))
ax.set_xticklabels(outcomes_labels, rotation=30, ha='right', fontsize=8)
ax.set_yticklabels(outcomes_labels, fontsize=8)

for i in range(len(outcomes_labels)):
    for j in range(len(outcomes_labels)):
        val = int(M_pos.iloc[i, j])
        ax.text(j, i, f"{val}", ha="center", va="center", color="white" if val > 100 else "black", fontsize=8, weight="bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("1-Min Pure Cross-Jump Frequency $M(A, B)$", fontsize=8)
ax.set_title("September FOMC 1-Min Pure Cross-Jump Matrix $M_{A, B}$", fontsize=9)
plt.tight_layout()
plt.savefig("plots/september_fomc_cross_jump_matrix.png", bbox_inches='tight')
plt.show()
print("\nSaved September FOMC 1-min cross-jump matrix heatmap to 'plots/september_fomc_cross_jump_matrix.png'.")"""

add_code(code_cell_9)

# Save notebook
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 2}, f, indent=1)

print(f"Successfully built dedicated notebook '{nb_path}'!")
