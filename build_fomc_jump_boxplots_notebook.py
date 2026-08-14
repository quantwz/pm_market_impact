import json
import os

nb_path = "fomc_jump_cross_impact_boxplots.ipynb"

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
add_md("""# Anchored >5% Price Jump Cross Impact Analysis Across FOMC Rate Decisions

**Authors**: Pair Programming & Quantitative Research Team  
**Date**: August 2026  
**Target Audience**: Collaborating Quantitative Researchers & Data Scientists  
**Research Focus**: Pre-Jump Anchoring, Ranked Outcome Impact ($Top 1, Top 2, Top 3$) & Horizon Boxplot Distributions  

---

## Executive Summary & Analytical Framework

When a major price shock occurs in prediction markets, informed traders or liquidity shocks cause a sudden price jump $> 5\\%$ ($> 5\\text{¢}$) in one of the outcome contracts. This notebook detects all such statistical jump anchors across historical FOMC rate decision markets in `data/polymarket_fomc_trade.db` and computes the resulting cross-impact dynamics.

### Workflow & Boxplot Architecture:
1. **Pre-Jump Anchor Identification ($t_{\\text{anchor}}$)**:
   - Identifies any 1-minute step where an outcome price jumps by $> 5\\text{¢}$ ($|\\Delta P| > 0.05$).
   - Uses the **pre-jump timestamp** $t_{\\text{anchor}}$ as the baseline start time.

2. **Outcome Ranking by Probability at Anchor Start Time**:
   - Ranks all outcome contracts in the meeting event by their probability price as of $t_{\\text{anchor}}$:
     - `Top 1`: Highest probability contract at anchor start time
     - `Top 2`: Second highest probability contract at anchor start time
     - `Top 3`: Third highest probability contract at anchor start time

3. **Horizon Price Impact Distribution ($1\\text{m}$ to $60\\text{m}$)**:
   - Computes price impact $\\Delta P(O_j, k) = P(t_{\\text{anchor}} + k) - P(t_{\\text{anchor}})$ in cents across lookahead horizons $k \\in \\{1\\text{m}, 2\\text{m}, 5\\text{m}, 10\\text{m}, 15\\text{m}, 30\\text{m}, 60\\text{m}\\}$.

4. **Multi-Panel Boxplot Visualizations**:
   - Renders side-by-side boxplot distributions showing how `Top 1`, `Top 2`, and `Top 3` outcomes re-equilibrate over time following a $> 5\\%$ shock.

---""")

# CELL 1: Imports & Database Connection API
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

DB_PATH = "data/polymarket_fomc_trade.db"

def get_fomc_events_list(db_path=DB_PATH):
    con = duckdb.connect(db_path, read_only=True)
    events_df = con.execute('''
        SELECT event_slug, COUNT(*) as trade_ticks, MIN(timestamp_utc) as start_time, MAX(timestamp_utc) as end_time
        FROM fomc_trade_ticks
        GROUP BY event_slug
        HAVING trade_ticks >= 5000
        ORDER BY trade_ticks DESC
    ''').df()
    con.close()
    return events_df

df_events = get_fomc_events_list()
print(f"Loaded {len(df_events)} major FOMC events with >= 5,000 trade ticks:")
print(df_events.head(10).to_string(index=False))"""

add_code(code_cell_1)

# CELL 2: Core Anchored >5% Jump & Ranking Impact Engine
code_cell_2 = r"""# ==========================================
# ANCHORED >5% JUMP & RANKED IMPACT ENGINE
# ==========================================

def analyze_anchored_jump_impact(event_slug, db_path=DB_PATH, jump_threshold_cents=5.0, resample_freq="1min", horizons=[1, 2, 5, 10, 15, 30, 60]):
    con = duckdb.connect(db_path, read_only=True)
    df_event = con.execute(f'''
        SELECT event_slug, slug, question, outcome_label, timestamp_utc, price
        FROM fomc_trade_ticks
        WHERE event_slug = '{event_slug}'
          AND price IS NOT NULL
        ORDER BY timestamp_utc ASC
    ''').df()
    con.close()
    
    if len(df_event) == 0:
        return None, None
        
    df_event['timestamp_utc'] = pd.to_datetime(df_event['timestamp_utc'], utc=True)
    
    p_piv = df_event.drop_duplicates(subset=['outcome_label', 'timestamp_utc'], keep='last') \
                    .pivot(index='timestamp_utc', columns='outcome_label', values='price') \
                    .resample(resample_freq).last().ffill().dropna()
                    
    dp_cents = p_piv.diff()
    abs_dp = abs(dp_cents) * 100.0
    
    jump_mask = (abs_dp > jump_threshold_cents).any(axis=1)
    anchor_times = dp_cents.index[jump_mask]
    
    impact_records = []
    
    for t_anc in anchor_times:
        prev_time = t_anc - pd.Timedelta(resample_freq)
        p_pre = p_piv.loc[prev_time] if prev_time in p_piv.index else p_piv.loc[t_anc]
        
        ranked_outcomes = p_pre.sort_values(ascending=False).index.tolist()
        rank_map = {f"Top {idx+1}": outcome for idx, outcome in enumerate(ranked_outcomes[:3])}
        
        for rank_label, outcome in rank_map.items():
            p_anchor_val = p_pre[outcome]
            
            for k in horizons:
                t_post = t_anc + pd.Timedelta(minutes=k)
                if t_post in p_piv.index:
                    p_post_val = p_piv.loc[t_post, outcome]
                    dp_impact_cents = (p_post_val - p_anchor_val) * 100.0
                    
                    impact_records.append({
                        'Event Slug': event_slug,
                        'Anchor Time': t_anc,
                        'Rank': rank_label,
                        'Outcome': outcome,
                        'Horizon (min)': k,
                        'Pre Price': round(p_anchor_val, 4),
                        'Post Price': round(p_post_val, 4),
                        'Price Impact (¢)': round(dp_impact_cents, 2)
                    })
                    
    df_impacts = pd.DataFrame(impact_records)
    return df_impacts, len(anchor_times)

# Test engine on July 2026 FOMC event
df_imp_july, n_jumps_july = analyze_anchored_jump_impact('fed-interest-rates-july-2026')
print(f"=== JULY 2026 FOMC ANCHORED JUMP ANALYSIS ===")
print(f"Detected {n_jumps_july} anchor timestamps with >5% price jump.")
print(f"Generated {len(df_imp_july)} post-shock impact observations.")
print(df_imp_july.head(10).to_string(index=False))"""

add_code(code_cell_2)

# CELL 3: Boxplot Figure Function (SciencePlots Style)
code_cell_3 = r"""# ==========================================
# SCIENCEPLOTS BOXPLOT VISUALIZATION ENGINE
# ==========================================

def plot_anchored_jump_boxplots(df_impacts, event_title="FOMC Event", horizons=[1, 2, 5, 10, 15, 30, 60]):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=300, sharey=True)
    
    rank_labels = ['Top 1', 'Top 2', 'Top 3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, r_label in enumerate(rank_labels):
        ax = axes[idx]
        sub = df_impacts[df_impacts['Rank'] == r_label]
        
        data_by_h = [sub[sub['Horizon (min)'] == k]['Price Impact (¢)'].values for k in horizons]
        
        bp = ax.boxplot(data_by_h, patch_artist=True, tick_labels=[f"{k}m" for k in horizons])
        
        for patch in bp['boxes']:
            patch.set_facecolor(colors[idx])
            patch.set_alpha(0.65)
            
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(1.2)
            
        ax.axhline(0, color='black', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.set_title(f"{r_label} Outcome Impact Distribution", fontsize=9)
        ax.set_xlabel("Horizon $k$ (Minutes Post-Shock)", fontsize=8)
        if idx == 0:
            ax.set_ylabel("Price Impact $\Delta P$ (¢)", fontsize=8)
            
    plt.suptitle(f"Price Impact Distribution Following >5% Price Jump ({event_title})", fontsize=10, y=1.02)
    plt.tight_layout()
    
    out_filename = f"plots/jump_impact_boxplots_{event_title.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(out_filename, bbox_inches='tight')
    plt.show()
    print(f"Saved boxplot figure to '{out_filename}'.")

plot_anchored_jump_boxplots(df_imp_july, "July 2026 FOMC")"""

add_code(code_cell_3)

# CELL 4: Multi-Event Comparative Boxplot Engine
code_cell_4 = r"""# ==========================================
# MULTI-EVENT COMPARATIVE BOXPLOT ENGINE
# ==========================================

target_events = [
    'fed-interest-rates-july-2026',
    'fed-interest-rates-september-2026',
    'fed-decision-in-december',
    'fed-decision-in-january'
]

all_event_impacts = []

for ev in target_events:
    df_imp, n_jumps = analyze_anchored_jump_impact(ev)
    if df_imp is not None and len(df_imp) > 0:
        all_event_impacts.append(df_imp)

df_all_impacts = pd.concat(all_event_impacts, ignore_index=True)

print(f"=== COMBINED MULTI-EVENT ANCHORED JUMP ANALYSIS ({len(target_events)} EVENTS) ===")
print(f"Total Impact Observations across events: {len(df_all_impacts):,}")

plot_anchored_jump_boxplots(df_all_impacts, "Combined FOMC Events")"""

add_code(code_cell_4)

# CELL 5: Interactive Plotly Boxplot Dashboard
code_cell_5 = r"""# ==========================================
# INTERACTIVE PLOTLY BOXPLOT DASHBOARD
# ==========================================

def plot_plotly_jump_boxplots(df_impacts, event_title="Combined FOMC Events", horizons=[1, 2, 5, 10, 15, 30, 60]):
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=("<b>Top 1 Outcome Impact</b>", "<b>Top 2 Outcome Impact</b>", "<b>Top 3 Outcome Impact</b>")
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    rank_labels = ['Top 1', 'Top 2', 'Top 3']
    
    for idx, r_label in enumerate(rank_labels, start=1):
        sub = df_impacts[df_impacts['Rank'] == r_label]
        color = colors[idx - 1]
        
        for k in horizons:
            sub_k = sub[sub['Horizon (min)'] == k]
            fig.add_trace(
                go.Box(y=sub_k['Price Impact (¢)'], name=f"{k}m",
                       marker_color=color, boxmean=True, showlegend=False),
                row=1, col=idx
            )
            
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=idx)
        
    fig.update_layout(
        height=450,
        title_text=f"<b>Interactive Price Impact Distribution Following >5% Price Shock</b> ({event_title})",
        template="plotly_white",
        hovermode="y unified"
    )
    
    fig.update_yaxes(title_text="Price Impact ΔP (¢)", row=1, col=1)
    for c in range(1, 4):
        fig.update_xaxes(title_text="Horizon (Minutes Post-Shock)", row=1, col=c)
        
    return fig

fig_plotly = plot_plotly_jump_boxplots(df_all_impacts, "Combined FOMC Events")
print("Interactive Plotly Jump Impact Boxplot Dashboard Ready!")
fig_plotly.show()"""

add_code(code_cell_5)

# Save notebook
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 2}, f, indent=1)

print(f"Successfully built dedicated notebook '{nb_path}'!")
