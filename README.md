# Polymarket Market Microstructure & Price Impact

This repository contains a quantitative pipeline for analyzing market impact and transaction-level microstructure on Polymarket. Using a dataset from https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data, we analyze the market impact and price dynamics of trades of Macro, Geopolitics, and Politics categories on Polymarket (/datat/selected_categorized_market_list_v2.csv). The dataset includes **trades.parquet** with 31 million trades and **markets.parquet** with market metadata.


---

### 📊 Core Microstructure Curves
* **Figure 1: Mean Price Impact vs. Trade Size by Category**
  * Plots mean price impact (probability change $\Delta p$ and logit change $\Delta x$) vs. absolute USD size ($S$) and ADV-normalized size ($S_{\mathrm{ADV}}$) across **Macro**, **Politics**, and **Geopolitics** categories.

![Figure 1](/plots/fig1_impact_by_category.png)

* **Figure 2: Mean Impact Conditional on Prior Probability ($p_{k-1}$)**
  * Bins price impact across 5 prior probability buckets ($[0, 0.2)$, $[0.2, 0.4)$, etc.).
  * *Insight*: Shows raw impact $\Delta p$ peaks in highly uncertain (50/50) markets, while logit impact $\Delta x$ is highest near the boundaries (deep out-of-the-money or in-the-money options).

![Figure 2](/plots/fig2_impact_by_prob_level.png)

* **Figure 3: Mean Impact Conditional on Time to Expiry ($\tau$)**
  * Groups contracts by horizon remaining ($\le 7$d, $7\text{--}30$d, $30\text{--}90$d, $>90$d).

![Figure 3](/plots/fig3_impact_by_expiry.png)

### 📊 Filtered Microstructure Curves (Trade Size $\ge$ $100)
To focus on more substantial market orders, we provide a parallel set of analysis figures filtering out trades smaller than $100. This removes retail noise (which accounts for ~88% of all trades, leaving ~3.8M trades for analysis).

* **Filtered Figure 1: Mean Price Impact by Category ($\ge$ $100)**
  * ![Filtered Figure 1](/plots/fig1_impact_by_category_filtered.png)

* **Filtered Figure 2: Mean Impact Conditional on Prior Probability ($\ge$ $100)**
  * ![Filtered Figure 2](/plots/fig2_impact_by_prob_level_filtered.png)

* **Filtered Figure 3: Mean Impact Conditional on Time to Expiry ($\ge$ $100)**
  * ![Filtered Figure 3](/plots/fig3_impact_by_expiry_filtered.png)

### Liquidity Mapping
* **Figure 8: Trade Size USD ($S$) vs. ADV Ratio ($S_{\mathrm{ADV}}$)**
  * A log-log panel plot showing how absolute trade sizes map to ADV-fractions. Since $S_{\mathrm{ADV}} = S / \text{ADV}$, the vertical shift of each line directly visualizes the baseline average daily volume ($\text{ADV}$) for that segment, explaining why absolute trades have a much smaller relative footprint in Macro than in Politics.

![Figure 8](/plots/fig8_size_relationship.png)


---

## 🚀 Quick Start Guide

### 1. Setup Virtual Environment
Ensure you have Python 3.10+ installed. Initialize your environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r market_impact/requirements.txt
```

### 2. Configure Raw Data Path
To run the full computation pipeline, you must have access to the raw 29GB `trades.parquet` file. Set the environment variable `TRADES_PARQUET_PATH` to point to it:
```bash
# Windows (PowerShell)
$env:TRADES_PARQUET_PATH = "C:/path/to/large/trades.parquet"

# Linux / macOS
export TRADES_PARQUET_PATH="/path/to/large/trades.parquet"
```

### 3. Run Pipeline
* **Full Pipeline (Ingestion + Binning + Plotting)**:
  ```bash
  python market_impact/run_analysis.py
  ```
* **Plotting Only (Skip Ingestion, runs on pre-computed aggregated parquets)**:
  ```bash
  python market_impact/run_analysis.py --plot
  ```

