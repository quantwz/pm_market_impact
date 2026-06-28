# Reader's Guide: Polymarket Market Impact Analysis

This guide provides a comprehensive breakdown of the terminology, mathematical definitions, plotting conventions, and economic interpretations used in the market impact study. This document is designed to serve as reference material for discussions with academic collaborators (e.g., at the Oxford Man Institute).

---

## 1. Key Terminology & Definitions

### A. YES-Normalised Price ($p_{\text{yes}}$)
Polymarket contracts trade as binary outcomes (YES or NO). The raw database records transaction prices relative to the specific token traded (`token0` or `token1`). To ensure comparability across all trades, we normalise all transaction prices to the **YES-token perspective**:
$$p_{\text{yes}} = \begin{cases} 
p_{\text{raw}} & \text{if YES contract traded (identified as token1)} \\
1 - p_{\text{raw}} & \text{if NO contract traded (identified as token2)}
\end{cases}$$
Thus, $p_{\text{yes}} \in [0, 1]$ directly represents the market's risk-neutral probability of the event occurring.

### B. Taker Sign ($D_k$)
Denotes the direction of the trade initiated by the market taker. It is signed relative to the demand for the YES contract:
* **$D_k = +1$ (Net YES Buy)**: Occurs when a taker buys the YES contract or sells the NO contract. This action represents demand that pushes the YES probability upwards.
* **$D_k = -1$ (Net YES Sell)**: Occurs when a taker sells the YES contract or buys the NO contract. This action represents demand that pushes the YES probability downwards.

### C. Trade Size Metrics
1. **USD Amount ($S$)**: The absolute transaction size in US Dollars:
   $$S = \text{usd\_amount}$$
2. **ADV-Adjusted Size ($S_{\text{ADV}}$)**: The trade size normalized by the Average Daily Volume of that specific market:
   $$S_{\text{ADV}} = \frac{\text{usd\_amount}}{\text{ADV}}$$
   where $\text{ADV}$ is computed as the time-series mean of daily traded volume across all active trading days for the contract. This measures the relative liquidity footprint of each order.

### D. Signed Price Impact Metrics
1. **Signed Probability Impact ($\Delta p_k$)**:
   $$\Delta p_k = D_k \cdot (p_{\text{yes}, k} - p_{\text{yes}, k-1})$$
   where $p_{\text{yes}, k}$ is the transaction price after trade $k$, and $p_{\text{yes}, k-1}$ is the price immediately prior to trade $k$. Multiplying by $D_k$ ensures that if the price moves in the direction of the taker's trade, the impact is positive.
2. **Signed Logit Impact ($\Delta x_k$)**:
   Because probability space is bounded in $[0, 1]$, price changes near 0 or 1 are mechanically constrained. To resolve this boundary effect, we map prices to the real line using the logit transformation:
   $$x = \text{logit}(p_{\text{yes}}) = \ln\left(\frac{p_{\text{yes}}}{1 - p_{\text{yes}}}\right)$$
   *(Note: $p_{\text{yes}}$ is clipped to $[10^{-5}, 1 - 10^{-5}]$ to prevent undefined values).*
   The signed logit impact is then:
   $$\Delta x_k = D_k \cdot (x_k - x_{k-1})$$
   This represents the information or order flow impact in log-odds space, which is linear in information content under classic risk-neutral probability updates.

---

## 2. Structure of the Plots

Each figure is structured as a **$2 \times 2$ panel grid** to evaluate the interaction of both trade size definitions (USD vs. ADV fraction) with both price impact definitions (Probability vs. Logit space):

| Panel | Y-Axis: Price Impact Measure | X-Axis: Trade Size Measure |
| :---: | :--- | :--- |
| **(a)** | Signed Probability Impact ($\Delta p$) | Trade Size $S$ (USD, log-scale) |
| **(b)** | Signed Probability Impact ($\Delta p$) | Trade Size $S_{\mathrm{ADV}}$ (fraction of ADV, log-scale) |
| **(c)** | Signed Logit Impact ($\Delta x$) | Trade Size $S$ (USD, log-scale) |
| **(d)** | Signed Logit Impact ($\Delta x$) | Trade Size $S_{\mathrm{ADV}}$ (fraction of ADV, log-scale) |

### Plot Conventions
* **X-Axis (Logarithmic scale)**: Handled dynamically with custom formatting. USD ticks are represented as standard denominations ($\$1$, $\$100$, $\$1\text{K}$, etc.), and ADV ticks are shown in exponential notation ($10^{-6}$, $10^{-4}$, etc.).
* **Solid/Dashed Lines**: Connect the mean price impact computed inside each quantile bin.
* **Shaded Bands**: Represent the **$95\%$ Confidence Interval** ($\text{Mean} \pm 2 \cdot \text{SE}$). The Standard Error (SE) is computed as $\text{SE} = \frac{\sigma}{\sqrt{N_{\text{bin}}}}$. Because our dataset is massive ($\sim 31.5\text{M}$ trades), the confidence bands are narrow and visually highlight the robustness of the mean estimations.

---

## 3. How to Interpret the Figures

### Figure 1: Impact vs. Trade Size by Category
This plot groups markets into three broad thematic categories: **Macro**, **Politics**, and **Geopolitics**.
* **Microstructure Interpretation**: Politics and Geopolitics exhibit significantly higher price impact coefficients (steeper curves) than Macro. This indicates that Politics/Geopolitics contracts are structurally more "illiquid" or subject to higher information asymmetry.
* **The Concavity / Decline in Large Sizes**: On the USD scale, price impact rises up to a certain point (typically $\$10$ to $\$30$) and then flattens or falls slightly. This is a classic pattern in financial markets:
  1. Larger trades are often executed by institutional/sophisticated actors who use execution algorithms to minimize footprint (order slicing).
  2. Large orders are typically directed to markets that already have extremely thick order books (high ADV), while small markets see smaller maximum trades.

### Figure 2: Conditional on Prior Probability Level $p_{k-1}$
This plot splits trades based on the prior contract price: $p_{\text{yes}, k-1} \in [0, 0.2)$, $[0.2, 0.4)$, $[0.4, 0.6)$, $[0.6, 0.8)$, or $[0.8, 1.0]$.
* **Microstructure Interpretation (Panels a & b)**: Raw probability impact ($\Delta p$) peaks for markets in the $[0.4, 0.6)$ bucket (highly uncertain 50/50 state). For extreme values ($p < 0.2$ or $p > 0.8$), $\Delta p$ is compressed. This is due to the mathematical constraint that price cannot exceed 1 or fall below 0.
* **Microstructure Interpretation (Panels c & d)**: When mapped to the logit space ($\Delta x$), the pattern flips. Extreme probability contracts show *greater* logit impact per trade than 50/50 contracts. This shows that near the boundaries, order flow induces major changes in information/log-odds space, reflecting the leverage-like behavior of deep out-of-the-money or in-the-money options.

### Figure 3: Conditional on Time to Expiry $\tau$
This plot groups trades by the number of days remaining until contract expiration: $\tau \le 7$ days, $7 < \tau \le 30$ days, $30 < \tau \le 90$ days, and $\tau > 90$ days.
* **Microstructure Interpretation**: Price impact is inversely related to time to expiry. Contracts with $\tau \le 7$ days (dark red) show the highest price sensitivity, whereas contracts with $\tau > 90$ days (cyan) have the lowest.
* **Theory Linkage**: As expiration approaches:
  1. Information asymmetry increases (the outcome becomes imminent, attracting informed traders).
  2. Inventory holding costs rise for liquidity providers (higher risk of holding toxic positions that cannot be hedged).
  3. Liquidity providers widen their bid-ask spreads, resulting in higher measured price impact.

### Figures 4, 5, 6, & 7: Zoom-In Microstructure Analysis
These plots zoom into four specific market subsets to analyze the full distribution of individual trades (using a downsampled background scatter plot of 30,000 points) overlaid with log-spaced boxplots and the binned mean price impact line.

1. **Figure 4 (Macro Markets, Full Sample)**: Isolates macro-themed contracts to observe their specific trade distribution.
2. **Figure 5 (Extreme Low Prior Probability, $p_{k-1} < 0.2$)**: Isolates trades occurring when the YES contract is priced below $0.2$.
3. **Figure 6 (Extreme High Prior Probability, $p_{k-1} \geq 0.8$)**: Isolates trades occurring when the YES contract is priced above $0.8$.
4. **Figure 7 (Close to Expiry, $\tau \leq 7$ Days)**: Isolates trades occurring in the final week before contract resolution.

#### Microstructure Interpretations:
* **The Skewness Effect (Mean-Median Divergence)**: Across all four cases, the median price impact is exactly zero for all trade size classes (the boxplot medians remain completely flat at 0). However, the **mean price impact** is positive and highly statistically significant. This indicates that the price impact distribution is heavily right-skewed; a typical trade has zero impact (matching existing order book liquidity), but a small proportion of trades trigger non-zero, positive price impacts.
* **Tail Dispersion Scaling**: Although the median remains at 0, the upper whiskers (representing the 90th percentile of the trade distribution) and the 75th percentile box edges scale upwards with larger trade sizes. This demonstrates that larger transaction sizes increase the likelihood and magnitude of a price movement, even if the typical trade still experiences zero impact.
* **Boundary Effects in Option Space (Figures 5 & 6)**: For extreme probabilities, the raw price impact $\Delta p$ is compressed due to the mathematical limits at 0 and 1. However, when viewed in logit-space ($\Delta x$), the dispersion (whiskers and box heights) is significantly wider than in the general sample, revealing high information-odds volatility near contract resolution boundaries.
* **Adverse Selection & Spread Widening (Figure 7)**: Trades executed in the final week ($\tau \leq 7$ days) show the highest overall variance and whiskers in price impact. This is consistent with market makers widening their spreads to protect against informed trading (adverse selection) near the resolution date.

### Figure 8: Trade Size in USD vs. Normalized Trade Size (Fraction of ADV)
This figure displays the binned geometric mean trade size in USD ($S$) on the x-axis against the binned geometric mean normalized trade size ($S_{\mathrm{ADV}}$) on the y-axis, plotted on a log-log scale.

#### Subplots:
1. **Panel (a) By Market Category**: Politics, Geopolitics, Macro.
2. **Panel (b) By Prior Probability**: Binned by $p_{k-1}$ into 5 buckets.
3. **Panel (c) By Time to Expiry**: Binned by $\tau$ into 4 buckets.

#### Microstructure Interpretations:
* **Typical ADV Mapping (Vertical Shifts)**: Since $S_{\mathrm{ADV}} = S / \text{ADV}$, the vertical shift of each line directly reflects the baseline typical ADV of that group. A higher line represents a higher fraction of ADV for a given USD trade size, meaning **lower typical ADV** (less liquid market segment).
* **Category Liquidity Profiles**: Panel (a) reveals that Politics lies highest, meaning Politics markets have the lowest typical ADV ($\approx \$65.6\text{K}$) in our sample. Macro lies lowest, indicating it is the most liquid category with a much higher typical ADV ($\approx \$531\text{K}$). This directly explains why the raw price impact of a $\$100$ trade is much higher in Politics than in Macro (as $\$100$ represents a much larger fraction of Politics ADV than Macro ADV).
* **State & Expiry Liquidity Mapping**: Panels (b) and (c) show that liquidity (ADV) is higher for near-expiry contracts ($\tau \leq 7$ days) and high-probability contracts ($p_{k-1} \geq 0.8$), while long-term contracts ($\tau > 90$ days) and low-probability contracts ($p_{k-1} < 0.2$) have lower baseline ADV.


