"""
market_impact/compute_impact.py
───────────────────────────────
Polymarket Market Impact Analysis – Data Pipeline

Loads raw trade data from trades.parquet, normalises prices to the YES-token
perspective, assigns taker signs, computes signed price impacts (probability
and logit), calculates per-market ADV, and bins trades by category, prior
probability level, and time-to-expiry.

Saves aggregated summary tables to market_impact/data/ for downstream plotting.

Outputs
-------
data/cat_impacts.parquet    – aggregated by market category
data/prob_impacts.parquet   – aggregated by prior probability bin
data/expiry_impacts.parquet – aggregated by time-to-expiry bin
data/impact_trade_level.parquet – lean trade-level dataset

Author: Market Impact Research Pipeline
"""

import os
import gc
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
OUT_DIR      = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─── Bin labels (used consistently across compute & plot) ─────────────────────
PROB_LABELS   = ["p_lo", "p_midlo", "p_mid", "p_midhi", "p_hi"]
EXPIRY_LABELS = ["tte_le7d", "tte_7_30d", "tte_30_90d", "tte_gt90d"]

N_BINS_MAIN = 20   # quantile bins for main (by-category) aggregation
N_BINS_COND = 14   # quantile bins for conditional aggregations (smaller samples)
MIN_OBS_PER_BIN = 5


# ─── Helpers ──────────────────────────────────────────────────────────────────
def logit_safe(p: np.ndarray) -> np.ndarray:
    """Clip-safe logit transform."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def geom_mid(interval) -> float:
    """Geometric midpoint of a pandas Interval (for log-scale x-axis)."""
    l, r = interval.left, interval.right
    if l <= 0:
        l = r * 0.01          # small fallback for first bin
    return np.sqrt(l * r)


# ─── Step 1: Load & prepare trade data ───────────────────────────────────────
def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("━━ Step 1 ─ Loading market metadata ━━")
    meta_path = os.path.join(OUT_DIR, "selected_categorized_market_list_v2.csv")
    if not os.path.exists(meta_path):
        # Fallback to general DATA_DIR if run in original repo structure without local copy
        meta_path = os.path.join(DATA_DIR, "selected_categorized_market_list_v2.csv")
    meta_raw = pd.read_csv(meta_path)
    meta_raw["id"] = meta_raw["id"].astype(str)
    meta = (meta_raw
            .drop_duplicates(subset="id")
            [["id", "category", "end_date", "volume"]]
            .copy())
    market_ids = meta["id"].tolist()
    log.info(f"   {len(market_ids)} unique markets  |  "
         f"categories: {meta['category'].value_counts().to_dict()}")

    log.info("━━ Step 2 ─ Loading trades.parquet (filtered) ━━")
    trades_path = os.environ.get("TRADES_PARQUET_PATH", os.path.join(DATA_DIR, "trades.parquet"))
    dataset = ds.dataset(trades_path)
    expr    = pc.field("market_id").isin(market_ids)
    cols    = ["timestamp", "market_id", "price", "usd_amount",
               "nonusdc_side", "taker_direction", "log_index"]
    df = dataset.to_table(filter=expr, columns=cols).to_pandas()
    log.info(f"   {len(df):,} rows  |  {df.memory_usage().sum()/1e9:.2f} GB")

    log.info("━━ Step 3 ─ Normalising to YES perspective & computing taker signs ━━")
    is_tok1    = df["nonusdc_side"].str.contains("token1", case=False, na=False)
    taker_buy  = df["taker_direction"] == "BUY"

    # YES-normalised price: p_yes ∈ [0,1] always represents P(YES)
    df["price_yes"] = np.where(is_tok1, df["price"].values, 1.0 - df["price"].values)

    # Taker sign: +1 = net buyer of YES, -1 = net seller of YES
    #   token1 (YES): BUY→+1, SELL→-1
    #   token2 (NO):  BUY→-1 (buying NO ≡ selling YES), SELL→+1
    df["taker_sign"] = np.where(
        is_tok1,
        np.where(taker_buy, np.int8(1), np.int8(-1)),
        np.where(taker_buy, np.int8(-1), np.int8(1)),
    ).astype(np.int8)

    # Drop rows with missing direction
    df.dropna(subset=["taker_direction"], inplace=True)
    log.info(f"   After cleaning: {len(df):,} rows")

    log.info("━━ Step 4 ─ Sorting by (market, timestamp, log_index) ━━")
    df.sort_values(["market_id", "timestamp", "log_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info("━━ Step 5 ─ Computing lagged price & signed price impacts ━━")
    # Lagged YES price (previous trade within the same market)
    df["price_prev"] = (df.groupby("market_id", sort=False)["price_yes"]
                          .shift(1))
    # Drop first observation per market (no previous trade)
    df.dropna(subset=["price_prev"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Probability impact:  ΔP_k = D_k · (p_k − p_{k-1})
    df["dp"] = df["taker_sign"].values * (df["price_yes"].values
                                          - df["price_prev"].values)

    # Logit impact:  Δx_k = D_k · (x_k − x_{k-1})
    df["dx"] = df["taker_sign"].values * (logit_safe(df["price_yes"].values)
                                          - logit_safe(df["price_prev"].values))

    log.info("━━ Step 6 ─ Computing per-market ADV ━━")
    df["date"]  = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.date
    daily_vol   = (df.groupby(["market_id", "date"])["usd_amount"]
                     .sum().reset_index()
                     .rename(columns={"usd_amount": "daily_vol"}))
    adv_series  = (daily_vol.groupby("market_id")["daily_vol"]
                             .mean().rename("adv"))
    df = df.join(adv_series, on="market_id")
    df["adv_ratio"] = df["usd_amount"].values / df["adv"].clip(lower=1.0).values
    del daily_vol, adv_series; gc.collect()

    log.info("━━ Step 7 ─ Merging category metadata & computing TTE ━━")
    df = df.merge(meta, left_on="market_id", right_on="id", how="left")

    df["end_dt"]   = pd.to_datetime(df["end_date"], utc=True)
    df["trade_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["tte_days"] = ((df["end_dt"] - df["trade_dt"])
                      .dt.total_seconds() / 86400.0).clip(lower=0.0)

    log.info("━━ Step 8 ─ Creating probability & TTE bins ━━")
    # Prior probability bins: 5 equal-width buckets over [0, 1]
    df["prob_bin"] = pd.cut(
        df["price_prev"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=PROB_LABELS,
        include_lowest=True,
        right=False,
    )

    # Time-to-expiry bins: near / short / medium / long
    df["expiry_bin"] = pd.cut(
        df["tte_days"],
        bins=[-np.inf, 7.0, 30.0, 90.0, np.inf],
        labels=EXPIRY_LABELS,
    )

    # Summary stats
    log.info(f"   Trades by category:\n{df['category'].value_counts().to_string()}")
    log.info(f"   Trades by prob_bin:\n{df['prob_bin'].value_counts().to_string()}")
    log.info(f"   Trades by expiry_bin:\n{df['expiry_bin'].value_counts().to_string()}")
    log.info(f"   Final shape: {df.shape}")
    return df, meta


# ─── Step 2: Aggregate ────────────────────────────────────────────────────────
def _quantile_aggregate(
    df: pd.DataFrame,
    size_col: str,
    impact_col: str,
    group_col: str,
    n_bins: int,
) -> pd.DataFrame:
    """
    For each unique value of group_col, bin trades by size_col into n_bins
    quantile bins and compute percentile statistics of impact_col per bin.

    Returns a tidy DataFrame with columns:
      group | size_type | impact_type | size_bin_center |
      n | q10 | q25 | q50 | q75 | q90 | mean | se
    """
    # Keep only positive trade sizes
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

        # Quantile bin edges from the within-group size distribution
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
    """Run all 12 aggregation combinations and pack into 3 summary tables."""
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
                log.info(f"   Aggregating  cond={gkey}  size={sl}  impact={il} …")
                tbl = _quantile_aggregate(df, sc, ic, gcol, n_bins)
                if not tbl.empty:
                    tbl["size_type"]   = sl
                    tbl["impact_type"] = il
                    results[gkey].append(tbl)

    return {k: pd.concat(v, ignore_index=True) for k, v in results.items() if v}


# ─── Step 3: Save ─────────────────────────────────────────────────────────────
def save_trade_level(df: pd.DataFrame) -> None:
    keep_cols = [
        "timestamp", "market_id", "category",
        "price_yes", "price_prev", "taker_sign",
        "usd_amount", "adv_ratio", "adv",
        "dp", "dx",
        "prob_bin", "expiry_bin", "tte_days",
    ]
    out_path = os.path.join(OUT_DIR, "impact_trade_level.parquet")
    df[keep_cols].to_parquet(out_path, index=False)
    log.info(f"   Saved {out_path}  ({len(df):,} rows)")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("═══ Market Impact Analysis: Data Pipeline ═══")

    df, meta = load_and_prepare()

    log.info("━━ Step 9 ─ Saving trade-level dataset ━━")
    save_trade_level(df)

    log.info("━━ Step 10 ─ Aggregating (all conditions × sizes × impacts) ━━")
    tables = aggregate_all(df)

    log.info("━━ Step 11 ─ Writing aggregation outputs ━━")
    for name, tbl in tables.items():
        path = os.path.join(OUT_DIR, f"{name}_impacts.parquet")
        tbl.to_parquet(path, index=False)
        log.info(f"   {name}_impacts.parquet  ->  {len(tbl):,} rows")

    log.info("═══ Pipeline complete ═══")


if __name__ == "__main__":
    main()
