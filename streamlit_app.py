"""
Multi-Outcome Prediction Cross-Elasticity
========================================
Streamlit app for analysing cross-outcome price elasticity around central-bank
decision shocks (Fed, BOJ, ECB) on Polymarket.
"""

import io
import duckdb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Outcome Prediction Cross-Elasticity",
    page_icon="📈",
    layout="wide",
)

# ── Style ─────────────────────────────────────────────────────────────────────
try:
    plt.style.use(["science", "no-latex"])
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH = Path("data/polymarket_fomc_trade.db")
CANONICAL = {
    "No Change",
    "Increase 25 bps",
    "Increase 50+ bps",
    "Decrease 25 bps",
    "Decrease 50+ bps",
}
MIN_OUTCOMES = 3
MIN_TRADES = 2000

# ── Central-bank slug prefixes ────────────────────────────────────────────────
CB_PREFIXES = {
    "Fed":  ("fed-", "federal-reserve-", "fomc-"),
    "BOJ":  ("boj-", "bank-of-japan-", "boj_", "boj-"),
    "ECB":  ("ecb-", "european-central-bank-", "ecb_"),
}

def classify_event(slug: str) -> str:
    """Return the central-bank group ('Fed', 'BOJ', 'ECB') for a slug, or 'Other'."""
    slug_lower = slug.lower()
    for bank, prefixes in CB_PREFIXES.items():
        if any(slug_lower.startswith(p) for p in prefixes):
            return bank
    return "Other"

COLOR_MOD = "#2196F3"   # Moderate
COLOR_LRG = "#E53935"   # Large / Extreme

@st.cache_data(show_spinner="Loading trade data…")
def load_data(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        "SELECT event_slug, outcome_label, timestamp_utc, price "
        "FROM fomc_trade_ticks"
    ).df()
    con.close()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


@st.cache_data(show_spinner="Loading single event ticks…")
def load_event_time_series(db_path: str, event_slug: str) -> pd.DataFrame:
    """Fetch trade ticks for a single event from DuckDB."""
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        "SELECT outcome_label, timestamp_utc, price, "
        "COALESCE(usd_amount, price) as usd_amount, "
        "COALESCE(token_amount, 1.0) as token_amount "
        "FROM fomc_trade_ticks "
        "WHERE event_slug = ? "
        "ORDER BY timestamp_utc",
        [event_slug]
    ).df()
    con.close()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


@st.cache_data(show_spinner="Qualifying events…")
def qualify_events(df: pd.DataFrame) -> tuple:
    """Apply canonical filter and event re-qualification."""
    df = df[df["outcome_label"].isin(CANONICAL)].copy()
    grp = df.groupby("event_slug")
    n_out = grp["outcome_label"].nunique()
    n_trd = grp.size()
    valid = n_out[n_out >= MIN_OUTCOMES].index.intersection(
        n_trd[n_trd >= MIN_TRADES].index
    )
    df = df[df["event_slug"].isin(valid)].copy()
    events = sorted(df["event_slug"].unique().tolist())
    return df, events


# Pre-computed CSV paths (fast path — avoids full recompute)
CSV_ELAS  = Path("data/fomc_general_elasticity_long.csv")
CSV_VOL   = Path("data/fomc_volume_elasticity_long.csv")


@st.cache_data(show_spinner="Loading pre-computed elasticity data...")
def load_csv_data(_elas_mtime: float, _vol_mtime: float) -> tuple:
    """Load pre-computed elasticity CSVs. Cache busts when files change on disk."""
    df_elas = pd.read_csv(CSV_ELAS, parse_dates=["Timestamp"])
    df_vol  = pd.read_csv(CSV_VOL,  parse_dates=["Timestamp"])
    return df_elas, df_vol


def apply_filters(
    df_elas: pd.DataFrame,
    df_vol: pd.DataFrame,
    shock_thresh: float,
    bucket_boundary: float,
    horizon_max: int,
    selected_events: tuple,
    dte_filter: str,
    err_ci: int,
    min_shock_vol: float = 1000.0,
) -> tuple:
    """Filter and re-bucket pre-computed data in-memory. Returns (df_lo, df_pp, df_vol)."""

    # 1. Shock threshold filter (Abs_Jump_Pct column stores raw % jump)
    if "Abs_Jump_Pct" in df_elas.columns:
        df_elas = df_elas[df_elas["Abs_Jump_Pct"] >= shock_thresh]
    if "Abs_Jump_Pct" in df_vol.columns:
        df_vol = df_vol[df_vol["Abs_Jump_Pct"] >= shock_thresh]

    # 1b. Min shock trade volume filter ($)
    if "Shock_Volume" in df_elas.columns:
        df_elas = df_elas[df_elas["Shock_Volume"] >= min_shock_vol]
    if "Shock_Volume" in df_vol.columns:
        df_vol = df_vol[df_vol["Shock_Volume"] >= min_shock_vol]

    # 2. Re-bucket tranches based on bucket_boundary
    def rebucket(df):
        df = df.copy()
        if "Abs_Jump_Pct" in df.columns:
            df["bucket"] = np.where(
                df["Abs_Jump_Pct"] < bucket_boundary,
                "Moderate",
                "Large/Extreme",
            )
        elif "Jump_Tranche" in df.columns:
            tranche_map = {
                'Moderate (2.5 - 5.0c)': 'Moderate',
                'Large / Extreme (> 5.0c)': 'Large/Extreme'
            }
            df["bucket"] = df["Jump_Tranche"].map(tranche_map).fillna("Moderate")
        else:
            df["bucket"] = "Moderate"
        return df

    df_elas = rebucket(df_elas)
    df_vol  = rebucket(df_vol)

    # 3. Horizon filter
    df_elas = df_elas[df_elas["Horizon_Min"] <= horizon_max]
    df_vol  = df_vol[df_vol["Horizon_Min"] <= horizon_max]

    # 4. Event filter
    if selected_events:
        df_elas = df_elas[df_elas["Event"].isin(selected_events)]
        df_vol  = df_vol[df_vol["Event"].isin(selected_events)]

    # 5. DTE filter (days to expiry)
    dte_days_map = {"Full Period": None, "30d": 30, "14d": 14, "7d": 7, "3d": 3, "1d": 1}
    dte_days = dte_days_map.get(dte_filter)
    if dte_days is not None and "Timestamp" in df_elas.columns:
        # Compute settlement date per event (max timestamp)
        settle_elas = df_elas.groupby("Event")["Timestamp"].max().rename("settle")
        settle_vol  = df_vol.groupby("Event")["Timestamp"].max().rename("settle")
        df_elas = df_elas.join(settle_elas, on="Event")
        df_vol  = df_vol.join(settle_vol,  on="Event")
        df_elas = df_elas[
            (df_elas["settle"] - df_elas["Timestamp"]).dt.total_seconds() / 86400 <= dte_days
        ].drop(columns=["settle"])
        df_vol = df_vol[
            (df_vol["settle"] - df_vol["Timestamp"]).dt.total_seconds() / 86400 <= dte_days
        ].drop(columns=["settle"])

    # Rename columns to match plot helper expectations
    df_elas = df_elas.rename(columns={"Horizon_Min": "horizon", "Driver_Target": "pair"})
    df_vol  = df_vol.rename(columns={"Horizon_Min": "horizon", "Driver_Target": "pair"})

    df_lo = df_elas[df_elas["Metric"] == "Log-Odds Elasticity"].copy()
    df_pp = df_elas[df_elas["Metric"] == "Probability Price Elasticity"].copy()

    return df_lo, df_pp, df_vol



# ── Plot helpers ──────────────────────────────────────────────────────────────
# Pair labels must match Driver_Target values in the CSV
PAIR_GRID = [
    ("Driver Top1 -> Target Top2", "Driver Top2 -> Target Top1", "Driver Top3 -> Target Top2"),
    ("Driver Top1 -> Target Top3", "Driver Top2 -> Target Top3", "Driver Top3 -> Target Top1"),
]

PAIR_TITLES = {
    "Driver Top1 -> Target Top2": "Driver: Top1\nTarget: Top2",
    "Driver Top2 -> Target Top1": "Driver: Top2\nTarget: Top1",
    "Driver Top3 -> Target Top2": "Driver: Top3\nTarget: Top2",
    "Driver Top1 -> Target Top3": "Driver: Top1\nTarget: Top3",
    "Driver Top2 -> Target Top3": "Driver: Top2\nTarget: Top3",
    "Driver Top3 -> Target Top1": "Driver: Top3\nTarget: Top1",
}

def _plot_elasticity(
    result: pd.DataFrame,
    metric: str,
    title: str,
    ref_val: float,
    err_ci: int,
    show_ref_lines: bool,
    y_label: str,
    shock_thresh: float,
    bucket_boundary: float,
    dte_filter: str,
    horizon_max: int,
    cb_scope: str = "All Central Banks",
    min_shock_vol: float = 1000.0,
    share_y: bool = True,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=150)

    subtitle = f"{title} [{cb_scope}]\n(Min Shock: >= {shock_thresh}%  |  Min Vol: >= ${min_shock_vol:,.0f}  |  Bucket Cutoff: {bucket_boundary}%  |  DTE: {dte_filter}  |  Max Horizon: {horizon_max} min)"
    fig.suptitle(subtitle, fontsize=11, fontweight="bold", y=0.99)

    palette = {"Moderate": COLOR_MOD, "Large/Extreme": COLOR_LRG}

    for row_i, row_pairs in enumerate(PAIR_GRID):
        for col_i, pair in enumerate(row_pairs):
            ax = axes[row_i][col_i]
            sub = result[result["pair"] == pair] if not result.empty else pd.DataFrame()
            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_title(PAIR_TITLES.get(pair, pair), fontsize=8)
                ax.set_xlim(1, horizon_max)
                continue

            sns.lineplot(
                data=sub,
                x="horizon",
                y=metric,
                hue="bucket",
                palette=palette,
                hue_order=["Moderate", "Large/Extreme"],
                err_style="band",
                errorbar=("ci", err_ci),
                ax=ax,
            )
            if show_ref_lines:
                ax.axhline(ref_val, ls="--", color="black", lw=0.8, alpha=0.7)
                ax.axhline(0.0, ls=":", color="gray", lw=0.6, alpha=0.6)
            ax.set_title(PAIR_TITLES.get(pair, pair), fontsize=8)
            ax.set_xlabel("Horizon (min)", fontsize=8)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlim(1, horizon_max)
            ax.tick_params(labelsize=7, labelleft=True)
            lgd = ax.get_legend()
            if lgd:
                lgd.remove()

    if share_y:
        # Align Y-axis across Columns 0 & 1 (Top1 and Top2 driver pairs)
        col01_axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
        col01_ymins = [ax.get_ylim()[0] for ax in col01_axes if ax.has_data()]
        col01_ymaxs = [ax.get_ylim()[1] for ax in col01_axes if ax.has_data()]
        if col01_ymins and col01_ymaxs:
            y_min_01 = min(col01_ymins)
            y_max_01 = max(col01_ymaxs)
            for ax in col01_axes:
                ax.set_ylim(y_min_01, y_max_01)

        # Align Y-axis across Column 2 (Top3 driver pairs) independently
        col2_axes = [axes[0][2], axes[1][2]]
        col2_ymins = [ax.get_ylim()[0] for ax in col2_axes if ax.has_data()]
        col2_ymaxs = [ax.get_ylim()[1] for ax in col2_axes if ax.has_data()]
        if col2_ymins and col2_ymaxs:
            y_min_2 = min(col2_ymins)
            y_max_2 = max(col2_ymaxs)
            for ax in col2_axes:
                ax.set_ylim(y_min_2, y_max_2)

    # Dynamic unified legend including exact shock thresholds
    label_mod = f"Moderate ({shock_thresh}%-{bucket_boundary}%)"
    label_lrg = f"Large/Extreme (>{bucket_boundary}%)"
    handles = [
        plt.Line2D([0], [0], color=COLOR_MOD, lw=2, label=label_mod),
        plt.Line2D([0], [0], color=COLOR_LRG, lw=2, label=label_lrg),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=2, fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _plot_single_event_timeseries(
    df_ticks: pd.DataFrame,
    event_slug: str,
    resample_freq: str = "5min",
    canonical_only: bool = True,
    metric_vol: str = "USD Volume",
) -> plt.Figure:
    """Plot 3-panel time series (Probability, Logit, Volume) for a single event."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, dpi=150)
    fig.suptitle(f"Single Event Time Series: {event_slug}", fontsize=12, fontweight="bold", y=0.99)

    df_filtered = df_ticks.copy()
    if canonical_only:
        df_filtered = df_filtered[df_filtered["outcome_label"].isin(CANONICAL)]

    outcomes = sorted(df_filtered["outcome_label"].unique().tolist())
    if not outcomes:
        ax1.text(0.5, 0.5, "No outcome data for event", ha="center", va="center")
        return fig

    # Color palette across outcomes
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(outcomes), 10)))
    color_map = {out: colors[i % len(colors)] for i, out in enumerate(outcomes)}

    freq_pandas = {
        "1min": "1min",
        "5min": "5min",
        "15min": "15min",
        "1hour": "1h",
        "Raw Ticks": None,
    }.get(resample_freq, "5min")

    vol_col = "usd_amount" if metric_vol == "USD Volume" else "token_amount"

    for out in outcomes:
        sub = df_filtered[df_filtered["outcome_label"] == out]
        if sub.empty:
            continue
        c = color_map[out]

        if freq_pandas is not None:
            s = sub.set_index("timestamp_utc")
            p_series = s["price"].resample(freq_pandas).last().ffill().dropna()
            if metric_vol == "Trade Count":
                v_series = s["price"].resample(freq_pandas).count().fillna(0)
            else:
                v_series = s[vol_col].resample(freq_pandas).sum().fillna(0)
            t_index = p_series.index
        else:
            t_index = sub["timestamp_utc"]
            p_series = sub["price"]
            v_series = sub[vol_col]

        # 1. Probability Price
        ax1.plot(t_index, p_series, label=out, color=c, lw=1.2, alpha=0.85)

        # 2. Logit Price
        p_clip = np.clip(p_series, 1e-4, 1 - 1e-4)
        logit_series = np.log(p_clip / (1 - p_clip))
        ax2.plot(t_index, logit_series, label=out, color=c, lw=1.2, alpha=0.85)

        # 3. Volume
        if freq_pandas is not None:
            ax3.plot(t_index, v_series, label=out, color=c, lw=1.1, alpha=0.8)
        else:
            ax3.scatter(t_index, v_series, label=out, color=c, s=6, alpha=0.5)

    ax1.set_ylabel("Probability Price P", fontsize=9)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=True)

    ax2.set_ylabel("Logit Price log(P / (1-P))", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    vol_label = f"Volume ({metric_vol} / {resample_freq})" if freq_pandas else f"Volume ({metric_vol} per tick)"
    ax3.set_ylabel(vol_label, fontsize=9)
    ax3.set_xlabel("Timestamp (UTC)", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout(rect=[0, 0, 0.86, 0.97])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📈 Multi-Outcome Prediction Cross-Elasticity")
    st.markdown(
        "_Cross-outcome price elasticity around central-bank decision shocks · Polymarket data_"
    )

    # ── Load pre-computed CSVs (fast) & qualify events ───────────────────────
    if not CSV_ELAS.exists() or not CSV_VOL.exists():
        st.warning(
            "Pre-computed CSVs not found. Run `plot_4col_2row_elasticity_general.py` "
            "and `plot_2x3_vol_elasticity.py` first."
        )
        st.stop()

    df_elas_raw, df_vol_raw = load_csv_data(CSV_ELAS.stat().st_mtime, CSV_VOL.stat().st_mtime)
    all_events = sorted(df_elas_raw["Event"].unique().tolist())
    n_events   = len(all_events)

    # Build per-bank event lists (used by sidebar widget)
    event_by_bank: dict[str, list[str]] = {"All": all_events, "Fed": [], "BOJ": [], "ECB": [], "Other": []}
    for ev in all_events:
        bank = classify_event(ev)
        event_by_bank.setdefault(bank, []).append(ev)
    # Only show groups that have at least one event
    available_groups = [g for g in ["All", "Fed", "BOJ", "ECB"] if event_by_bank.get(g)]

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")

        with st.expander("Shock Parameters", expanded=True):
            shock_thresh = st.slider(
                "Min Shock Level (%)", min_value=0.5, max_value=15.0,
                value=2.5, step=0.5,
            )
            bucket_boundary = st.slider(
                "Shock Bucket Boundary (%)", min_value=2.0, max_value=20.0,
                value=5.0, step=0.5,
            )
            min_shock_vol = st.slider(
                "Min Shock Trade Volume ($)", min_value=0, max_value=100000,
                value=1000, step=1000,
                help="Retain only shocks with driver trade volume >= min_shock_volume in the shock minute.",
            )

        with st.expander("Time Parameters", expanded=True):
            horizon_max = st.slider(
                "Max Horizon (min)", min_value=10, max_value=120,
                value=60, step=5,
            )
            st.caption("Bar granularity is fixed at 1min (pre-computed).")

        with st.expander("Sample Filter", expanded=True):
            # ── Central Bank group multi-selector ─────────────────────────────
            cb_icons = {"All": "🌐 All", "Fed": "🇺🇸 Fed", "BOJ": "🇯🇵 BOJ", "ECB": "🇪🇺 ECB"}
            # Always show all 4 options so the full list is visible in the dropdown
            bank_options = ["All", "Fed", "BOJ", "ECB"]
            bank_labels  = [cb_icons[g] for g in bank_options]

            # Default: Fed only
            default_bank_labels = [cb_icons["Fed"]]
            selected_bank_labels = st.multiselect(
                "Central Bank",
                options=bank_labels,
                default=default_bank_labels,
                help="Select one or more central banks. 'All' includes every event.",
                key="cb_group_multiselect",
            )
            # Fall back to default if user clears everything
            if not selected_bank_labels:
                selected_bank_labels = default_bank_labels

            selected_banks = [bank_options[bank_labels.index(lbl)] for lbl in selected_bank_labels]

            if "All" in selected_banks or set(selected_banks) == {"Fed", "BOJ", "ECB"}:
                cb_scope = "All Central Banks"
            else:
                cb_scope = " + ".join(selected_banks)

            # Build union of events for all selected banks
            if "All" in selected_banks:
                group_events = all_events
            else:
                seen = set()
                group_events = []
                for bank in selected_banks:
                    for ev in event_by_bank.get(bank, []):
                        if ev not in seen:
                            seen.add(ev)
                            group_events.append(ev)
                group_events = sorted(group_events)

            # Detect bank selection change → reset events multiselect
            _bank_key = tuple(sorted(selected_banks))
            if st.session_state.get("_cb_group") != _bank_key:
                st.session_state["_cb_group"] = _bank_key
                st.session_state["_events_key"] = st.session_state.get("_events_key", 0) + 1

            if not group_events:
                st.info("No events available for the selected central bank(s).")
            selected_events = st.multiselect(
                "Events",
                options=group_events,
                default=group_events,
                help="Pre-filtered by the Central Bank selector above.",
                key=f"events_{st.session_state.get('_events_key', 0)}",
            )
            dte_filter = st.selectbox(
                "Days to Expiry",
                options=["Full Period", "30d", "14d", "7d", "3d", "1d"],
                index=0,
            )

        with st.expander("Plot Settings", expanded=True):
            err_ci = st.slider(
                "CI Band (%)", min_value=50, max_value=99, value=95, step=5
            )
            show_ref_lines = st.checkbox("Show reference lines", value=True)
            align_y = st.checkbox("Align Y-axis across panels (Cols 1–2 grouped, Col 3 separate)", value=True)

    # ── Metrics row ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Events (selected)", len(selected_events))
    c2.metric("Elasticity records", f"{len(df_elas_raw):,}")
    c3.metric("Volume records", f"{len(df_vol_raw):,}")

    # ── Apply filters (fast, in-memory) ──────────────────────────────────────
    sel_ev_tuple = tuple(sorted(selected_events))
    with st.spinner("Applying filters..."):
        df_lo, df_pp, df_vol = apply_filters(
            df_elas=df_elas_raw,
            df_vol=df_vol_raw,
            shock_thresh=shock_thresh,
            bucket_boundary=bucket_boundary,
            horizon_max=horizon_max,
            selected_events=sel_ev_tuple,
            dte_filter=dte_filter,
            err_ci=err_ci,
            min_shock_vol=min_shock_vol,
        )

    with st.expander("📋 Data Summary & Filtered Shock Statistics", expanded=False):
        st.write(f"**Pre-computed records:** {len(df_elas_raw):,} elasticity · {len(df_vol_raw):,} volume")
        st.write(f"**Filtered records:** {len(df_lo):,} log-odds · {len(df_pp):,} prob-price · {len(df_vol):,} volume")

        if not df_lo.empty:
            shocks_sample = df_lo.drop_duplicates(subset=["Event", "Timestamp", "Driver"] if "Driver" in df_lo.columns else ["Event", "Timestamp"]).copy()
            shocks_sample["Size_Prob"] = shocks_sample["Abs_Jump_Pct"] / 100.0 if "Abs_Jump_Pct" in shocks_sample.columns else None

            ranks = ["All", "Top1", "Top2", "Top3"]

            # 1. Shock Size (|ΔP| prob price) grouped by Driver Rank
            size_rows = []
            for r in ranks:
                sub = shocks_sample if r == "All" else shocks_sample[shocks_sample["Driver_Rank"] == r]
                if sub.empty or "Size_Prob" not in sub.columns:
                    continue
                d = sub["Size_Prob"].describe()
                size_rows.append({
                    "Driver Rank": f"Group: {r}",
                    "Count": f"{int(d['count']):,}",
                    "Mean": f"{d['mean']:.4f}",
                    "Std": f"{d['std']:.4f}",
                    "Min": f"{d['min']:.4f}",
                    "25%": f"{d['25%']:.4f}",
                    "50% (Median)": f"{d['50%']:.4f}",
                    "75%": f"{d['75%']:.4f}",
                    "Max": f"{d['max']:.4f}",
                })

            # 2. Shock Volume ($ USD) grouped by Driver Rank
            vol_rows = []
            for r in ranks:
                sub = shocks_sample if r == "All" else shocks_sample[shocks_sample["Driver_Rank"] == r]
                if sub.empty or "Shock_Volume" not in sub.columns:
                    continue
                d = sub["Shock_Volume"].describe()
                vol_rows.append({
                    "Driver Rank": f"Group: {r}",
                    "Count": f"{int(d['count']):,}",
                    "Mean": f"${d['mean']:,.2f}",
                    "Std": f"${d['std']:,.2f}",
                    "Min": f"${d['min']:,.2f}",
                    "25%": f"${d['25%']:,.2f}",
                    "50% (Median)": f"${d['50%']:,.2f}",
                    "75%": f"${d['75%']:,.2f}",
                    "Max": f"${d['max']:,.2f}",
                })

            c_s, c_v = st.columns(2)
            with c_s:
                if size_rows:
                    st.markdown("##### 📐 Filtered Shock Size (|ΔP| prob price)")
                    st.dataframe(pd.DataFrame(size_rows).set_index("Driver Rank"), use_container_width=True)

            with c_v:
                if vol_rows:
                    st.markdown("##### 💵 Filtered Shock Volume ($ USD)")
                    st.dataframe(pd.DataFrame(vol_rows).set_index("Driver Rank"), use_container_width=True)

    st.caption(
        f"Showing {len(df_lo):,} log-odds · {len(df_pp):,} prob-price · "
        f"{len(df_vol):,} volume records after filters."
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_lo, tab_pp, tab_vol, tab_ts = st.tabs(
        ["Log-Odds Elasticity", "Prob Price Elasticity", "Volume Elasticity", "Single Event Time Series"]
    )

    # ── Param tags for unique download filenames ────────────────────────────
    scope_tag = cb_scope.lower().replace(" + ", "_").replace(" ", "_")
    shock_tag = f"shock{str(shock_thresh).replace('.', 'p')}"
    vol_tag = f"vol{int(min_shock_vol)}usd"
    bucket_tag = f"cutoff{str(bucket_boundary).replace('.', 'p')}"
    dte_tag = f"dte_{dte_filter.lower().replace(' ', '')}"
    horizon_tag = f"h{horizon_max}m"
    ci_tag = f"ci{err_ci}"
    param_suffix = f"{scope_tag}_{shock_tag}_{vol_tag}_{bucket_tag}_{dte_tag}_{horizon_tag}_{ci_tag}"
    if align_y:
        param_suffix += "_aligny"

    with tab_lo:
        st.subheader("Log-Odds Cross-Elasticity (γ_LO)")
        st.caption("γ_LO = Δlog-odds(target) / Δlog-odds(driver). Reference line at −1.0.")
        fig_lo = _plot_elasticity(
            result=df_lo, metric="Elasticity",
            title="Log-Odds Cross-Elasticity", ref_val=-1.0,
            err_ci=err_ci, show_ref_lines=show_ref_lines, y_label="γ_LO",
            shock_thresh=shock_thresh, bucket_boundary=bucket_boundary,
            dte_filter=dte_filter, horizon_max=horizon_max, cb_scope=cb_scope,
            min_shock_vol=min_shock_vol, share_y=align_y,
        )
        st.pyplot(fig_lo)
        st.download_button(
            "⬇️ Download PNG",
            data=fig_to_bytes(fig_lo),
            file_name=f"log_odds_elasticity_{param_suffix}.png",
            mime="image/png",
            key="dl_lo",
        )
        plt.close(fig_lo)

    with tab_pp:
        st.subheader("Prob-Price Cross-Elasticity (γ_PP)")
        st.caption("γ_PP = ΔP(target) / ΔP(driver). Reference line at −1.0.")
        fig_pp = _plot_elasticity(
            result=df_pp, metric="Elasticity",
            title="Prob-Price Cross-Elasticity", ref_val=-1.0,
            err_ci=err_ci, show_ref_lines=show_ref_lines, y_label="γ_PP",
            shock_thresh=shock_thresh, bucket_boundary=bucket_boundary,
            dte_filter=dte_filter, horizon_max=horizon_max, cb_scope=cb_scope,
            min_shock_vol=min_shock_vol, share_y=align_y,
        )
        st.pyplot(fig_pp)
        st.download_button(
            "⬇️ Download PNG",
            data=fig_to_bytes(fig_pp),
            file_name=f"prob_price_elasticity_{param_suffix}.png",
            mime="image/png",
            key="dl_pp",
        )
        plt.close(fig_pp)

    with tab_vol:
        st.subheader("Volume Cross-Elasticity (γ_VOL)")
        st.caption(
            "γ_VOL = cum_V(target) / cum_V(driver) in [t_anc, t_anc+k]. "
            "Min driver vol = 5; upper clip = 10×. Reference line at 1.0."
        )
        fig_vol = _plot_elasticity(
            result=df_vol, metric="Elasticity",
            title="Volume Cross-Elasticity", ref_val=1.0,
            err_ci=err_ci, show_ref_lines=show_ref_lines, y_label="γ_VOL",
            shock_thresh=shock_thresh, bucket_boundary=bucket_boundary,
            dte_filter=dte_filter, horizon_max=horizon_max, cb_scope=cb_scope,
            min_shock_vol=min_shock_vol, share_y=align_y,
        )
        st.pyplot(fig_vol)
        st.download_button(
            "⬇️ Download PNG",
            data=fig_to_bytes(fig_vol),
            file_name=f"volume_elasticity_{param_suffix}.png",
            mime="image/png",
            key="dl_vol",
        )
        plt.close(fig_vol)

    with tab_ts:
        st.subheader("Single Event Time Series")
        st.caption("3-panel time series plot showing probability price, logit price, and trading volume across all outcomes.")

        c_ev, c_freq, c_met, c_can = st.columns([2.5, 1, 1, 1])
        with c_ev:
            selected_ts_event = st.selectbox(
                "Event for Time Series",
                options=selected_events if selected_events else all_events,
                index=0,
                key="ts_event_selector",
            )
        with c_freq:
            resample_freq = st.selectbox(
                "Granularity",
                options=["1min", "5min", "15min", "1hour", "Raw Ticks"],
                index=1,
                key="ts_freq_selector",
            )
        with c_met:
            metric_vol = st.selectbox(
                "Volume Metric",
                options=["USD Volume", "Trade Count"],
                index=0,
                key="ts_vol_metric",
            )
        with c_can:
            st.write("")
            st.write("")
            canonical_only = st.checkbox("Canonical Outcomes", value=True, key="ts_canonical_only")

        if selected_ts_event:
            df_event_ticks = load_event_time_series(str(DB_PATH), selected_ts_event)
            if not df_event_ticks.empty:
                fig_ts = _plot_single_event_timeseries(
                    df_ticks=df_event_ticks,
                    event_slug=selected_ts_event,
                    resample_freq=resample_freq,
                    canonical_only=canonical_only,
                    metric_vol=metric_vol,
                )
                st.pyplot(fig_ts)

                ts_freq_tag = resample_freq.lower().replace(" ", "")
                ts_vol_tag = metric_vol.lower().replace(" ", "")
                ts_canon_tag = "canonical" if canonical_only else "all"
                ts_file_name = f"timeseries_{selected_ts_event}_{ts_freq_tag}_{ts_vol_tag}_{ts_canon_tag}.png"

                st.download_button(
                    "⬇️ Download PNG",
                    data=fig_to_bytes(fig_ts),
                    file_name=ts_file_name,
                    mime="image/png",
                    key="dl_ts",
                )
                plt.close(fig_ts)
            else:
                st.info(f"No trade tick data available for event '{selected_ts_event}'.")

    st.divider()
    st.caption(
        "Data: Polymarket FOMC trade ticks · "
        "Analysis: Cross-outcome price elasticity around decision shocks"
    )


if __name__ == "__main__" or True:
    main()
