"""Lightweight local dashboard (Streamlit): current P&L, open exposure,
and live calibration vs. the backtest baseline.

Run with: `streamlit run src/kalshi_quant/monitoring/dashboard.py`

Reads directly from the audit log (the same JSONL files execution/audit.py
writes) rather than a separate metrics database -- there is exactly one
record of what happened, and the dashboard is a view over it, not a
second copy that could drift out of sync.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kalshi_quant.config import load_config
from kalshi_quant.monitoring.reports import build_period_report


def _load_records(log_dir: Path, days: int) -> list[dict]:
    records = []
    today = date.today()
    for i in range(days):
        d = today - timedelta(days=i)
        path = log_dir / f"{d.isoformat()}.jsonl"
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    return records


def main() -> None:
    st.set_page_config(page_title="Kalshi Quant Dashboard", layout="wide")
    st.title("Kalshi Quant Trading Dashboard")

    config = load_config()
    log_dir = Path(config.get("logging", {}).get("audit_log_dir", "audit_logs"))
    lookback_days = st.sidebar.slider("Lookback window (days)", 1, 90, 30)

    records = _load_records(log_dir, lookback_days)
    if not records:
        st.warning(f"No audit log records found in {log_dir} for the last {lookback_days} days.")
        return

    settlements = [r for r in records if r.get("event_type") == "settlement"]
    df = pd.DataFrame(settlements)

    col1, col2, col3 = st.columns(3)
    total_pnl = df["pnl"].sum() if not df.empty and "pnl" in df else 0.0
    win_rate = (df["won"].mean() if not df.empty and "won" in df else 0.0)
    col1.metric("Total P&L (window)", f"${total_pnl:+.2f}")
    col2.metric("Win rate", f"{win_rate:.1%}")
    col3.metric("Settlements", len(df))

    if not df.empty:
        df = df.sort_values("timestamp")
        df["cumulative_pnl"] = df["pnl"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["cumulative_pnl"], mode="lines", name="Cumulative P&L"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Current open exposure")
    fills = [r for r in records if r.get("event_type") == "fill"]
    settled_tickers = {r.get("market_ticker") for r in settlements}
    open_fills = [f for f in fills if f.get("fill", {}).get("market_ticker") not in settled_tickers]
    if open_fills:
        st.dataframe(pd.DataFrame([f["fill"] for f in open_fills]))
    else:
        st.write("No open positions in this window.")

    st.subheader("Calibration (this window)")
    report = build_period_report(log_dir, date.today() - timedelta(days=lookback_days), date.today())
    if report.calibration:
        cal = report.calibration
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=cal.bin_mean_predicted, y=cal.bin_mean_observed, mode="markers+lines",
                                   name="Observed vs predicted"))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                   line=dict(dash="dash")))
        st.plotly_chart(fig2, use_container_width=True)
        st.write(f"Brier score: {cal.brier_score:.4f} | Log loss: {cal.log_loss_value:.4f}")
    else:
        st.write("Not enough matched decision/settlement pairs yet to plot calibration.")


if __name__ == "__main__":
    main()
