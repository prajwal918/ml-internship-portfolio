from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import ModelResult, RANDOM_STATE, metric_table
from ml_portfolio.data import find_file


FEATURES = ["return_1d", "ma_gap", "rsi", "volatility"]


@st.cache_data(show_spinner=False)
def sample_data(seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    dates = pd.bdate_range("2024-01-01", periods=360)
    for ticker, drift in {"AAPL": 0.0004, "MSFT": 0.0003, "INFY.NS": 0.0005, "RELIANCE.NS": 0.0002, "TCS.NS": 0.00035}.items():
        returns = rng.normal(drift, 0.018, len(dates))
        close = 100 * np.exp(np.cumsum(returns))
        open_price = close * (1 + rng.normal(0, 0.004, len(dates)))
        high = np.maximum(open_price, close) * (1 + rng.random(len(dates)) * 0.018)
        low = np.minimum(open_price, close) * (1 - rng.random(len(dates)) * 0.018)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": rng.integers(300_000, 4_000_000, len(dates)),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    return add_indicators(df)


def load_real_data() -> pd.DataFrame | None:
    csv_path = find_file("stock_ohlcv.csv", "*ohlcv*.csv")
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        required = {"date", "ticker", "open", "high", "low", "close", "volume"}
        if required.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"])
            return add_indicators(df)

    tickers = ["AAPL", "MSFT", "INFY.NS", "RELIANCE.NS", "TCS.NS"]
    frames = []
    for ticker in tickers:
        try:
            raw = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
        except Exception:
            raw = pd.DataFrame()
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        date_col = "Date" if "Date" in raw.columns else raw.columns[0]
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[date_col]),
                "ticker": ticker,
                "open": pd.to_numeric(raw["Open"], errors="coerce"),
                "high": pd.to_numeric(raw["High"], errors="coerce"),
                "low": pd.to_numeric(raw["Low"], errors="coerce"),
                "close": pd.to_numeric(raw["Close"], errors="coerce"),
                "volume": pd.to_numeric(raw["Volume"], errors="coerce"),
            }
        ).dropna()
        frames.append(frame)
    if frames:
        return add_indicators(pd.concat(frames, ignore_index=True))
    return None


def dataset() -> tuple[pd.DataFrame, str]:
    real = load_real_data()
    if real is not None and not real.empty:
        return real, "real"
    return sample_data(), "sample"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, group in df.sort_values(["ticker", "date"]).groupby("ticker"):
        g = group.copy()
        g["return_1d"] = g["close"].pct_change()
        g["ma_fast"] = g["close"].rolling(10).mean()
        g["ma_slow"] = g["close"].rolling(30).mean()
        g["ma_gap"] = (g["ma_fast"] - g["ma_slow"]) / g["ma_slow"]
        delta = g["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        g["rsi"] = 100 - (100 / (1 + rs))
        g["volatility"] = g["return_1d"].rolling(14).std()
        g["target"] = (g["close"].shift(-5) > g["close"]).astype(int)
        out.append(g)
    return pd.concat(out).dropna().reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_model() -> ModelResult:
    df, source = dataset()
    x = df[FEATURES]
    y = df["target"]
    split = int(len(df) * 0.75)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    baseline_pred = (x_test["ma_gap"] > 0).astype(int)
    final = Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(n_estimators=180, max_depth=5, random_state=RANDOM_STATE))])
    final.fit(x_train, y_train)
    model_pred = final.predict(x_test)
    baseline_f1 = f1_score(y_test, baseline_pred)
    model_f1 = f1_score(y_test, model_pred)
    final_pred = model_pred if model_f1 >= baseline_f1 else baseline_pred.to_numpy()

    metrics = {
        "baseline_accuracy": accuracy_score(y_test, baseline_pred),
        "final_accuracy": accuracy_score(y_test, final_pred),
        "baseline_f1": baseline_f1,
        "final_f1": f1_score(y_test, final_pred),
    }
    scored = df.iloc[split:].copy()
    scored["prediction"] = final_pred
    scored["strategy_return"] = scored["prediction"].shift(1).fillna(0) * scored["return_1d"]
    scored["buy_hold_return"] = scored["return_1d"]
    return ModelResult("Random Forest trend classifier", metrics, final, df, {"scored": scored, "source": source})


def render_page() -> None:
    st.header("Stock Market Trend Analysis")
    st.warning("Educational demo only. This is not investment advice.")
    result = train_model()
    st.caption(f"Data source: {result.extra['source']} dataset")

    cols = st.columns(4)
    for col, (name, value) in zip(cols, result.metrics.items()):
        col.metric(name.replace("_", " ").title(), f"{value:.3f}")

    ticker = st.selectbox("Ticker", sorted(result.data["ticker"].unique()))
    chart_df = result.data[result.data["ticker"] == ticker].tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=chart_df["date"], open=chart_df["open"], high=chart_df["high"], low=chart_df["low"], close=chart_df["close"], name=ticker))
    fig.add_trace(go.Scatter(x=chart_df["date"], y=chart_df["ma_fast"], name="MA 10"))
    fig.add_trace(go.Scatter(x=chart_df["date"], y=chart_df["ma_slow"], name="MA 30"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Backtest Snapshot")
    scored = result.extra["scored"]
    st.line_chart(scored.groupby("date")[["strategy_return", "buy_hold_return"]].mean().cumsum())
    st.dataframe(metric_table(result.metrics), use_container_width=True)
