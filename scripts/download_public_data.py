from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def download_movielens_100k() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "ml-100k.zip"
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    print(f"Downloading {url}")
    urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(RAW)
    print(f"Extracted MovieLens 100K into {RAW}")


def download_csv(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urlretrieve(url, destination)
    print(f"Saved {destination}")


def download_tabular_public_datasets() -> None:
    download_csv(
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        RAW / "Telco-Customer-Churn.csv",
    )
    download_csv(
        "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv",
        RAW / "fake_or_real_news.csv",
    )
    download_csv(
        "https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        RAW / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
    )


def download_stock_ohlcv() -> None:
    tickers = ["AAPL", "MSFT", "INFY.NS", "RELIANCE.NS", "TCS.NS"]
    frames = []
    for ticker in tickers:
        print(f"Downloading OHLCV for {ticker}")
        raw = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        date_col = "Date" if "Date" in raw.columns else raw.columns[0]
        frames.append(
            pd.DataFrame(
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
        )
    if frames:
        path = RAW / "stock_ohlcv.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"Saved {path}")


if __name__ == "__main__":
    download_movielens_100k()
    download_tabular_public_datasets()
    download_stock_ohlcv()
    print("Public datasets downloaded. Kaggle API is optional for alternate dataset copies.")
