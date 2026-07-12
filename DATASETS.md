# Dataset Guide

## Check Current Data Sources

```powershell
$env:PYTHONPATH="src"
python scripts/check_data_sources.py
```

Expected output after running `python scripts/download_public_data.py`:

```text
MovieLens: real
Customer Churn: real
Fake News: real
Employee Performance: real
Stock Market: real
```

## Automatic Public Download

```powershell
python scripts/download_public_data.py
```

This downloads:

- MovieLens 100K from GroupLens
- IBM/Telco Customer Churn CSV
- Fake/Real News CSV
- IBM HR Analytics CSV
- Real stock OHLCV data through `yfinance`

## Optional Kaggle Download

Kaggle requires an API token.

1. Go to Kaggle Account settings.
2. Create a new API token.
3. Save `kaggle.json` to `C:\Users\<you>\.kaggle\kaggle.json`.
4. Install requirements if needed:

```powershell
python -m pip install -r requirements.txt
```

5. Download datasets:

```powershell
python scripts/download_kaggle_data.py
```

## Manual CSV Placement

Place these files anywhere under `data/raw/`:

- `Telco-Customer-Churn.csv`
- `Fake.csv`
- `True.csv`
- `WA_Fn-UseC_-HR-Employee-Attrition.csv`

Restart Streamlit after adding files.
