# Deployment Guide

## Local

```powershell
cd C:\Users\jogip\OneDrive\Desktop\MY_ORGANIZED_DESKTOP\dfg\urlr
$env:PYTHONPATH="src"
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Streamlit Community Cloud

1. Push this folder to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.

## Dataset Notes

Do not commit large Kaggle datasets to GitHub. Place them in `data/raw/` locally or upload them through the deployment provider if needed.

The app works without Kaggle data by using sample fallback data, but final model claims should use real datasets.
