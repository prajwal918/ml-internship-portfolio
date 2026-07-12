from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASETS = {
    "telco_churn": "blastchar/telco-customer-churn",
    "fake_real_news": "clmentbisaillon/fake-and-real-news-dataset",
    "ibm_hr": "pavansubhasht/ibm-hr-analytics-attrition-dataset",
}


def main() -> int:
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_config.exists():
        print("Missing Kaggle config: ~/.kaggle/kaggle.json")
        print("Create a Kaggle API token from Kaggle Account settings, then rerun this script.")
        return 1

    RAW.mkdir(parents=True, exist_ok=True)
    for name, slug in DATASETS.items():
        target = RAW / name
        target.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {slug} -> {target}")
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"],
            check=True,
        )
    print("Kaggle datasets downloaded. Restart Streamlit to use real data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
