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
    return kaggle_config.exists()


def main() -> int:
    try:
        if not check_kaggle_credentials():
            logger.error("Kaggle credentials not found or invalid.")
            return 1
            
        logger.info("Starting Kaggle data downloads...")
        
        RAW.mkdir(parents=True, exist_ok=True)
        for name, slug in DATASETS.items():
            target = RAW / name
            target.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading {slug} -> {target}")
            subprocess.run(
                [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"],
                check=True,
            )
        
        logger.info("Kaggle data downloads placeholder finished.")
        return 0
    except Exception as e:
        logger.error(f"Failed during Kaggle data processing: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user.")
        sys.exit(130)
