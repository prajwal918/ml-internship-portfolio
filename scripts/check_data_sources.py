from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_portfolio.projects import churn, employee, fake_news, movie, stock
from ml_portfolio.logger import get_logger

logger = get_logger(__name__)

PROJECTS = ["movie", "churn", "fake_news", "employee", "stock"]

def main() -> int:
    try:
        logger.info("Checking data source availability...")
        status = []
        for p in PROJECTS:
            mod = getattr(__import__("ml_portfolio.projects", fromlist=[p]), p)
            status.append(f"{p}: {mod.dataset()[2] if len(mod.dataset()) > 2 else mod.dataset()[1]}")
            
        logger.info("\n".join(status))
        return 0
    except Exception as e:
        logger.error(f"Data source check failed unexpectedly: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Check interrupted by user.")
        sys.exit(130)
