from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_portfolio.projects import churn, employee, fake_news, movie, stock


CHECKS = [
    ("MovieLens", lambda: movie.dataset()[2]),
    ("Customer Churn", lambda: churn.dataset()[1]),
    ("Fake News", lambda: fake_news.dataset()[1]),
    ("Employee Performance", lambda: employee.dataset()[1]),
    ("Stock Market", lambda: stock.dataset()[1]),
]


def main() -> int:
    print("Dataset source check")
    print("--------------------")
    for name, check in CHECKS:
        print(f"{name}: {check()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
