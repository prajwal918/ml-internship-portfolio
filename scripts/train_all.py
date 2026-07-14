from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_portfolio.artifacts import REPORTS_DIR, markdown_metric_table, save_metrics, save_model
from ml_portfolio.projects import churn, employee, fake_news, movie, stock
from ml_portfolio.logger import get_logger

logger = get_logger(__name__)

PROJECTS = [
    ("movie_recommendation", "Movie Recommendation System", movie.train_model),
    ("customer_churn", "Customer Churn Prediction", churn.train_model),
    ("stock_trend", "Stock Market Trend Analysis", stock.train_model),
    ("fake_news", "Fake News Detection System", fake_news.train_model),
    ("employee_performance", "Employee Performance Prediction", employee.train_model),
]


def main() -> int:
    sections: list[str] = ["# Generated Model Metrics\n"]
    for slug, display_name, trainer in PROJECTS:
        logger.info(f"Training {display_name}...")
        result = trainer()
        source = str(result.extra.get("source", "sample"))
        model_path = save_model(slug, result.model)
        metrics_path = save_metrics(slug, result.metrics, source)
        sections.append(markdown_metric_table(display_name, result.metrics, source))
        logger.info(f"Saved model: {model_path}")
        logger.info(f"Saved metrics: {metrics_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "MODEL_METRICS.md"
    summary_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info(f"Summary report generated: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
