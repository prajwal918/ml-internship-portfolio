from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "generated"


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_model(name: str, model: Any) -> Path:
    ensure_dirs()
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def save_metrics(name: str, metrics: dict[str, float], source: str) -> Path:
    ensure_dirs()
    payload = {
        "project": name,
        "data_source": source,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    path = REPORTS_DIR / f"{name}_metrics.json"
    write_json(path, payload)
    return path


def markdown_metric_table(project_name: str, metrics: dict[str, float], source: str) -> str:
    rows = "\n".join(f"| {key} | {value:.4f} |" for key, value in metrics.items())
    return (
        f"## {project_name}\n\n"
        f"Data source: **{source}**\n\n"
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"{rows}\n"
    )
