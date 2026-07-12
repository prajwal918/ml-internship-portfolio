from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


RANDOM_STATE = 42


@dataclass
class ModelResult:
    name: str
    metrics: dict[str, float]
    model: Any
    data: pd.DataFrame
    extra: dict[str, Any]


def metric_table(metrics: dict[str, float]) -> pd.DataFrame:
    rows = [{"Metric": key, "Value": round(value, 4)} for key, value in metrics.items()]
    return pd.DataFrame(rows)


def risk_tier(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"
