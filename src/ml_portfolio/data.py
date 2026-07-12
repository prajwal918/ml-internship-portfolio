from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def find_file(*patterns: str) -> Path | None:
    if not RAW_DIR.exists():
        return None
    for pattern in patterns:
        matches = sorted(RAW_DIR.rglob(pattern))
        if matches:
            return matches[0]
    return None


def read_csv_if_exists(*patterns: str, **kwargs) -> pd.DataFrame | None:
    path = find_file(*patterns)
    if path is None:
        return None
    return pd.read_csv(path, **kwargs)
