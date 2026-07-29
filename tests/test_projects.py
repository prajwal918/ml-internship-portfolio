"""
Unit tests for ml-internship-portfolio.
Tests pure data and math logic only — no Streamlit context required.
These tests run cleanly in CI/CD environments.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ── Shared helpers ──────────────────────────────────────────────────────────

RANDOM_STATE = 42


def make_churn_df(rows: int = 200, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure": rng.integers(1, 73, rows),
        "monthly_charges": rng.normal(72, 24, rows).clip(20, 130).round(2),
        "support_tickets": rng.poisson(1.3, rows),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], rows),
        "internet_service": rng.choice(["DSL", "Fiber optic", "No"], rows),
        "payment_method": rng.choice(
            ["Electronic check", "Credit card", "Bank transfer", "Mailed check"], rows
        ),
    })
    df["total_charges"] = (
        df["tenure"] * df["monthly_charges"] + rng.normal(0, 120, rows)
    ).clip(0).round(2)
    logit = (
        -1.2
        - 0.035 * df["tenure"]
        + 0.018 * df["monthly_charges"]
        + 0.36 * df["support_tickets"]
    )
    prob = 1 / (1 + np.exp(-logit))
    df["churn"] = rng.binomial(1, prob)
    return df


NUMERIC = ["tenure", "monthly_charges", "total_charges", "support_tickets"]
CATEGORICAL = ["contract", "internet_service", "payment_method"]


def build_pipeline(model) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


# ── Data shape tests ─────────────────────────────────────────────────────────

def test_churn_dataframe_columns():
    df = make_churn_df()
    expected = set(NUMERIC + CATEGORICAL + ["churn"])
    assert expected.issubset(df.columns)


def test_churn_dataframe_rows():
    df = make_churn_df(rows=100)
    assert len(df) == 100


def test_churn_target_is_binary():
    df = make_churn_df()
    assert set(df["churn"].unique()).issubset({0, 1})


def test_monthly_charges_in_range():
    df = make_churn_df(rows=500)
    assert df["monthly_charges"].between(20, 130).all()


def test_total_charges_non_negative():
    df = make_churn_df()
    assert (df["total_charges"] >= 0).all()


# ── Model pipeline tests ─────────────────────────────────────────────────────

def test_logistic_regression_pipeline_trains():
    df = make_churn_df()
    x = df[NUMERIC + CATEGORICAL]
    y = df["churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE
    )
    pipe = build_pipeline(LogisticRegression(max_iter=500, random_state=RANDOM_STATE))
    pipe.fit(x_train, y_train)
    probs = pipe.predict_proba(x_test)[:, 1]
    assert probs.shape[0] == len(y_test)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_random_forest_pipeline_trains():
    df = make_churn_df()
    x = df[NUMERIC + CATEGORICAL]
    y = df["churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE
    )
    pipe = build_pipeline(
        RandomForestClassifier(n_estimators=20, random_state=RANDOM_STATE)
    )
    pipe.fit(x_train, y_train)
    probs = pipe.predict_proba(x_test)[:, 1]
    assert probs.shape[0] == len(y_test)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_prediction_probability_range():
    df = make_churn_df()
    x = df[NUMERIC + CATEGORICAL]
    y = df["churn"]
    x_train, x_test, y_train, _ = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE
    )
    pipe = build_pipeline(LogisticRegression(max_iter=500, random_state=RANDOM_STATE))
    pipe.fit(x_train, y_train)
    sample = x_test.iloc[[0]]
    prob = float(pipe.predict_proba(sample)[:, 1][0])
    assert 0.0 <= prob <= 1.0


def test_pipeline_predict_shape_matches_input():
    df = make_churn_df(rows=300)
    x = df[NUMERIC + CATEGORICAL]
    y = df["churn"]
    x_train, x_test, y_train, _ = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_STATE
    )
    pipe = build_pipeline(
        RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE)
    )
    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)
    assert len(preds) == len(x_test)


# ── Feature engineering tests ────────────────────────────────────────────────

def test_risk_tier_logic():
    def risk_tier(prob: float) -> str:
        if prob >= 0.7:
            return "High"
        elif prob >= 0.4:
            return "Medium"
        return "Low"

    assert risk_tier(0.8) == "High"
    assert risk_tier(0.5) == "Medium"
    assert risk_tier(0.2) == "Low"
    assert risk_tier(0.0) == "Low"
    assert risk_tier(1.0) == "High"


def test_numpy_operations_deterministic():
    rng1 = np.random.default_rng(RANDOM_STATE)
    rng2 = np.random.default_rng(RANDOM_STATE)
    arr1 = rng1.integers(0, 100, 50)
    arr2 = rng2.integers(0, 100, 50)
    np.testing.assert_array_equal(arr1, arr2)
