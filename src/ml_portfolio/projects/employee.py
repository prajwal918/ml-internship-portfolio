from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_portfolio.data import read_csv_if_exists

from .common import ModelResult, RANDOM_STATE, metric_table


NUMERIC = ["age", "tenure_years", "training_hours", "projects_completed", "attendance_rate", "previous_rating"]
CATEGORICAL = ["department", "job_level", "gender"]


@st.cache_data(show_spinner=False)
def sample_data(seed: int = RANDOM_STATE, rows: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(22, 58, rows),
            "tenure_years": rng.normal(4.5, 3.0, rows).clip(0.2, 18).round(1),
            "training_hours": rng.normal(38, 18, rows).clip(0, 120).round(1),
            "projects_completed": rng.poisson(5, rows).clip(0, 16),
            "attendance_rate": rng.normal(0.93, 0.05, rows).clip(0.72, 1.0).round(3),
            "previous_rating": rng.normal(3.2, 0.65, rows).clip(1, 5).round(1),
            "department": rng.choice(["Engineering", "Sales", "HR", "Finance", "Support"], rows),
            "job_level": rng.choice(["Junior", "Mid", "Senior"], rows, p=[0.35, 0.45, 0.20]),
            "gender": rng.choice(["Female", "Male"], rows),
        }
    )
    score = (
        -4.2
        + 0.85 * df["previous_rating"]
        + 0.18 * df["projects_completed"]
        + 1.8 * df["attendance_rate"]
        + 0.01 * df["training_hours"]
        + (df["job_level"] == "Senior") * 0.35
    )
    probability = 1 / (1 + np.exp(-score))
    df["high_performer"] = rng.binomial(1, probability)
    return df


def load_real_data() -> pd.DataFrame | None:
    raw = read_csv_if_exists("*HR*.csv", "*attrition*.csv", "*employee*.csv")
    if raw is None or "PerformanceRating" not in raw.columns:
        return None
    required = {"Age", "YearsAtCompany", "TrainingTimesLastYear", "Department", "JobLevel", "Gender"}
    if not required.issubset(raw.columns):
        return None
    df = pd.DataFrame(
        {
            "age": pd.to_numeric(raw["Age"], errors="coerce"),
            "tenure_years": pd.to_numeric(raw["YearsAtCompany"], errors="coerce"),
            "training_hours": pd.to_numeric(raw["TrainingTimesLastYear"], errors="coerce") * 8,
            "projects_completed": pd.to_numeric(raw.get("NumCompaniesWorked", 3), errors="coerce"),
            "attendance_rate": 0.9 + pd.to_numeric(raw.get("JobInvolvement", 3), errors="coerce").fillna(3) * 0.02,
            "previous_rating": pd.to_numeric(raw.get("JobSatisfaction", 3), errors="coerce"),
            "department": raw["Department"].astype(str),
            "job_level": raw["JobLevel"].astype(str),
            "gender": raw["Gender"].astype(str),
            "high_performer": (pd.to_numeric(raw["PerformanceRating"], errors="coerce") >= 4).astype(int),
        }
    )
    return df.dropna()


def dataset() -> tuple[pd.DataFrame, str]:
    real = load_real_data()
    if real is not None and not real.empty:
        return real, "real"
    return sample_data(), "sample"


def _pipeline(model) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _demographic_parity(scored: pd.DataFrame) -> float:
    rates = scored.groupby("gender")["prediction"].mean()
    return float(rates.max() - rates.min()) if len(rates) > 1 else 0.0


@st.cache_resource(show_spinner=False)
def train_model() -> ModelResult:
    df, source = dataset()
    x = df[NUMERIC + CATEGORICAL]
    y = df["high_performer"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    baseline = _pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))
    final = _pipeline(RandomForestClassifier(n_estimators=220, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE))
    baseline.fit(x_train, y_train)
    final.fit(x_train, y_train)

    threshold = 0.25
    baseline_pred = (baseline.predict_proba(x_test)[:, 1] >= threshold).astype(int)
    final_prob = final.predict_proba(x_test)[:, 1]
    final_pred = (final_prob >= threshold).astype(int)
    scored = x_test.copy()
    scored["actual"] = y_test.values
    scored["prediction"] = final_pred
    scored["high_performer_probability"] = final_prob
    metrics = {
        "baseline_f1": f1_score(y_test, baseline_pred),
        "final_f1": f1_score(y_test, final_pred),
        "final_accuracy": accuracy_score(y_test, final_pred),
        "final_precision": precision_score(y_test, final_pred, zero_division=0),
        "final_recall": recall_score(y_test, final_pred, zero_division=0),
        "demographic_parity_difference": _demographic_parity(scored),
    }
    return ModelResult("Random Forest employee performance classifier", metrics, final, df, {"scored": scored, "baseline": baseline, "source": source})


def predict_employee(record: dict[str, object]) -> float:
    result = train_model()
    return float(result.model.predict_proba(pd.DataFrame([record]))[:, 1][0])


def render_page() -> None:
    st.header("Employee Performance Prediction Model")
    st.write("Decision-support dashboard with fairness metric reporting. Not for automated HR decisions.")
    result = train_model()
    st.caption(f"Data source: {result.extra['source']} dataset")

    cols = st.columns(3)
    for col, (name, value) in zip(cols * 2, result.metrics.items()):
        col.metric(name.replace("_", " ").title(), f"{value:.3f}")

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Score an Employee")
        record = {
            "age": st.slider("Age", 22, 58, 30),
            "tenure_years": st.slider("Tenure years", 0.2, 18.0, 3.0),
            "training_hours": st.slider("Training hours", 0.0, 120.0, 40.0),
            "projects_completed": st.slider("Projects completed", 0, 16, 5),
            "attendance_rate": st.slider("Attendance rate", 0.72, 1.0, 0.94),
            "previous_rating": st.slider("Previous rating", 1.0, 5.0, 3.4),
            "department": st.selectbox("Department", ["Engineering", "Sales", "HR", "Finance", "Support"]),
            "job_level": st.selectbox("Job level", ["Junior", "Mid", "Senior"]),
            "gender": st.selectbox("Gender", ["Female", "Male"]),
        }
        probability = predict_employee(record)
        st.metric("High Performer Probability", f"{probability:.1%}")
    with right:
        st.subheader("Scored Employee Sample")
        st.dataframe(result.extra["scored"].sort_values("high_performer_probability", ascending=False).head(20), use_container_width=True)

    st.subheader("Model and Fairness Metrics")
    st.dataframe(metric_table(result.metrics), use_container_width=True)
