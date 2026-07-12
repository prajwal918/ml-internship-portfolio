from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_portfolio.data import read_csv_if_exists

from .common import ModelResult, RANDOM_STATE, metric_table, risk_tier


NUMERIC = ["tenure", "monthly_charges", "total_charges", "support_tickets"]
CATEGORICAL = ["contract", "internet_service", "payment_method"]


@st.cache_data(show_spinner=False)
def sample_data(seed: int = RANDOM_STATE, rows: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "tenure": rng.integers(1, 73, rows),
            "monthly_charges": rng.normal(72, 24, rows).clip(20, 130).round(2),
            "support_tickets": rng.poisson(1.3, rows),
            "contract": rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[0.55, 0.25, 0.20]),
            "internet_service": rng.choice(["DSL", "Fiber optic", "No"], rows, p=[0.35, 0.52, 0.13]),
            "payment_method": rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], rows),
        }
    )
    df["total_charges"] = (df["tenure"] * df["monthly_charges"] + rng.normal(0, 120, rows)).clip(0).round(2)
    logit = (
        -1.2
        - 0.035 * df["tenure"]
        + 0.018 * df["monthly_charges"]
        + 0.36 * df["support_tickets"]
        + (df["contract"] == "Month-to-month") * 1.2
        + (df["payment_method"] == "Electronic check") * 0.45
        + (df["internet_service"] == "Fiber optic") * 0.35
    )
    probability = 1 / (1 + np.exp(-logit))
    df["churn"] = rng.binomial(1, probability)
    return df


def load_real_data() -> pd.DataFrame | None:
    raw = read_csv_if_exists("Telco-Customer-Churn.csv", "*churn*.csv")
    if raw is None:
        return None
    required = {"tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod", "Churn"}
    if not required.issubset(raw.columns):
        return None
    df = pd.DataFrame(
        {
            "tenure": pd.to_numeric(raw["tenure"], errors="coerce"),
            "monthly_charges": pd.to_numeric(raw["MonthlyCharges"], errors="coerce"),
            "total_charges": pd.to_numeric(raw["TotalCharges"], errors="coerce"),
            "support_tickets": 0,
            "contract": raw["Contract"].astype(str),
            "internet_service": raw["InternetService"].astype(str),
            "payment_method": raw["PaymentMethod"].astype(str),
            "churn": raw["Churn"].astype(str).str.lower().map({"yes": 1, "no": 0}),
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


@st.cache_resource(show_spinner=False)
def train_model() -> ModelResult:
    df, source = dataset()
    x = df[NUMERIC + CATEGORICAL]
    y = df["churn"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    baseline = _pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))
    final = _pipeline(RandomForestClassifier(n_estimators=180, min_samples_leaf=4, class_weight="balanced", random_state=RANDOM_STATE))
    baseline.fit(x_train, y_train)
    final.fit(x_train, y_train)

    baseline_prob = baseline.predict_proba(x_test)[:, 1]
    final_prob = final.predict_proba(x_test)[:, 1]
    final_pred = (final_prob >= 0.5).astype(int)
    metrics = {
        "baseline_roc_auc": roc_auc_score(y_test, baseline_prob),
        "final_roc_auc": roc_auc_score(y_test, final_prob),
        "final_precision": precision_score(y_test, final_pred),
        "final_recall": recall_score(y_test, final_pred),
        "final_f1": f1_score(y_test, final_pred),
    }
    scored = x_test.copy()
    scored["actual_churn"] = y_test.values
    scored["churn_probability"] = final_prob
    scored["risk_tier"] = scored["churn_probability"].map(risk_tier)
    return ModelResult("Random Forest churn classifier", metrics, final, df, {"scored": scored, "baseline": baseline, "source": source})


def predict_customer(record: dict[str, object]) -> float:
    result = train_model()
    return float(result.model.predict_proba(pd.DataFrame([record]))[:, 1][0])


def render_page() -> None:
    st.header("Customer Churn Prediction")
    st.write("Risk scoring for customer retention with baseline and final classifier metrics.")
    result = train_model()
    st.caption(f"Data source: {result.extra['source']} dataset")

    cols = st.columns(5)
    for col, (name, value) in zip(cols, result.metrics.items()):
        col.metric(name.replace("_", " ").title(), f"{value:.3f}")

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Score a Customer")
        record = {
            "tenure": st.slider("Tenure months", 1, 72, 12),
            "monthly_charges": st.slider("Monthly charges", 20.0, 130.0, 80.0),
            "total_charges": st.slider("Total charges", 0.0, 9000.0, 960.0),
            "support_tickets": st.slider("Support tickets", 0, 8, 2),
            "contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
            "internet_service": st.selectbox("Internet service", ["DSL", "Fiber optic", "No"]),
            "payment_method": st.selectbox("Payment method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"]),
        }
        probability = predict_customer(record)
        st.metric("Churn Risk", f"{probability:.1%}", risk_tier(probability))
    with right:
        st.subheader("Highest Risk Customers")
        st.dataframe(result.extra["scored"].sort_values("churn_probability", ascending=False).head(20), use_container_width=True)

    st.subheader("Model Metrics")
    st.dataframe(metric_table(result.metrics), use_container_width=True)
