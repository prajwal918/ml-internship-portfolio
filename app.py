import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

import pandas as pd

from ml_portfolio.catalog import DELIVERABLES, PROJECT_CATALOG
from ml_portfolio.projects import churn, employee, fake_news, movie, stock


PROJECTS = {
    "Portfolio Overview": None,
    "Movie Recommendation System": movie.render_page,
    "Customer Churn Prediction": churn.render_page,
    "Stock Market Trend Analysis": stock.render_page,
    "Fake News Detection System": fake_news.render_page,
    "Employee Performance Prediction": employee.render_page,
}


st.set_page_config(
    page_title="ML Internship Portfolio",
    layout="wide",
)

choice = st.sidebar.radio("Project", list(PROJECTS.keys()))
st.sidebar.divider()
st.sidebar.markdown("**Submission Package**")
for item in DELIVERABLES:
    st.sidebar.markdown(f"- {item}")


def render_overview() -> None:
    st.title("Data Science / ML Internship Portfolio")
    st.caption("Five ML projects with dashboards, model training, testing, and report-ready outputs.")

    metrics = st.columns(4)
    metrics[0].metric("Projects", "5")
    metrics[1].metric("Tests", "5 smoke tests")
    metrics[2].metric("Real Data Ready", "Yes")
    metrics[3].metric("App Type", "Streamlit")

    st.subheader("Project Portfolio")
    st.dataframe(pd.DataFrame(PROJECT_CATALOG), use_container_width=True, hide_index=True)

    st.subheader("Dataset Status")
    status_rows = [
        {"Project": "Movie Recommendation", "Current Status": movie.dataset()[2]},
        {"Project": "Customer Churn", "Current Status": churn.dataset()[1]},
        {"Project": "Stock Market Trend", "Current Status": stock.dataset()[1]},
        {"Project": "Fake News", "Current Status": fake_news.dataset()[1]},
        {"Project": "Employee Performance", "Current Status": employee.dataset()[1]},
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.subheader("What Is Included")
    st.markdown(
        "\n".join(
            [
                "- Working Streamlit dashboard for all five projects",
                "- Baseline and final model metrics where applicable",
                "- Real dataset loaders for public/Kaggle CSVs",
                "- Training script to export model artifacts and metrics",
                "- Report and submission documentation",
            ]
        )
    )


try:
    if choice == "Portfolio Overview":
        render_overview()
    else:
        PROJECTS[choice]()
except Exception as e:
    st.error("An unexpected error occurred while loading the module.")
    st.exception(e)
