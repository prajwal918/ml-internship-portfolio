from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_string_dtype
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_portfolio.data import find_file

from .common import ModelResult, RANDOM_STATE, metric_table


REAL_PATTERNS = [
    "Officials released a report after reviewing public records and verified interviews.",
    "The company announced quarterly results with revenue growth and audited statements.",
    "Researchers published findings in a peer reviewed journal with supporting data.",
    "The court issued a written decision after hearing arguments from both parties.",
    "Local authorities confirmed the update in a press briefing on Monday.",
]
FAKE_PATTERNS = [
    "Secret sources reveal a shocking plan that nobody wants you to know.",
    "This miracle cure is being hidden by powerful groups around the world.",
    "A viral post claims impossible numbers without evidence or documents.",
    "Anonymous insiders say the entire event was staged for hidden reasons.",
    "You will not believe this explosive claim spreading online tonight.",
]


@st.cache_data(show_spinner=False)
def sample_data(seed: int = RANDOM_STATE, rows: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for idx in range(rows):
        label = int(idx % 2 == 0)
        base = rng.choice(FAKE_PATTERNS if label else REAL_PATTERNS)
        topic = rng.choice(["health", "finance", "election", "sports", "technology", "education"])
        text = f"{base} The story concerns {topic} and has been shared {rng.integers(10, 5000)} times."
        records.append({"text": text, "fake": label})
    return pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_real_data() -> pd.DataFrame | None:
    fake_path = find_file("Fake.csv", "*fake*.csv")
    true_path = find_file("True.csv", "*true*.csv", "*real*.csv")
    if fake_path and true_path and fake_path != true_path:
        fake = pd.read_csv(fake_path)
        true = pd.read_csv(true_path)
        text_col = "text" if "text" in fake.columns else fake.columns[0]
        title_col = "title" if "title" in fake.columns else None
        fake_text = fake[text_col].astype(str)
        true_text = true[text_col].astype(str)
        if title_col:
            fake_text = fake[title_col].astype(str) + " " + fake_text
            true_text = true[title_col].astype(str) + " " + true_text
        return pd.concat(
            [
                pd.DataFrame({"text": fake_text, "fake": 1}),
                pd.DataFrame({"text": true_text, "fake": 0}),
            ],
            ignore_index=True,
        ).dropna()

    generic = find_file("train.csv", "*news*.csv", "*liar*.csv")
    if generic:
        raw = pd.read_csv(generic)
        text_candidates = [col for col in ["text", "statement", "content", "article", "title"] if col in raw.columns]
        label_candidates = [col for col in ["fake", "label", "target", "class"] if col in raw.columns]
        if text_candidates and label_candidates:
            text_col = text_candidates[0]
            label_col = label_candidates[0]
            labels = raw[label_col]
            if labels.dtype == object or is_string_dtype(labels):
                label_text = labels.astype(str).str.strip().str.lower()
                labels = label_text.isin(["fake", "false", "1", "pants-fire", "barely-true"]).astype(int)
            else:
                labels = pd.to_numeric(labels, errors="coerce")
            return pd.DataFrame({"text": raw[text_col].astype(str), "fake": labels}).dropna().assign(fake=lambda df: df["fake"].astype(int))
    return None


def dataset() -> tuple[pd.DataFrame, str]:
    real = load_real_data()
    if real is not None and not real.empty:
        if len(real) > 12000:
            real = real.sample(12000, random_state=RANDOM_STATE)
        return real, "real"
    return sample_data(), "sample"


@st.cache_resource(show_spinner=False)
def train_model() -> ModelResult:
    df, source = dataset()
    x_train, x_test, y_train, y_test = train_test_split(df["text"], df["fake"], test_size=0.25, stratify=df["fake"], random_state=RANDOM_STATE)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_test, probs),
        "precision_fake": precision_score(y_test, preds),
        "recall_fake": recall_score(y_test, preds),
        "f1_fake": f1_score(y_test, preds),
    }
    scored = pd.DataFrame({"text": x_test, "actual_fake": y_test, "fake_probability": probs, "prediction": preds})
    return ModelResult("TF-IDF Logistic Regression fake news classifier", metrics, model, df, {"scored": scored, "source": source})


def predict_text(text: str) -> tuple[str, float]:
    result = train_model()
    probability = float(result.model.predict_proba([text])[:, 1][0])
    return ("Likely Fake" if probability >= 0.5 else "Likely Real", probability)


def render_page() -> None:
    st.header("Fake News Detection System")
    st.write("Assistive text classifier with confidence scoring. It should not be used as a final truth authority.")
    result = train_model()
    st.caption(f"Data source: {result.extra['source']} dataset")

    cols = st.columns(4)
    for col, (name, value) in zip(cols, result.metrics.items()):
        col.metric(name.replace("_", " ").title(), f"{value:.3f}")

    text = st.text_area("Paste news text", value="Anonymous insiders say a shocking secret has been hidden from everyone.", height=140)
    label, probability = predict_text(text)
    st.metric("Prediction", label, f"{probability:.1%} fake probability")

    st.subheader("Sample Scored Texts")
    st.dataframe(result.extra["scored"].head(20), use_container_width=True)
    st.subheader("Model Metrics")
    st.dataframe(metric_table(result.metrics), use_container_width=True)
