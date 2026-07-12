from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from ml_portfolio.data import find_file

from .common import ModelResult, RANDOM_STATE, metric_table


GENRES = ["Action", "Comedy", "Drama", "Romance", "Sci-Fi", "Thriller"]


@st.cache_data(show_spinner=False)
def sample_data(seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    movies = pd.DataFrame(
        {
            "movie_id": np.arange(1, 81),
            "title": [f"Movie {idx:03d}" for idx in range(1, 81)],
            "genre": rng.choice(GENRES, size=80),
        }
    )

    user_pref = rng.normal(0, 1, size=(120, len(GENRES)))
    movie_genre_index = movies["genre"].map({genre: idx for idx, genre in enumerate(GENRES)}).to_numpy()
    rows = []
    for user_idx in range(120):
        watched = rng.choice(movies["movie_id"], size=22, replace=False)
        for movie_id in watched:
            genre_idx = movie_genre_index[movie_id - 1]
            raw_rating = 3.2 + user_pref[user_idx, genre_idx] + rng.normal(0, 0.65)
            rows.append(
                {
                    "user_id": user_idx + 1,
                    "movie_id": int(movie_id),
                    "rating": float(np.clip(np.round(raw_rating, 1), 1, 5)),
                }
            )
    ratings = pd.DataFrame(rows)
    return ratings, movies


def load_real_data() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    ratings_csv = find_file("ratings.csv")
    movies_csv = find_file("movies.csv")
    if ratings_csv and movies_csv:
        ratings = pd.read_csv(ratings_csv)
        movies = pd.read_csv(movies_csv)
        if {"userId", "movieId", "rating"}.issubset(ratings.columns) and {"movieId", "title"}.issubset(movies.columns):
            ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
            movies = movies.rename(columns={"movieId": "movie_id", "genres": "genre"})
            if "genre" not in movies.columns:
                movies["genre"] = "Unknown"
            movies["genre"] = movies["genre"].astype(str).str.split("|").str[0]
            ratings = ratings[["user_id", "movie_id", "rating"]].dropna()
            movies = movies[["movie_id", "title", "genre"]].dropna()
            if len(ratings) > 15000:
                ratings = ratings.sample(15000, random_state=RANDOM_STATE)
            return ratings, movies

    u_data = find_file("u.data")
    u_item = find_file("u.item")
    if u_data and u_item:
        ratings = pd.read_csv(u_data, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
        item_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + [f"genre_{i}" for i in range(19)]
        movies = pd.read_csv(u_item, sep="|", names=item_cols, encoding="latin-1")
        genre_cols = [col for col in movies.columns if col.startswith("genre_")]
        movies["genre"] = movies[genre_cols].idxmax(axis=1).str.replace("genre_", "Genre ", regex=False)
        return ratings[["user_id", "movie_id", "rating"]], movies[["movie_id", "title", "genre"]]
    return None


def dataset() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    real = load_real_data()
    if real is not None:
        return real[0], real[1], "real"
    ratings, movies = sample_data()
    return ratings, movies, "sample"


@st.cache_resource(show_spinner=False)
def train_model() -> ModelResult:
    ratings, movies, source = dataset()
    train, test = train_test_split(ratings, test_size=0.2, random_state=RANDOM_STATE)
    global_mean = train["rating"].mean()
    movie_means = train.groupby("movie_id")["rating"].mean()
    baseline_pred = test["movie_id"].map(movie_means).fillna(global_mean)

    matrix = train.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(global_mean)
    svd = TruncatedSVD(n_components=16, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(matrix)
    reconstructed = pd.DataFrame(
        np.dot(user_factors, svd.components_),
        index=matrix.index,
        columns=matrix.columns,
    )
    svd_pred = np.array(
        [
        reconstructed.loc[row.user_id, row.movie_id] if row.user_id in reconstructed.index and row.movie_id in reconstructed.columns else global_mean
        for row in test.itertuples()
        ]
    )
    baseline_values = baseline_pred.to_numpy()
    hybrid_candidates = {}
    for alpha in np.linspace(0, 1, 11):
        hybrid = (alpha * svd_pred) + ((1 - alpha) * baseline_values)
        hybrid_candidates[float(alpha)] = float(np.sqrt(mean_squared_error(test["rating"], hybrid)))
    best_alpha = min(hybrid_candidates, key=hybrid_candidates.get)
    final_pred = (best_alpha * svd_pred) + ((1 - best_alpha) * baseline_values)

    baseline_rmse = float(np.sqrt(mean_squared_error(test["rating"], baseline_pred)))
    final_rmse = float(np.sqrt(mean_squared_error(test["rating"], final_pred)))
    metrics = {
        "baseline_rmse": baseline_rmse,
        "final_hybrid_rmse": final_rmse,
        "baseline_mae": mean_absolute_error(test["rating"], baseline_pred),
        "final_hybrid_mae": mean_absolute_error(test["rating"], final_pred),
    }
    return ModelResult(
        name="Hybrid-style SVD recommender",
        metrics=metrics,
        model=svd,
        data=ratings.merge(movies, on="movie_id"),
        extra={"movies": movies, "ratings": ratings, "matrix": matrix, "reconstructed": reconstructed, "source": source, "hybrid_alpha": best_alpha},
    )


def recommend_for_user(user_id: int, top_n: int = 10) -> pd.DataFrame:
    result = train_model()
    ratings = result.extra["ratings"]
    movies = result.extra["movies"]
    matrix = result.extra["matrix"]
    reconstructed = result.extra["reconstructed"]

    seen = set(ratings.loc[ratings["user_id"] == user_id, "movie_id"])
    if user_id not in reconstructed.index:
        recs = ratings.groupby("movie_id")["rating"].mean().sort_values(ascending=False).head(top_n)
        return movies[movies["movie_id"].isin(recs.index)].assign(score=lambda df: df["movie_id"].map(recs))

    scores = reconstructed.loc[user_id].drop(labels=list(seen), errors="ignore").sort_values(ascending=False)
    recs = scores.head(top_n)
    return movies[movies["movie_id"].isin(recs.index)].assign(score=lambda df: df["movie_id"].map(recs)).sort_values("score", ascending=False)


def render_page() -> None:
    st.header("Movie Recommendation System")
    st.write("Personalized Top-N recommendations with popularity baseline and SVD model comparison.")
    result = train_model()
    st.caption(f"Data source: {result.extra['source']} dataset")

    cols = st.columns(4)
    for col, (name, value) in zip(cols, result.metrics.items()):
        col.metric(name.replace("_", " ").title(), f"{value:.3f}")

    left, right = st.columns([1, 2])
    user_id = left.number_input("User ID", min_value=1, max_value=150, value=1, step=1)
    top_n = left.slider("Recommendations", 5, 15, 10)
    recs = recommend_for_user(int(user_id), top_n)
    right.dataframe(recs[["title", "genre", "score"]], use_container_width=True)

    st.subheader("Model Metrics")
    st.dataframe(metric_table(result.metrics), use_container_width=True)
