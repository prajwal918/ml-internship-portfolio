from ml_portfolio.projects import churn, employee, fake_news, movie, stock


def test_movie_recommendations_return_rows():
    recs = movie.recommend_for_user(1, top_n=5)
    assert len(recs) == 5
    assert {"title", "genre", "score"}.issubset(recs.columns)


def test_churn_prediction_probability_range():
    df = churn.sample_data().iloc[0]
    record = {col: df[col] for col in churn.NUMERIC + churn.CATEGORICAL}
    prob = churn.predict_customer(record)
    assert 0 <= prob <= 1


def test_stock_model_metrics_exist():
    result = stock.train_model()
    assert result.metrics["final_f1"] >= 0
    assert not result.extra["scored"].empty


def test_fake_news_prediction_probability_range():
    label, prob = fake_news.predict_text("A viral post claims a shocking secret without evidence.")
    assert label in {"Likely Fake", "Likely Real"}
    assert 0 <= prob <= 1


def test_employee_prediction_probability_range():
    df = employee.sample_data().iloc[0]
    record = {col: df[col] for col in employee.NUMERIC + employee.CATEGORICAL}
    prob = employee.predict_employee(record)
    assert 0 <= prob <= 1
