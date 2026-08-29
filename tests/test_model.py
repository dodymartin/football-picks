import pytest

from cfb_picks.db import get_connection, init_db
from cfb_picks.model import (
    FEATURE_ORDER,
    gather_training_data,
    load_model,
    predict_margin,
    save_model,
    train_model,
)


def test_train_and_predict_recovers_simple_relationship():
    feature_dicts = [
        {"elo_diff": 100, "success_rate_diff": 0, "ppa_diff": 0, "neutral_site": 0},
        {"elo_diff": -100, "success_rate_diff": 0, "ppa_diff": 0, "neutral_site": 0},
        {"elo_diff": 200, "success_rate_diff": 0, "ppa_diff": 0, "neutral_site": 0},
        {"elo_diff": 0, "success_rate_diff": 0, "ppa_diff": 0, "neutral_site": 0},
    ]
    margins = [10, -10, 20, 0]

    weights = train_model(feature_dicts, margins, alpha=0.001)

    assert weights.coefficients["elo_diff"] == pytest.approx(0.1, abs=0.02)
    predicted = predict_margin(
        weights, {"elo_diff": 150, "success_rate_diff": 0, "ppa_diff": 0, "neutral_site": 0}
    )
    assert predicted == pytest.approx(15, abs=3)


def test_save_and_load_model_roundtrip(tmp_path):
    weights = train_model(
        [{"elo_diff": 100, "success_rate_diff": 0.1, "ppa_diff": 0.2, "neutral_site": 0}],
        [10],
    )
    path = tmp_path / "model.json"

    save_model(weights, path)
    loaded = load_model(path)

    assert loaded.intercept == pytest.approx(weights.intercept)
    assert loaded.coefficients == pytest.approx(weights.coefficients)


def test_gather_training_data_builds_features_and_margins(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('A', 'fbs'), ('B', 'fbs')")
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (1, 2024, 1, 'regular', NULL, 1, 0, 'A', 'B', 30, 10)"
    )
    conn.commit()

    feature_dicts, margins = gather_training_data(conn, [2024])

    assert margins == [20]
    assert set(feature_dicts[0].keys()) == set(FEATURE_ORDER)


def test_gather_training_data_excludes_fbs_vs_fcs_games(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('A', 'fbs')")
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (1, 2024, 1, 'regular', NULL, 1, 0, 'A', 'Some FCS School', 50, 3)"
    )
    conn.commit()

    feature_dicts, margins = gather_training_data(conn, [2024])

    assert margins == []
    assert feature_dicts == []
