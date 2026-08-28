import pytest

from cfb_picks.db import get_connection, init_db
from cfb_picks.features import build_features


def test_build_features_uses_only_prior_data(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (1, 2024, 1, 'regular', NULL, 1, 0, 'A', 'B', 30, 10)"
    )
    conn.execute(
        "INSERT INTO team_game_stats (game_id, team, success_rate, ppa) VALUES (1, 'A', 0.6, 0.5)"
    )
    conn.execute(
        "INSERT INTO elo_ratings (season, week, team, rating) VALUES "
        "(2024, 0, 'A', 1600), (2024, 0, 'B', 1400)"
    )
    conn.commit()

    features_week2 = build_features(conn, 2024, 2, "A", "B")
    assert features_week2["elo_diff"] == pytest.approx(200)
    assert features_week2["success_rate_diff"] == pytest.approx(0.6)
    assert features_week2["ppa_diff"] == pytest.approx(0.5)
    assert features_week2["neutral_site"] == 0.0

    # Week 1's own stats must not count toward its own pre-game features.
    features_week1 = build_features(conn, 2024, 1, "A", "B")
    assert features_week1["success_rate_diff"] == pytest.approx(0.0)
    assert features_week1["ppa_diff"] == pytest.approx(0.0)


def test_build_features_excludes_null_efficiency_stats(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (1, 2024, 1, 'regular', NULL, 1, 0, 'A', 'B', 30, 10)"
    )
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (2, 2024, 2, 'regular', NULL, 1, 0, 'A', 'C', 20, 10)"
    )
    # Week 1's stats for 'A' are missing (NULL) and must be excluded from the
    # trailing average rather than crashing the sum with a TypeError.
    conn.execute(
        "INSERT INTO team_game_stats (game_id, team, success_rate, ppa) VALUES (1, 'A', NULL, NULL)"
    )
    conn.execute(
        "INSERT INTO team_game_stats (game_id, team, success_rate, ppa) VALUES (2, 'A', 0.6, 0.5)"
    )
    conn.commit()

    features = build_features(conn, 2024, 3, "A", "B")

    assert features["success_rate_diff"] == pytest.approx(0.6)
    assert features["ppa_diff"] == pytest.approx(0.5)


def test_build_features_passes_through_neutral_site(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    features = build_features(conn, 2024, 1, "A", "B", neutral_site=True)
    assert features["neutral_site"] == 1.0
