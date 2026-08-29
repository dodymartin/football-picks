import math

import pytest

from cfb_picks.db import get_connection, init_db
from cfb_picks.elo import (
    BASELINE_RATING,
    FCS_BASELINE_RATING,
    compute_elo_history,
    expected_win_prob,
    get_rating_as_of,
    is_fbs_team,
    mov_multiplier,
    regress_to_mean,
    update_ratings,
)
from cfb_picks.ingest import upsert_teams

GAME_COLUMNS = (
    "id, season, week, season_type, start_date, completed, neutral_site, "
    "home_team, away_team, home_points, away_points"
)


def _insert_game(conn, row):
    conn.execute(f"INSERT INTO games ({GAME_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)


def test_expected_win_prob_equal_ratings_is_half():
    assert expected_win_prob(0) == pytest.approx(0.5)


def test_mov_multiplier_increases_with_margin():
    assert mov_multiplier(28) > mov_multiplier(3)


def test_update_ratings_home_win_neutral_site():
    new_home, new_away = update_ratings(1500, 1500, 31, 17, neutral_site=True, k=20, home_field=65)
    expected_change = 20 * math.log(15) * 0.5
    assert new_home == pytest.approx(1500 + expected_change)
    assert new_away == pytest.approx(1500 - expected_change)


def test_update_ratings_accounts_for_home_field():
    new_home_neutral, _ = update_ratings(1500, 1500, 24, 20, neutral_site=True, k=20, home_field=65)
    new_home_with_field, _ = update_ratings(1500, 1500, 24, 20, neutral_site=False, k=20, home_field=65)
    assert new_home_with_field < new_home_neutral


def test_regress_to_mean_pulls_toward_baseline():
    assert regress_to_mean(1700, factor=0.66, mean=BASELINE_RATING) == pytest.approx(1500 + 0.66 * 200)


def test_compute_elo_history_updates_ratings(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO teams (school, classification) VALUES ('Ohio State', 'fbs'), ('Akron', 'fbs')"
    )
    _insert_game(conn, (1, 2024, 1, "regular", None, 1, 0, "Ohio State", "Akron", 52, 6))
    conn.commit()

    compute_elo_history(conn, [2024])

    assert get_rating_as_of(conn, 2024, 2, "Ohio State") > BASELINE_RATING
    assert get_rating_as_of(conn, 2024, 2, "Akron") < BASELINE_RATING


def test_compute_elo_history_skips_completed_games_with_missing_scores(tmp_path):
    # CFBD occasionally marks obscure games completed=1 with a null score
    # (missing stats rather than an in-progress game); these must not crash
    # the Elo pass or be treated as a 0-0 result.
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO teams (school, classification) VALUES ('Ohio State', 'fbs'), ('Akron', 'fbs')"
    )
    _insert_game(conn, (1, 2024, 1, "regular", None, 1, 0, "Ohio State", "Akron", None, 6))
    conn.commit()

    compute_elo_history(conn, [2024])

    assert get_rating_as_of(conn, 2024, 2, "Ohio State") == BASELINE_RATING
    assert get_rating_as_of(conn, 2024, 2, "Akron") == BASELINE_RATING


def test_compute_elo_history_starts_non_fbs_opponents_at_lower_baseline(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('Ohio State', 'fbs')")
    # 'Some FCS School' has no teams row, so it's treated as non-FBS.
    _insert_game(conn, (1, 2024, 1, "regular", None, 1, 0, "Ohio State", "Some FCS School", 45, 3))
    conn.commit()

    compute_elo_history(conn, [2024])

    preseason_row = conn.execute(
        "SELECT rating FROM elo_ratings WHERE season = 2024 AND week = 0 AND team = 'Some FCS School'"
    ).fetchone()
    assert preseason_row["rating"] == FCS_BASELINE_RATING


def test_is_fbs_team_treats_mixed_case_classification_as_fbs(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('Ohio State', 'FBS')")
    conn.commit()
    assert is_fbs_team(conn, "Ohio State") is True


def test_is_fbs_team_treats_team_ingested_from_fbs_endpoint_with_missing_classification_as_fbs(
    tmp_path,
):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    # Simulates ingesting a /teams/fbs response where CFBD omitted the
    # (nullable) classification field; upsert_teams should default it to
    # "fbs" and is_fbs_team should recognize it.
    upsert_teams(conn, [{"school": "Ohio State", "conference": "Big Ten"}])
    assert is_fbs_team(conn, "Ohio State") is True


def test_get_rating_as_of_returns_fbs_baseline_for_known_fbs_team(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('Ohio State', 'fbs')")
    conn.commit()
    assert get_rating_as_of(conn, 2024, 1, "Ohio State") == BASELINE_RATING


def test_get_rating_as_of_returns_fcs_baseline_for_unknown_team(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    assert get_rating_as_of(conn, 2024, 1, "Random FCS School") == FCS_BASELINE_RATING


def test_get_rating_as_of_carries_prior_season_forward(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO teams (school, classification) VALUES ('Ohio State', 'fbs'), ('Akron', 'fbs')"
    )
    _insert_game(conn, (1, 2023, 15, "regular", None, 1, 0, "Ohio State", "Akron", 52, 6))
    conn.commit()

    compute_elo_history(conn, [2023])

    rating_end_2023 = get_rating_as_of(conn, 2023, 16, "Ohio State")
    rating_2024 = get_rating_as_of(conn, 2024, 1, "Ohio State")
    assert rating_2024 == pytest.approx(rating_end_2023)
