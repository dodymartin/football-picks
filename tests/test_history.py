import pytest

from cfb_picks.db import get_connection, init_db
from cfb_picks.history import season_record


def test_season_record_computes_points_and_pct(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    rows = [
        (2024, 1, "A", "B", -3, 10, 7, "A", 1, "win"),
        (2024, 1, "C", "D", 3, -1, -4, "D", 0, "loss"),
        (2024, 2, "E", "F", -3, 10, 7, "E", 0, "win"),
        (2024, 2, "G", "H", -3, 10, 7, "G", 0, None),
    ]
    conn.executemany(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    record = season_record(conn, 2024)

    assert record["picks"] == 3
    assert record["wins"] == 2
    assert record["points"] == 3
    assert record["ats_pct"] == pytest.approx(2 / 3)
    assert record["best_pick_record"] == "1/1"


def test_season_record_handles_push_without_double_counting(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    rows = [
        (2024, 1, "A", "B", -3, 10, 7, "A", 1, "win"),
        # A push: counts toward games played but not wins/losses/points, and
        # doesn't count toward the best-pick denominator either.
        (2024, 1, "C", "D", -3, 5, 8, "C", 1, "push"),
        (2024, 2, "E", "F", -3, 10, 7, "F", 0, "loss"),
    ]
    conn.executemany(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    record = season_record(conn, 2024)

    assert record["picks"] == 3
    assert record["wins"] == 1
    assert record["points"] == 2
    assert record["ats_pct"] == pytest.approx(1 / 2)
    assert record["best_pick_record"] == "1/1"
