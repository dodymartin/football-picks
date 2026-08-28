from cfb_picks.db import get_connection, init_db
from cfb_picks.results import grade_week


def test_grade_week_marks_win_and_loss(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (1, 2024, 1, 'regular', NULL, 1, 0, 'A', 'B', 30, 10)"
    )
    conn.execute(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) "
        "VALUES (2024, 1, 'A', 'B', -3, 10, 7, 'A', 1, NULL)"
    )
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (2, 2024, 1, 'regular', NULL, 1, 0, 'C', 'D', 10, 30)"
    )
    conn.execute(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) "
        "VALUES (2024, 1, 'C', 'D', -3, 5, 8, 'C', 0, NULL)"
    )
    conn.commit()

    grade_week(conn, 2024, 1)

    win_row = conn.execute("SELECT result FROM picks WHERE home_team = 'A'").fetchone()
    loss_row = conn.execute("SELECT result FROM picks WHERE home_team = 'C'").fetchone()
    assert win_row["result"] == "win"
    assert loss_row["result"] == "loss"


def test_grade_week_skips_ungraded_games(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) "
        "VALUES (2024, 1, 'A', 'B', -3, 10, 7, 'A', 1, NULL)"
    )
    conn.commit()

    grade_week(conn, 2024, 1)

    row = conn.execute("SELECT result FROM picks WHERE home_team = 'A'").fetchone()
    assert row["result"] is None
