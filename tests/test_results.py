from cfb_picks.db import get_connection, init_db
from cfb_picks.results import grade_ats, grade_week


def test_grade_ats_exact_push_for_home_side_pick():
    # actual_margin == -spread exactly: neither side covers.
    assert grade_ats(actual_margin=3, spread=-3, picked_home=True) == "push"


def test_grade_ats_exact_push_for_away_side_pick():
    assert grade_ats(actual_margin=3, spread=-3, picked_home=False) == "push"


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

    graded_count = grade_week(conn, 2024, 1)

    win_row = conn.execute("SELECT result FROM picks WHERE home_team = 'A'").fetchone()
    loss_row = conn.execute("SELECT result FROM picks WHERE home_team = 'C'").fetchone()
    assert win_row["result"] == "win"
    assert loss_row["result"] == "loss"
    assert graded_count == 2


def test_grade_week_skips_ungraded_games(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) "
        "VALUES (2024, 1, 'A', 'B', -3, 10, 7, 'A', 1, NULL)"
    )
    conn.commit()

    graded_count = grade_week(conn, 2024, 1)

    row = conn.execute("SELECT result FROM picks WHERE home_team = 'A'").fetchone()
    assert row["result"] is None
    assert graded_count == 0


def test_grade_week_grades_exact_push(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (1, 2024, 1, 'regular', NULL, 1, 0, 'A', 'B', 20, 17)"
    )
    conn.execute(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) "
        "VALUES (2024, 1, 'A', 'B', -3, 10, 7, 'A', 1, NULL)"
    )
    conn.commit()

    graded_count = grade_week(conn, 2024, 1)

    row = conn.execute("SELECT result FROM picks WHERE home_team = 'A'").fetchone()
    assert row["result"] == "push"
    assert graded_count == 1
