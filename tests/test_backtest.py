# tests/test_backtest.py
import pytest

from cfb_picks.backtest import run_backtest, summarize_backtest
from cfb_picks.db import get_connection, init_db
from cfb_picks.elo import compute_elo_history

GAME_COLUMNS = (
    "id, season, week, season_type, start_date, completed, neutral_site, "
    "home_team, away_team, home_points, away_points"
)


def _insert_game(conn, row):
    conn.execute(f"INSERT INTO games ({GAME_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)


def test_run_backtest_trains_only_on_prior_seasons(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('A', 'fbs'), ('B', 'fbs')")
    _insert_game(conn, (1, 2023, 1, "regular", None, 1, 0, "A", "B", 50, 0))
    _insert_game(conn, (2, 2024, 1, "regular", None, 1, 0, "A", "B", 30, 10))
    conn.execute("INSERT INTO betting_lines (game_id, provider, spread) VALUES (2, 'consensus', -3)")
    conn.commit()
    compute_elo_history(conn, [2023, 2024])

    results = run_backtest(conn, [2023, 2024])

    # 2023 has no prior season to train on and is skipped; only 2024 is evaluated.
    assert len(results) == 1


def test_run_backtest_skips_push_games_without_crashing(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('A', 'fbs'), ('B', 'fbs')")
    _insert_game(conn, (1, 2023, 1, "regular", None, 1, 0, "A", "B", 50, 0))
    # Exact push: actual margin (3) equals -spread (3).
    _insert_game(conn, (2, 2024, 1, "regular", None, 1, 0, "A", "B", 20, 17))
    conn.execute("INSERT INTO betting_lines (game_id, provider, spread) VALUES (2, 'consensus', -3)")
    conn.commit()
    compute_elo_history(conn, [2023, 2024])

    results = run_backtest(conn, [2023, 2024])

    # The 2024 game is a push and must be excluded rather than crashing or
    # being counted as a loss.
    assert results == []


def test_run_backtest_skips_games_without_a_market_spread(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('A', 'fbs'), ('B', 'fbs')")
    _insert_game(conn, (1, 2023, 1, "regular", None, 1, 0, "A", "B", 50, 0))
    _insert_game(conn, (2, 2024, 1, "regular", None, 1, 0, "A", "B", 30, 10))
    conn.commit()
    compute_elo_history(conn, [2023, 2024])

    results = run_backtest(conn, [2023, 2024])

    assert results == []


def test_summarize_backtest_buckets_by_edge_size():
    results = [
        {"edge": 1, "correct": True},
        {"edge": 9, "correct": True},
        {"edge": 9, "correct": False},
    ]

    summary = summarize_backtest(results)

    assert summary["overall_accuracy"] == pytest.approx(2 / 3)
    assert summary["by_edge_bucket"]["0-2"] == pytest.approx(1.0)
    assert summary["by_edge_bucket"]["8-15"] == pytest.approx(0.5)
    assert summary["n"] == 3


def test_summarize_backtest_handles_empty_results():
    summary = summarize_backtest([])
    assert summary["overall_accuracy"] is None
