import pytest

from cfb_picks.db import get_connection, init_db
from cfb_picks.ingest import (
    extract_consensus_spread,
    fetch_data,
    upsert_betting_lines,
    upsert_games,
    upsert_team_game_stats,
    upsert_teams,
)

SAMPLE_GAME = {
    "id": 401520145,
    "season": 2024,
    "week": 1,
    "seasonType": "regular",
    "startDate": "2024-08-31T00:00:00.000Z",
    "completed": True,
    "neutralSite": False,
    "homeTeam": "Ohio State",
    "awayTeam": "Akron",
    "homePoints": 52,
    "awayPoints": 6,
}


def _conn(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    return conn


def test_upsert_teams(tmp_path):
    conn = _conn(tmp_path)
    upsert_teams(conn, [{"school": "Ohio State", "conference": "Big Ten", "classification": "fbs"}])
    row = conn.execute("SELECT * FROM teams WHERE school = 'Ohio State'").fetchone()
    assert row["conference"] == "Big Ten"


def test_upsert_games_is_idempotent(tmp_path):
    conn = _conn(tmp_path)
    upsert_games(conn, [SAMPLE_GAME])
    upsert_games(conn, [SAMPLE_GAME])
    row = conn.execute("SELECT * FROM games WHERE id = ?", (401520145,)).fetchone()
    assert row["home_team"] == "Ohio State"
    assert row["home_points"] == 52
    assert conn.execute("SELECT COUNT(*) c FROM games").fetchone()["c"] == 1


def test_extract_consensus_spread_averages_providers():
    entry = {"lines": [{"provider": "DraftKings", "spread": -45.5}, {"provider": "Bovada", "spread": -44.0}]}
    assert extract_consensus_spread(entry) == pytest.approx(-44.75)


def test_extract_consensus_spread_none_when_missing():
    assert extract_consensus_spread({"lines": []}) is None


def test_upsert_betting_lines(tmp_path):
    conn = _conn(tmp_path)
    upsert_games(conn, [SAMPLE_GAME])
    upsert_betting_lines(
        conn, [{"id": 401520145, "lines": [{"provider": "DraftKings", "spread": -45.5}]}]
    )
    row = conn.execute(
        "SELECT spread FROM betting_lines WHERE game_id = ?", (401520145,)
    ).fetchone()
    assert row["spread"] == -45.5


def test_upsert_team_game_stats(tmp_path):
    conn = _conn(tmp_path)
    upsert_games(conn, [SAMPLE_GAME])
    upsert_team_game_stats(
        conn,
        [
            {
                "gameId": 401520145,
                "team": "Ohio State",
                "offense": {"successRate": 0.55, "ppa": 0.42},
            }
        ],
    )
    row = conn.execute(
        "SELECT success_rate, ppa FROM team_game_stats WHERE game_id = ?", (401520145,)
    ).fetchone()
    assert row["success_rate"] == 0.55
    assert row["ppa"] == 0.42


def test_fetch_data_calls_client_and_persists_everything(tmp_path):
    conn = _conn(tmp_path)

    class FakeClient:
        def get_fbs_teams(self, year):
            return [{"school": "Ohio State", "abbreviation": "OSU"}]

        def get_games(self, year):
            return [SAMPLE_GAME]

        def get_lines(self, year):
            return [{"id": 401520145, "lines": [{"provider": "DraftKings", "spread": -45.5}]}]

        def get_advanced_game_stats(self, year):
            return [
                {
                    "gameId": 401520145,
                    "team": "Ohio State",
                    "offense": {"successRate": 0.55, "ppa": 0.42},
                }
            ]

    fetch_data(conn, FakeClient(), [2024])

    assert conn.execute("SELECT COUNT(*) c FROM games").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM betting_lines").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM team_game_stats").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM team_aliases").fetchone()["c"] >= 2
