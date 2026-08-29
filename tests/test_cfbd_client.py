from unittest.mock import MagicMock

import pytest

from cfb_picks.cfbd_client import CFBDClient


def _client_with_fake_session(payload):
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    session.get.return_value = response
    return CFBDClient(api_key="test-key", session=session), session


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        CFBDClient(session=MagicMock())


def test_get_games_calls_correct_endpoint():
    client, session = _client_with_fake_session([{"id": 1, "season": 2024}])

    result = client.get_games(2024)

    session.get.assert_called_once_with(
        "https://api.collegefootballdata.com/games",
        params={"year": 2024, "seasonType": "regular"},
        headers={"Authorization": "Bearer test-key"},
        timeout=30,
    )
    assert result == [{"id": 1, "season": 2024}]


def test_get_fbs_teams_calls_correct_endpoint():
    client, session = _client_with_fake_session([{"school": "Ohio State"}])

    result = client.get_fbs_teams(2024)

    session.get.assert_called_once_with(
        "https://api.collegefootballdata.com/teams/fbs",
        params={"year": 2024},
        headers={"Authorization": "Bearer test-key"},
        timeout=30,
    )
    assert result == [{"school": "Ohio State"}]


def test_get_lines_and_advanced_stats_use_season_type_param():
    client, session = _client_with_fake_session([])

    client.get_lines(2024)
    session.get.assert_called_with(
        "https://api.collegefootballdata.com/lines",
        params={"year": 2024, "seasonType": "regular"},
        headers={"Authorization": "Bearer test-key"},
        timeout=30,
    )

    client.get_advanced_game_stats(2024)
    session.get.assert_called_with(
        "https://api.collegefootballdata.com/stats/game/advanced",
        params={"year": 2024, "seasonType": "regular"},
        headers={"Authorization": "Bearer test-key"},
        timeout=30,
    )
