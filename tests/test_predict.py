# tests/test_predict.py
import pytest

from cfb_picks.predict import compute_edge, make_picks


def test_compute_edge_home_favorite():
    assert compute_edge(predicted_margin=10, spread=-3) == pytest.approx(7)


def test_compute_edge_home_underdog():
    assert compute_edge(predicted_margin=-1, spread=6) == pytest.approx(5)


def test_make_picks_selects_side_and_best_pick_by_raw_edge_without_calibration():
    games = [
        {"home_team": "A", "away_team": "B", "spread": -3, "predicted_margin": 10},
        {"home_team": "C", "away_team": "D", "spread": 6, "predicted_margin": -1},
    ]

    picks = make_picks(games)

    assert picks[0].pick_team == "A"
    assert picks[0].edge == pytest.approx(7)
    assert picks[1].pick_team == "C"
    assert picks[1].edge == pytest.approx(5)
    assert picks[0].is_best_pick is True
    assert picks[1].is_best_pick is False


def test_make_picks_best_pick_uses_calibration_when_provided():
    games = [
        {"home_team": "A", "away_team": "B", "spread": -1, "predicted_margin": 7},  # edge 6 -> "5-8"
        {"home_team": "C", "away_team": "D", "spread": -1, "predicted_margin": 10},  # edge 9 -> "8+"
    ]
    calibration = {"5-8": 0.9, "8+": 0.3}

    picks = make_picks(games, calibration=calibration)

    # Raw edge favors the second game (9 > 6), but calibrated confidence
    # (6*0.9=5.4 vs 9*0.3=2.7) favors the first.
    assert picks[0].is_best_pick is True
    assert picks[1].is_best_pick is False
