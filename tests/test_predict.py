# tests/test_predict.py
import pytest

from cfb_picks.predict import compute_edge, make_picks, rank_by_confidence


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
        {"home_team": "C", "away_team": "D", "spread": -1, "predicted_margin": 10},  # edge 9 -> "8-15"
    ]
    calibration = {"5-8": 0.9, "8-15": 0.3}

    picks = make_picks(games, calibration=calibration)

    # Raw edge favors the second game (9 > 6), but win rate is the primary
    # ranking key, and calibrated win rate (90% vs 30%) favors the first.
    assert picks[0].is_best_pick is True
    assert picks[1].is_best_pick is False


def test_rank_by_confidence_orders_highest_confidence_first():
    games = [
        {"home_team": "A", "away_team": "B", "spread": -1, "predicted_margin": 7},  # edge 6 -> "5-8"
        {"home_team": "C", "away_team": "D", "spread": -1, "predicted_margin": 10},  # edge 9 -> "8-15"
    ]
    calibration = {"5-8": 0.9, "8-15": 0.3}

    picks = make_picks(games, calibration=calibration)
    ranked = rank_by_confidence(picks, calibration)

    # Raw edge favors C@D (9 > 6), but calibrated win rate (90% vs 30%)
    # favors A@B, so it should rank first.
    assert [p.home_team for p in ranked] == ["A", "C"]


def test_rank_by_confidence_breaks_ties_within_bucket_by_edge_size():
    games = [
        {"home_team": "A", "away_team": "B", "spread": 0, "predicted_margin": 1},  # edge 1 -> "0-2"
        {"home_team": "C", "away_team": "D", "spread": 0, "predicted_margin": 1.8},  # edge 1.8 -> "0-2"
    ]
    calibration = {"0-2": 0.55}

    picks = make_picks(games, calibration=calibration)
    ranked = rank_by_confidence(picks, calibration)

    # Both picks share the same calibrated win rate (same bucket), so the
    # larger raw edge should break the tie and rank first.
    assert [p.home_team for p in ranked] == ["C", "A"]
    assert picks[1].is_best_pick is True  # C@D, the larger-edge pick in the tie


def test_make_picks_uses_neutral_default_for_edge_bucket_missing_from_calibration():
    games = [
        {"home_team": "A", "away_team": "B", "spread": -1, "predicted_margin": 7},  # edge 6 -> "5-8"
        {"home_team": "C", "away_team": "D", "spread": -1, "predicted_margin": 10},  # edge 9 -> "8-15"
    ]
    # "8-15" never showed up in the backtest, so it's absent from calibration —
    # not corrupted data, just a normal, sparsely-populated calibration file.
    calibration = {"5-8": 0.9}

    picks = make_picks(games, calibration=calibration)

    # A@B's calibrated win rate (90%) beats C@D's neutral-default win rate
    # (50%), so A@B should win regardless of its smaller raw edge.
    assert picks[0].is_best_pick is True
    assert picks[1].is_best_pick is False
