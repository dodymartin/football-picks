from dataclasses import dataclass

from .calibration import edge_bucket


@dataclass
class Pick:
    home_team: str
    away_team: str
    spread: float
    predicted_margin: float
    edge: float
    pick_team: str
    is_best_pick: bool


DEFAULT_WIN_RATE = 0.5


def compute_edge(predicted_margin, spread):
    market_home_margin = -spread
    return predicted_margin - market_home_margin


def confidence_score(pick, calibration):
    # Historical win rate for picks of this edge size, as a percentage.
    # Without a calibration table there's nothing to look up, so fall back to
    # raw edge size (bigger disagreement with the market = more "confidence").
    if not calibration:
        return abs(pick.edge)
    win_rate = calibration.get(edge_bucket(abs(pick.edge)), DEFAULT_WIN_RATE)
    return win_rate * 100


def _rank_key(pick, calibration):
    # Rank primarily by calibrated win rate — how often this edge size has
    # actually covered historically — since edge size alone is a weak
    # predictor of win rate outside the largest-edge bucket. Break ties
    # (e.g. every pick in the same bucket) by raw edge size so picks within
    # a bucket aren't left in arbitrary order.
    return (confidence_score(pick, calibration), abs(pick.edge))


def rank_by_confidence(picks, calibration=None):
    return sorted(picks, key=lambda pick: _rank_key(pick, calibration), reverse=True)


def make_picks(games, calibration=None):
    picks = []
    for game in games:
        edge = compute_edge(game["predicted_margin"], game["spread"])
        pick_team = game["home_team"] if edge > 0 else game["away_team"]
        picks.append(
            Pick(
                home_team=game["home_team"],
                away_team=game["away_team"],
                spread=game["spread"],
                predicted_margin=game["predicted_margin"],
                edge=edge,
                pick_team=pick_team,
                is_best_pick=False,
            )
        )
    if picks:
        best = max(picks, key=lambda pick: _rank_key(pick, calibration))
        best.is_best_pick = True
    return picks
