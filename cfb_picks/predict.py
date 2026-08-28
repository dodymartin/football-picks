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


def compute_edge(predicted_margin, spread):
    market_home_margin = -spread
    return predicted_margin - market_home_margin


def _confidence_score(pick, calibration):
    if not calibration:
        return abs(pick.edge)
    win_rate = calibration.get(edge_bucket(abs(pick.edge)))
    return abs(pick.edge) * win_rate if win_rate is not None else abs(pick.edge)


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
        best = max(picks, key=lambda pick: _confidence_score(pick, calibration))
        best.is_best_pick = True
    return picks
