import json
from dataclasses import asdict, dataclass

from sklearn.linear_model import Ridge

from .elo import is_fbs_team
from .features import build_features

FEATURE_ORDER = ["elo_diff", "success_rate_diff", "ppa_diff", "neutral_site"]


@dataclass
class ModelWeights:
    intercept: float
    coefficients: dict


def train_model(feature_dicts, margins, alpha=1.0):
    x = [[features[name] for name in FEATURE_ORDER] for features in feature_dicts]
    ridge = Ridge(alpha=alpha)
    ridge.fit(x, margins)
    coefficients = {name: float(coef) for name, coef in zip(FEATURE_ORDER, ridge.coef_)}
    return ModelWeights(intercept=float(ridge.intercept_), coefficients=coefficients)


def predict_margin(weights, features):
    total = weights.intercept
    for name, coef in weights.coefficients.items():
        total += coef * features.get(name, 0.0)
    return total


def save_model(weights, path):
    with open(path, "w") as f:
        json.dump(asdict(weights), f)


def load_model(path):
    with open(path) as f:
        data = json.load(f)
    return ModelWeights(intercept=data["intercept"], coefficients=data["coefficients"])


def gather_training_data(conn, seasons):
    placeholders = ",".join("?" * len(seasons))
    games = conn.execute(
        f"SELECT * FROM games WHERE season IN ({placeholders}) AND completed = 1 "
        "ORDER BY season, week",
        seasons,
    ).fetchall()

    feature_dicts, margins = [], []
    for game in games:
        if not (is_fbs_team(conn, game["home_team"]) and is_fbs_team(conn, game["away_team"])):
            continue
        features = build_features(
            conn,
            game["season"],
            game["week"],
            game["home_team"],
            game["away_team"],
            bool(game["neutral_site"]),
        )
        feature_dicts.append(features)
        margins.append(game["home_points"] - game["away_points"])
    return feature_dicts, margins
