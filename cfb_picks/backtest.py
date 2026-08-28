# cfb_picks/backtest.py
from .calibration import edge_bucket
from .elo import is_fbs_team
from .features import build_features
from .model import gather_training_data, predict_margin, train_model
from .predict import compute_edge
from .results import grade_ats


def _market_spread(conn, game_id):
    row = conn.execute(
        "SELECT spread FROM betting_lines WHERE game_id = ? AND provider = 'consensus'",
        (game_id,),
    ).fetchone()
    return row["spread"] if row else None


def _evaluate_season(conn, weights, season):
    games = conn.execute(
        "SELECT * FROM games WHERE season = ? AND completed = 1 ORDER BY week", (season,)
    ).fetchall()

    results = []
    for game in games:
        if not (is_fbs_team(conn, game["home_team"]) and is_fbs_team(conn, game["away_team"])):
            continue

        spread = _market_spread(conn, game["id"])
        if spread is None:
            continue

        features = build_features(
            conn,
            game["season"],
            game["week"],
            game["home_team"],
            game["away_team"],
            bool(game["neutral_site"]),
        )
        predicted_margin = predict_margin(weights, features)
        edge = compute_edge(predicted_margin, spread)

        actual_margin = game["home_points"] - game["away_points"]
        picked_home = edge > 0
        grade = grade_ats(actual_margin, spread, picked_home)
        if grade == "push":
            continue
        results.append({"edge": edge, "correct": grade == "win"})

    return results


def run_backtest(conn, seasons):
    sorted_seasons = sorted(seasons)
    results = []
    for index, season in enumerate(sorted_seasons):
        training_seasons = sorted_seasons[:index]
        if not training_seasons:
            continue  # no prior seasons to train on; can't evaluate out-of-sample

        feature_dicts, margins = gather_training_data(conn, training_seasons)
        if not feature_dicts:
            continue

        weights = train_model(feature_dicts, margins)
        results.extend(_evaluate_season(conn, weights, season))

    return results


def summarize_backtest(results):
    if not results:
        return {"overall_accuracy": None, "by_edge_bucket": {}, "n": 0}

    overall = sum(r["correct"] for r in results) / len(results)
    buckets = {}
    for r in results:
        buckets.setdefault(edge_bucket(abs(r["edge"])), []).append(r["correct"])

    by_bucket = {
        name: (sum(values) / len(values) if values else None)
        for name, values in buckets.items()
    }
    return {"overall_accuracy": overall, "by_edge_bucket": by_bucket, "n": len(results)}
