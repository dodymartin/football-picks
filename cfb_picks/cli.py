# cfb_picks/cli.py
import csv

import click

from .aliases import normalize_team_name
from .backtest import run_backtest, summarize_backtest
from .calibration import load_calibration, save_calibration
from .cfbd_client import CFBDClient
from .config import CALIBRATION_PATH, DB_PATH, MODEL_PATH, SEASONS
from .db import get_connection, init_db
from .elo import compute_elo_history
from .features import build_features
from .history import season_record
from .ingest import fetch_data as fetch_data_impl
from .ingest import upsert_games
from .model import gather_training_data, load_model, predict_margin, save_model, train_model
from .predict import make_picks
from .results import grade_week


@click.group()
def cli():
    pass


@cli.command("fetch-data")
@click.option(
    "--seasons",
    default=None,
    help="Comma-separated seasons for a cheap incremental refresh; defaults to the full configured range.",
)
def fetch_data_cmd(seasons):
    conn = get_connection(DB_PATH)
    init_db(conn)
    years = [int(s) for s in seasons.split(",")] if seasons else SEASONS
    fetch_data_impl(conn, CFBDClient(), years)
    click.echo(f"Fetched data for seasons: {years}")


@cli.command("build-ratings")
def build_ratings_cmd():
    conn = get_connection(DB_PATH)
    init_db(conn)

    compute_elo_history(conn, SEASONS)

    feature_dicts, margins = gather_training_data(conn, SEASONS)
    if not feature_dicts:
        raise click.ClickException("No training data available — has fetch-data been run?")
    weights = train_model(feature_dicts, margins)
    save_model(weights, MODEL_PATH)

    backtest_results = run_backtest(conn, SEASONS)
    calibration = summarize_backtest(backtest_results)["by_edge_bucket"]
    save_calibration(calibration, CALIBRATION_PATH)

    click.echo(f"Ratings + model trained on seasons {SEASONS}, saved to {MODEL_PATH}")
    click.echo(f"Calibration saved to {CALIBRATION_PATH}: {calibration}")


@cli.command("backtest")
def backtest_cmd():
    conn = get_connection(DB_PATH)
    results = run_backtest(conn, SEASONS)
    click.echo(summarize_backtest(results))


@cli.command("predict")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--season", required=True, type=int)
@click.option("--week", required=True, type=int)
def predict_cmd(input_path, season, week):
    conn = get_connection(DB_PATH)
    weights = load_model(MODEL_PATH)
    try:
        calibration = load_calibration(CALIBRATION_PATH)
    except FileNotFoundError:
        calibration = None

    games = []
    with open(input_path, newline="") as f:
        for row in csv.DictReader(f):
            home = normalize_team_name(conn, row["home_team"])
            away = normalize_team_name(conn, row["away_team"])
            spread = float(row["spread"])
            neutral_site = (row.get("neutral_site") or "0").strip().lower() in ("1", "true")
            features = build_features(conn, season, week, home, away, neutral_site)
            predicted_margin = predict_margin(weights, features)
            games.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "spread": spread,
                    "predicted_margin": predicted_margin,
                }
            )

    picks = make_picks(games, calibration=calibration)
    # Clear any stale/corrected ungraded predictions for this exact slate before
    # inserting the new one, so a re-run can't leave duplicate or stale
    # is_best_pick rows behind. Rows that already have a result are left alone.
    conn.execute(
        "DELETE FROM picks WHERE season = ? AND week = ? AND result IS NULL", (season, week)
    )
    for pick in picks:
        conn.execute(
            "INSERT OR IGNORE INTO picks (season, week, home_team, away_team, spread, "
            "predicted_margin, edge, pick_team, is_best_pick, result) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                season,
                week,
                pick.home_team,
                pick.away_team,
                pick.spread,
                pick.predicted_margin,
                pick.edge,
                pick.pick_team,
                int(pick.is_best_pick),
            ),
        )
    conn.commit()

    for pick in sorted(picks, key=lambda p: abs(p.edge), reverse=True):
        marker = " *** BEST PICK ***" if pick.is_best_pick else ""
        click.echo(
            f"{pick.away_team} @ {pick.home_team} ({pick.spread:+}): "
            f"pick {pick.pick_team} (edge {pick.edge:+.1f}){marker}"
        )


@cli.command("record-results")
@click.option("--season", required=True, type=int)
@click.option("--week", required=True, type=int)
def record_results_cmd(season, week):
    conn = get_connection(DB_PATH)
    client = CFBDClient()
    upsert_games(conn, client.get_games(season))
    count = grade_week(conn, season, week)
    if count == 0:
        click.echo(
            "No picks were graded — check that games have completed and week/season match"
        )
    else:
        click.echo(f"Graded {count} picks for week {week}, {season}")


@cli.command("history")
@click.option("--season", default=None, type=int)
def history_cmd(season):
    conn = get_connection(DB_PATH)
    click.echo(season_record(conn, season))


def main():
    cli()


if __name__ == "__main__":
    main()
