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
from .predict import Pick, confidence_score, make_picks, rank_by_confidence
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
    with open(input_path, newline="", encoding="utf-8") as f:
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

    # make_picks() flags a local best pick among just this CSV's games, but
    # that decision is discarded and recomputed below across the whole week's
    # ungraded picks (or left alone if the week is already locked).
    picks = make_picks(games, calibration=calibration)

    # Remember each game's current is_best_pick flag before it's deleted below,
    # so that if the week turns out to be locked (see below), a correction to
    # the already-locked-in game doesn't lose its flag.
    prior_best_pick = {}
    for pick in picks:
        row = conn.execute(
            "SELECT is_best_pick FROM picks WHERE season = ? AND week = ? "
            "AND home_team = ? AND away_team = ? AND result IS NULL",
            (season, week, pick.home_team, pick.away_team),
        ).fetchone()
        if row:
            prior_best_pick[(pick.home_team, pick.away_team)] = bool(row["is_best_pick"])

    # Scope the DELETE to only the games actually present in this CSV, and only
    # to ungraded rows. This lets a partial re-predict (e.g. one corrected game)
    # replace just that game without wiping other ungraded predictions for the
    # same week that this run doesn't touch. Graded rows are never deleted.
    for pick in picks:
        conn.execute(
            "DELETE FROM picks WHERE season = ? AND week = ? AND home_team = ? "
            "AND away_team = ? AND result IS NULL",
            (season, week, pick.home_team, pick.away_team),
        )

    # The "best pick" is locked in once any pick for the week has been graded —
    # the real contest rule is that best pick is decided before the week's
    # games are played. If any pick for this week is already graded, this run
    # must not (re)assign a new best pick, regardless of which game it is.
    locked_best_pick = conn.execute(
        "SELECT 1 FROM picks WHERE season = ? AND week = ? AND result IS NOT NULL",
        (season, week),
    ).fetchone()

    if locked_best_pick:
        # Leave every other ungraded row's flag untouched; only restore each
        # of this run's own (just-deleted) games to whatever it was before.
        for pick in picks:
            pick.is_best_pick = prior_best_pick.get((pick.home_team, pick.away_team), False)
        click.echo(
            "Note: this week's best pick is already locked in from a graded "
            "result; not assigning a new best pick."
        )
    else:
        # Ungraded rows for this week not touched by the current run (games
        # from a previous partial predict that this run isn't replacing). The
        # per-game DELETE above already removed this run's own ungraded rows,
        # so anything left here belongs to other games.
        other_rows = conn.execute(
            "SELECT * FROM picks WHERE season = ? AND week = ? AND result IS NULL",
            (season, week),
        ).fetchall()
        other_picks = [
            Pick(
                home_team=row["home_team"],
                away_team=row["away_team"],
                spread=row["spread"],
                predicted_margin=row["predicted_margin"],
                edge=row["edge"],
                pick_team=row["pick_team"],
                is_best_pick=False,
            )
            for row in other_rows
        ]

        # Clear any stale is_best_pick flag on other ungraded rows for this
        # week. This can never touch a graded row, since it's scoped to
        # result IS NULL.
        conn.execute(
            "UPDATE picks SET is_best_pick = 0 WHERE season = ? AND week = ? AND result IS NULL",
            (season, week),
        )

        # Recompute the best pick across every ungraded game for the week —
        # this run's games plus any held-back games from earlier runs — so a
        # partial re-predict can't steal the best-pick flag away from a
        # higher-confidence game it doesn't touch.
        for pick in picks:
            pick.is_best_pick = False
        all_candidates = picks + other_picks
        if all_candidates:
            best = rank_by_confidence(all_candidates, calibration)[0]
            best.is_best_pick = True
            if any(best is other for other in other_picks):
                conn.execute(
                    "UPDATE picks SET is_best_pick = 1 WHERE season = ? AND week = ? "
                    "AND home_team = ? AND away_team = ? AND result IS NULL",
                    (season, week, best.home_team, best.away_team),
                )
                click.echo(
                    f"Best pick for the week stays with {best.away_team} @ "
                    f"{best.home_team} (not in this slate)."
                )

    for pick in picks:
        # The DELETE above is scoped per-game, so any remaining PK conflict
        # here is by definition a graded row for that exact game (deliberately
        # not deleted) — INSERT OR IGNORE leaves it untouched rather than
        # clobbering it back to result = NULL.
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


@cli.command("top-picks")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--season", required=True, type=int)
@click.option("--week", required=True, type=int)
@click.option("--count", default=10, type=int)
@click.option("--output", "output_path", default=None, type=click.Path())
def top_picks_cmd(input_path, season, week, count, output_path):
    conn = get_connection(DB_PATH)
    weights = load_model(MODEL_PATH)
    try:
        calibration = load_calibration(CALIBRATION_PATH)
    except FileNotFoundError:
        calibration = None

    games = []
    with open(input_path, newline="", encoding="utf-8") as f:
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

    # This is a read-only report for betting outside the contest: it never
    # writes to the picks table, so it can freely overlap with the contest
    # slate without touching its grading/lock state.
    picks = make_picks(games, calibration=calibration)
    ranked = rank_by_confidence(picks, calibration)

    if output_path is None:
        output_path = f"top_picks_{season}_week{week}.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "home_team",
                "away_team",
                "spread",
                "pick_team",
                "predicted_margin",
                "edge",
                "confidence",
                "top_pick",
            ]
        )
        for rank, pick in enumerate(ranked, start=1):
            writer.writerow(
                [
                    rank,
                    pick.home_team,
                    pick.away_team,
                    pick.spread,
                    pick.pick_team,
                    round(pick.predicted_margin, 1),
                    round(pick.edge, 1),
                    round(confidence_score(pick, calibration), 1),
                    int(rank <= count),
                ]
            )

    for pick in ranked[:count]:
        click.echo(
            f"{pick.away_team} @ {pick.home_team} ({pick.spread:+}): "
            f"pick {pick.pick_team} (edge {pick.edge:+.1f}, "
            f"confidence {confidence_score(pick, calibration):.1f})"
        )
    click.echo(f"Full ranked list ({len(ranked)} games) written to {output_path}")


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
