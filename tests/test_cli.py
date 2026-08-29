# tests/test_cli.py
from click.testing import CliRunner

import cfb_picks.cli as cli_module
from cfb_picks.db import get_connection, init_db
from cfb_picks.model import ModelWeights, save_model


def _setup_env(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    model_path = tmp_path / "model.json"
    calibration_path = tmp_path / "calibration.json"
    monkeypatch.setattr(cli_module, "DB_PATH", db_path)
    monkeypatch.setattr(cli_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(cli_module, "CALIBRATION_PATH", calibration_path)

    conn = get_connection(db_path)
    init_db(conn)
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('A', 'A')")
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('B', 'B')")
    conn.commit()

    weights = ModelWeights(
        intercept=5,
        coefficients={"elo_diff": 0, "success_rate_diff": 0, "ppa_diff": 0, "neutral_site": 0},
    )
    save_model(weights, model_path)
    return db_path


def test_predict_persists_picks_and_prints_best_pick(tmp_path, monkeypatch):
    db_path = _setup_env(tmp_path, monkeypatch)
    slate = tmp_path / "slate.csv"
    slate.write_text("home_team,away_team,spread\nA,B,-3\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )

    assert result.exit_code == 0
    assert "BEST PICK" in result.output

    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM picks WHERE home_team = 'A'").fetchone()
    assert row["pick_team"] == "A"
    assert row["is_best_pick"] == 1


def test_predict_rerun_replaces_stale_rows_without_duplicating_best_pick(tmp_path, monkeypatch):
    db_path = _setup_env(tmp_path, monkeypatch)
    slate = tmp_path / "slate.csv"
    slate.write_text("home_team,away_team,spread\nA,B,-3\n")

    runner = CliRunner()
    runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )

    # Re-run with a corrected slate for the same season/week.
    slate.write_text("home_team,away_team,spread\nA,B,-7\n")
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0

    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'A' AND away_team = 'B'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["spread"] == -7
    total_best_picks = conn.execute(
        "SELECT SUM(is_best_pick) c FROM picks WHERE season = 2024 AND week = 1"
    ).fetchone()["c"]
    assert total_best_picks <= 1


def test_predict_rerun_does_not_erase_already_graded_result(tmp_path, monkeypatch):
    db_path = _setup_env(tmp_path, monkeypatch)
    slate = tmp_path / "slate.csv"
    slate.write_text("home_team,away_team,spread\nA,B,-3\n")

    runner = CliRunner()
    runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )

    conn = get_connection(db_path)
    conn.execute(
        "UPDATE picks SET result = 'win' WHERE season = 2024 AND week = 1 "
        "AND home_team = 'A' AND away_team = 'B'"
    )
    conn.commit()

    # Re-running predict for the same graded slate must not wipe the result.
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0

    row = conn.execute(
        "SELECT result FROM picks WHERE season = 2024 AND week = 1 "
        "AND home_team = 'A' AND away_team = 'B'"
    ).fetchone()
    assert row["result"] == "win"


def test_predict_partial_rerun_preserves_other_games(tmp_path, monkeypatch):
    db_path = _setup_env(tmp_path, monkeypatch)
    conn = get_connection(db_path)
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('C', 'C')")
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('D', 'D')")
    conn.commit()

    slate = tmp_path / "slate.csv"
    slate.write_text("home_team,away_team,spread\nA,B,-3\nC,D,-3\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0, result.output

    original_cd_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'C' AND away_team = 'D'"
    ).fetchone()
    assert original_cd_row is not None

    # "Correction" re-predict containing only the A/B game.
    correction = tmp_path / "correction.csv"
    correction.write_text("home_team,away_team,spread\nA,B,-7\n")
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(correction), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0, result.output

    cd_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'C' AND away_team = 'D'"
    ).fetchone()
    assert cd_row is not None
    assert cd_row["spread"] == original_cd_row["spread"]
    assert cd_row["pick_team"] == original_cd_row["pick_team"]
    assert cd_row["predicted_margin"] == original_cd_row["predicted_margin"]

    ab_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'A' AND away_team = 'B'"
    ).fetchone()
    assert ab_row["spread"] == -7


def test_predict_does_not_duplicate_best_pick_after_grading(tmp_path, monkeypatch):
    db_path = _setup_env(tmp_path, monkeypatch)
    conn = get_connection(db_path)
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('C', 'C')")
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('D', 'D')")
    # A graded, already-scored best pick for the week.
    conn.execute(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) VALUES "
        "(2024, 1, 'A', 'B', -3, 10, 7, 'A', 1, 'win')"
    )
    conn.commit()

    slate = tmp_path / "slate.csv"
    # Large spread mismatch so make_picks would normally flag this as best pick.
    slate.write_text("home_team,away_team,spread\nC,D,-30\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0, result.output

    total_best_picks = conn.execute(
        "SELECT SUM(is_best_pick) c FROM picks WHERE season = 2024 AND week = 1"
    ).fetchone()["c"]
    assert total_best_picks == 1

    cd_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'C' AND away_team = 'D'"
    ).fetchone()
    assert cd_row["is_best_pick"] == 0


def test_predict_shifts_best_pick_before_any_grading(tmp_path, monkeypatch):
    db_path = _setup_env(tmp_path, monkeypatch)
    conn = get_connection(db_path)
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('C', 'C')")
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('D', 'D')")
    conn.commit()

    slate_a = tmp_path / "slate_a.csv"
    slate_a.write_text("home_team,away_team,spread\nA,B,-3\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate_a), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0, result.output

    ab_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'A' AND away_team = 'B'"
    ).fetchone()
    assert ab_row["is_best_pick"] == 1

    slate_b = tmp_path / "slate_b.csv"
    # Much bigger spread mismatch so make_picks flags this as best pick on its own.
    slate_b.write_text("home_team,away_team,spread\nC,D,-30\n")
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate_b), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0, result.output

    ab_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'A' AND away_team = 'B'"
    ).fetchone()
    cd_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'C' AND away_team = 'D'"
    ).fetchone()
    assert ab_row["is_best_pick"] == 0
    assert cd_row["is_best_pick"] == 1


def test_predict_partial_rerun_does_not_steal_best_pick_from_held_back_game(
    tmp_path, monkeypatch
):
    db_path = _setup_env(tmp_path, monkeypatch)
    conn = get_connection(db_path)
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('C', 'C')")
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('D', 'D')")
    conn.commit()

    slate = tmp_path / "slate.csv"
    # A/B has a much bigger edge than C/D, so A/B is the week's best pick.
    slate.write_text("home_team,away_team,spread\nA,B,-30\nC,D,-3\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0, result.output

    ab_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'A' AND away_team = 'B'"
    ).fetchone()
    assert ab_row["is_best_pick"] == 1

    # Correction re-predict touching only the untouched, lower-edge C/D game.
    correction = tmp_path / "correction.csv"
    correction.write_text("home_team,away_team,spread\nC,D,-4\n")
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(correction), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0, result.output

    ab_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'A' AND away_team = 'B'"
    ).fetchone()
    cd_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'C' AND away_team = 'D'"
    ).fetchone()
    assert ab_row["is_best_pick"] == 1
    assert cd_row["is_best_pick"] == 0


def test_predict_locks_best_pick_once_any_game_in_week_is_graded(tmp_path, monkeypatch):
    db_path = _setup_env(tmp_path, monkeypatch)
    conn = get_connection(db_path)
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('C', 'C')")
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('D', 'D')")
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('E', 'E')")
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('F', 'F')")
    conn.commit()

    slate = tmp_path / "slate.csv"
    # A/B is the week's best pick; C/D is not.
    slate.write_text("home_team,away_team,spread\nA,B,-30\nC,D,-3\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0, result.output

    # C/D (not the flagged best pick) finishes and gets graded.
    conn.execute(
        "UPDATE picks SET result = 'win' WHERE season = 2024 AND week = 1 "
        "AND home_team = 'C' AND away_team = 'D'"
    )
    conn.commit()

    # A later run adds a new game with an enormous edge that would otherwise
    # become the new best pick.
    addition = tmp_path / "addition.csv"
    addition.write_text("home_team,away_team,spread\nE,F,-100\n")
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(addition), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "already locked in" in result.output

    ab_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'A' AND away_team = 'B'"
    ).fetchone()
    ef_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'E' AND away_team = 'F'"
    ).fetchone()
    assert ab_row["is_best_pick"] == 1
    assert ef_row["is_best_pick"] == 0


def test_predict_correction_to_locked_best_pick_keeps_its_flag(tmp_path, monkeypatch):
    db_path = _setup_env(tmp_path, monkeypatch)
    conn = get_connection(db_path)
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('C', 'C')")
    conn.execute("INSERT INTO team_aliases (alias, canonical_school) VALUES ('D', 'D')")
    conn.commit()

    slate = tmp_path / "slate.csv"
    # A/B is the week's best pick; C/D is not.
    slate.write_text("home_team,away_team,spread\nA,B,-30\nC,D,-3\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0, result.output

    # C/D (not the best pick) finishes first — the week is now locked.
    conn.execute(
        "UPDATE picks SET result = 'win' WHERE season = 2024 AND week = 1 "
        "AND home_team = 'C' AND away_team = 'D'"
    )
    conn.commit()

    # Correct the spread on A/B, which is itself the locked-in best pick.
    correction = tmp_path / "correction.csv"
    correction.write_text("home_team,away_team,spread\nA,B,-28\n")
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(correction), "--season", "2024", "--week", "1"],
    )
    assert result.exit_code == 0, result.output

    ab_row = conn.execute(
        "SELECT * FROM picks WHERE season = 2024 AND week = 1 AND home_team = 'A' AND away_team = 'B'"
    ).fetchone()
    assert ab_row["spread"] == -28
    assert ab_row["is_best_pick"] == 1
    total_best_picks = conn.execute(
        "SELECT SUM(is_best_pick) c FROM picks WHERE season = 2024 AND week = 1"
    ).fetchone()["c"]
    assert total_best_picks == 1


def test_predict_reads_optional_neutral_site_column(tmp_path, monkeypatch):
    _setup_env(tmp_path, monkeypatch)
    slate = tmp_path / "slate.csv"
    slate.write_text("home_team,away_team,spread,neutral_site\nA,B,-3,1\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )

    assert result.exit_code == 0


def test_predict_handles_short_row_missing_neutral_site_value(tmp_path, monkeypatch):
    _setup_env(tmp_path, monkeypatch)
    slate = tmp_path / "slate.csv"
    # Header declares neutral_site, but the data row is short by that trailing
    # field, so csv.DictReader fills it with None (not absent, not "0").
    slate.write_text("home_team,away_team,spread,neutral_site\nA,B,-3\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )

    assert result.exit_code == 0, result.output


def test_build_ratings_raises_clean_error_when_no_training_data(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(cli_module, "DB_PATH", db_path)
    monkeypatch.setattr(cli_module, "MODEL_PATH", tmp_path / "model.json")
    monkeypatch.setattr(cli_module, "CALIBRATION_PATH", tmp_path / "calibration.json")
    monkeypatch.setattr(cli_module, "SEASONS", [2024])
    conn = get_connection(db_path)
    init_db(conn)
    conn.commit()

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["build-ratings"])

    assert result.exit_code != 0
    assert "No training data available" in str(result.output) + str(result.exception)


def test_build_ratings_rejects_seasons_option(tmp_path, monkeypatch):
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["build-ratings", "--seasons", "2024"])

    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_backtest_rejects_seasons_option(tmp_path, monkeypatch):
    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["backtest", "--seasons", "2024"])

    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_history_reports_season_record(tmp_path, monkeypatch):
    db_path = _setup_env(tmp_path, monkeypatch)
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) VALUES (2024, 1, 'A', 'B', -3, 10, 7, 'A', 1, 'win')"
    )
    conn.commit()

    runner = CliRunner()
    result = runner.invoke(cli_module.cli, ["history", "--season", "2024"])

    assert result.exit_code == 0
    assert "'points': 2" in result.output
