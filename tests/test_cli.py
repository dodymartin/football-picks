# tests/test_cli.py
import csv

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


def test_predict_reads_utf8_team_names(tmp_path, monkeypatch):
    # Windows' default text-mode encoding is cp1252, not utf-8; a slate CSV
    # saved as utf-8 with an accented team name must still be read correctly.
    db_path = _setup_env(tmp_path, monkeypatch)
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO team_aliases (alias, canonical_school) VALUES (?, ?)",
        ("San José State", "San José State"),
    )
    conn.commit()

    slate = tmp_path / "slate.csv"
    slate.write_bytes("home_team,away_team,spread\nA,San José State,-3\n".encode("utf-8"))

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["predict", "--input", str(slate), "--season", "2024", "--week", "1"],
    )

    assert result.exit_code == 0, result.output


def test_top_picks_ranks_by_confidence_writes_full_csv_and_does_not_touch_picks_table(
    tmp_path, monkeypatch
):
    db_path = _setup_env(tmp_path, monkeypatch)
    conn = get_connection(db_path)
    for team in ("C", "D", "E", "F"):
        conn.execute(
            "INSERT INTO team_aliases (alias, canonical_school) VALUES (?, ?)", (team, team)
        )
    conn.commit()

    # predicted_margin is always 5 (intercept=5, all coefficients 0), so
    # edge = 5 - (-spread) = 5 + spread, and with no calibration file,
    # confidence = abs(edge):
    #   A vs B: spread -1 -> edge 4
    #   C vs D: spread -8 -> edge -3
    #   E vs F: spread  2 -> edge 7  (highest confidence)
    slate = tmp_path / "slate.csv"
    slate.write_text("home_team,away_team,spread\nA,B,-1\nC,D,-8\nE,F,2\n")
    output = tmp_path / "out.csv"

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "top-picks",
            "--input",
            str(slate),
            "--season",
            "2024",
            "--week",
            "1",
            "--count",
            "2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()

    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert [r["home_team"] for r in rows] == ["E", "A", "C"]
    assert [r["top_pick"] for r in rows] == ["1", "1", "0"]

    conn = get_connection(db_path)
    assert conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0] == 0


def test_top_picks_writes_default_output_path_when_not_given(tmp_path, monkeypatch):
    db_path = _setup_env(tmp_path, monkeypatch)
    slate = tmp_path / "slate.csv"
    slate.write_text("home_team,away_team,spread\nA,B,-1\n")
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["top-picks", "--input", str(slate), "--season", "2024", "--week", "1"],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "top_picks_2024_week1.csv").exists()


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
