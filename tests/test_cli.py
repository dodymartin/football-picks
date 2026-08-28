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
