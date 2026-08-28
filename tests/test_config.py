import datetime
import importlib
from pathlib import Path


def test_seasons_starts_at_2021_excludes_2020_and_includes_current_year():
    from cfb_picks.config import SEASONS

    assert SEASONS[0] == 2021
    assert 2020 not in SEASONS
    assert SEASONS[-1] == datetime.date.today().year
    assert SEASONS == sorted(SEASONS)


def test_paths_overridable_by_env_vars(monkeypatch, tmp_path):
    db_path = str(tmp_path / "custom.db")
    model_path = str(tmp_path / "custom_model.json")
    calibration_path = str(tmp_path / "custom_calibration.json")
    monkeypatch.setenv("CFB_PICKS_DB", db_path)
    monkeypatch.setenv("CFB_PICKS_MODEL", model_path)
    monkeypatch.setenv("CFB_PICKS_CALIBRATION", calibration_path)

    import cfb_picks.config as config

    importlib.reload(config)

    assert config.DB_PATH == Path(db_path)
    assert config.MODEL_PATH == Path(model_path)
    assert config.CALIBRATION_PATH == Path(calibration_path)

    monkeypatch.delenv("CFB_PICKS_DB", raising=False)
    monkeypatch.delenv("CFB_PICKS_MODEL", raising=False)
    monkeypatch.delenv("CFB_PICKS_CALIBRATION", raising=False)
    importlib.reload(config)
