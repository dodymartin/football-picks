import datetime
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CURRENT_YEAR = datetime.date.today().year

DB_PATH = Path(os.environ.get("CFB_PICKS_DB", _PROJECT_ROOT / "data" / "cfb_picks.db"))
MODEL_PATH = Path(
    os.environ.get("CFB_PICKS_MODEL", _PROJECT_ROOT / "data" / "model_weights.json")
)
CALIBRATION_PATH = Path(
    os.environ.get("CFB_PICKS_CALIBRATION", _PROJECT_ROOT / "data" / "calibration.json")
)
SEASONS = [year for year in range(2021, _CURRENT_YEAR + 1) if year != 2020]
