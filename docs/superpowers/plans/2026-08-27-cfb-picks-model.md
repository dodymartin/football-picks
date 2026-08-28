# CFB Picks Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI (`cfb-picks`) that predicts college football spread coverage using an Elo power rating blended with a few CFBD efficiency stats via ridge regression, backtested out-of-sample on 2021-present historical data, to help pick a weekly 13-game ATS slate plus a "best pick."

**Architecture:** SQLite stores historical games/lines/stats pulled from the CFBD API. An Elo engine computes team ratings game-by-game with season carryover regression and FCS-aware baselines. A feature builder turns any matchup into `[elo_diff, success_rate_diff, ppa_diff, neutral_site]`, using only data available before that game (no lookahead). A ridge regression (trained on all historical FBS-vs-FBS games) maps features to a predicted margin; comparing that to the market spread gives an edge, which drives the pick. A walk-forward backtest harness (training only on seasons prior to the one being evaluated) validates ATS accuracy and produces an edge-size calibration table, which is what actually drives "best pick" selection. Everything is wired together via a Click CLI, with picks and results persisted to SQLite for season-long record tracking.

**Tech Stack:** Python 3.10+, `requests` (CFBD HTTP client), `click` (CLI), `scikit-learn` (Ridge regression), `python-dotenv` (load `CFBD_API_KEY` from `.env`), stdlib `sqlite3` (storage), `pytest` (tests).

**Spec:** `docs/superpowers/specs/2026-08-27-cfb-picks-model-design.md`

## Global Constraints

- No paid data/odds APIs. All historical data comes from CFBD's free tier (1,000 calls/month) — fetch data at season granularity (one call per endpoint per season), never per-team or per-week, to stay well under the limit.
- Never scrape Splash. The weekly slate + spread is provided by the user (pasted text/screenshot converted to a CSV) — the app only ever reads a local CSV for the weekly slate.
- Training/backtest data: seasons 2021 through the **current** season inclusive, with 2020 excluded (COVID-disrupted). The season range is computed dynamically from today's date, not hardcoded, so the tool keeps working as seasons roll forward.
- Single user, local-only. No hosting, no web UI, no multi-user support.
- Spread sign convention (matches CFBD and standard sportsbook convention): **negative spread = home team favored by that many points.** `market_home_margin = -spread`. `edge = predicted_margin - market_home_margin`; positive edge picks the home team, negative picks the away team.
- CFBD API confirmed endpoints used: `GET /teams/fbs?year=`, `GET /games?year=&seasonType=`, `GET /lines?year=&seasonType=`, `GET /stats/game/advanced?year=&seasonType=`. Auth via `Authorization: Bearer <CFBD_API_KEY>` header.
- Backtest calibration is derived from CFBD's consensus closing lines, which can diverge from Splash's contest line (set earlier, off-market). Treat calibration hit-rates as directional guidance, not exact live probabilities — noted in the README.
- Elo/regression hyperparameters (`K_FACTOR`, `HOME_FIELD_ELO`, `REGRESSION_FACTOR`, ridge `alpha`) are intentionally left as hand-tunable constants rather than an automated search, to keep v1 lean (YAGNI) — reviewed manually against backtest output, documented in the README.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `cfb_picks/__init__.py`
- Create: `data/.gitkeep`
- Create: `pytest.ini`

**Interfaces:**
- Produces: an installable `cfb_picks` package and a `cfb-picks` console script entry point (wired to `cfb_picks.cli:main` in a later task).

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "cfb-picks"
version = "0.1.0"
description = "Power-rating model for picking college football games against the spread"
requires-python = ">=3.10"
dependencies = [
    "requests>=2.31",
    "click>=8.1",
    "scikit-learn>=1.3",
    "python-dotenv>=1.0",
]

[project.scripts]
cfb-picks = "cfb_picks.cli:main"

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["cfb_picks*"]
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
data/*.db
data/*.json
.env
```

- [ ] **Step 3: Create `.env.example`**

```
CFBD_API_KEY=your_api_key_here
```

- [ ] **Step 4: Create empty `cfb_picks/__init__.py` and `data/.gitkeep`**

- [ ] **Step 5: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
```

- [ ] **Step 6: Install and verify the package imports**

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev]"
python -c "import cfb_picks"
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore .env.example cfb_picks/__init__.py data/.gitkeep pytest.ini
git commit -m "chore: project scaffolding"
```

---

## Task 2: SQLite schema & connection layer

**Files:**
- Create: `cfb_picks/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces: `get_connection(db_path) -> sqlite3.Connection` (row_factory = `sqlite3.Row`), `init_db(conn) -> None`.
- Tables: `teams(school PK, conference, classification)`, `team_aliases(alias PK, canonical_school)`, `games(id PK, season, week, season_type, start_date, completed, neutral_site, home_team, away_team, home_points, away_points)`, `betting_lines(game_id, provider, spread, PK(game_id, provider))`, `team_game_stats(game_id, team, success_rate, ppa, PK(game_id, team))`, `elo_ratings(season, week, team, rating, PK(season, week, team))`, `picks(season, week, home_team, away_team, spread, predicted_margin, edge, pick_team, is_best_pick, result, PK(season, week, home_team, away_team))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
from cfb_picks.db import get_connection, init_db


def test_init_db_creates_tables(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    expected = {
        "teams", "team_aliases", "games", "betting_lines",
        "team_game_stats", "elo_ratings", "picks",
    }
    assert expected.issubset(tables)


def test_get_connection_creates_parent_dir_and_row_factory(tmp_path):
    db_path = tmp_path / "nested" / "test.db"
    conn = get_connection(db_path)
    assert db_path.parent.exists()
    init_db(conn)
    conn.execute("INSERT INTO teams (school) VALUES ('Ohio State')")
    conn.commit()
    row = conn.execute("SELECT school FROM teams").fetchone()
    assert row["school"] == "Ohio State"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.db'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/db.py
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    school TEXT PRIMARY KEY,
    conference TEXT,
    classification TEXT
);

CREATE TABLE IF NOT EXISTS team_aliases (
    alias TEXT PRIMARY KEY,
    canonical_school TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    season_type TEXT NOT NULL,
    start_date TEXT,
    completed INTEGER NOT NULL,
    neutral_site INTEGER NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_points INTEGER,
    away_points INTEGER
);

CREATE TABLE IF NOT EXISTS betting_lines (
    game_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    spread REAL,
    PRIMARY KEY (game_id, provider)
);

CREATE TABLE IF NOT EXISTS team_game_stats (
    game_id INTEGER NOT NULL,
    team TEXT NOT NULL,
    success_rate REAL,
    ppa REAL,
    PRIMARY KEY (game_id, team)
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team TEXT NOT NULL,
    rating REAL NOT NULL,
    PRIMARY KEY (season, week, team)
);

CREATE TABLE IF NOT EXISTS picks (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    spread REAL NOT NULL,
    predicted_margin REAL NOT NULL,
    edge REAL NOT NULL,
    pick_team TEXT NOT NULL,
    is_best_pick INTEGER NOT NULL,
    result TEXT,
    PRIMARY KEY (season, week, home_team, away_team)
);
"""


def get_connection(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/db.py tests/test_db.py
git commit -m "feat: add SQLite schema and connection layer"
```

---

## Task 3: CFBD API client

**Files:**
- Create: `cfb_picks/cfbd_client.py`
- Test: `tests/test_cfbd_client.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `CFBDClient(api_key=None, session=None)` with methods `get_fbs_teams(year) -> list[dict]`, `get_games(year, season_type="regular") -> list[dict]`, `get_lines(year, season_type="regular") -> list[dict]`, `get_advanced_game_stats(year, season_type="regular") -> list[dict]`. Raises `RuntimeError` if no API key is available (constructor arg or `CFBD_API_KEY` env var).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cfbd_client.py
from unittest.mock import MagicMock

import pytest

from cfb_picks.cfbd_client import CFBDClient


def _client_with_fake_session(payload):
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    session.get.return_value = response
    return CFBDClient(api_key="test-key", session=session), session


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        CFBDClient(session=MagicMock())


def test_get_games_calls_correct_endpoint():
    client, session = _client_with_fake_session([{"id": 1, "season": 2024}])

    result = client.get_games(2024)

    session.get.assert_called_once_with(
        "https://api.collegefootballdata.com/games",
        params={"year": 2024, "seasonType": "regular"},
        headers={"Authorization": "Bearer test-key"},
        timeout=30,
    )
    assert result == [{"id": 1, "season": 2024}]


def test_get_fbs_teams_calls_correct_endpoint():
    client, session = _client_with_fake_session([{"school": "Ohio State"}])

    result = client.get_fbs_teams(2024)

    session.get.assert_called_once_with(
        "https://api.collegefootballdata.com/teams/fbs",
        params={"year": 2024},
        headers={"Authorization": "Bearer test-key"},
        timeout=30,
    )
    assert result == [{"school": "Ohio State"}]


def test_get_lines_and_advanced_stats_use_season_type_param():
    client, session = _client_with_fake_session([])

    client.get_lines(2024)
    session.get.assert_called_with(
        "https://api.collegefootballdata.com/lines",
        params={"year": 2024, "seasonType": "regular"},
        headers={"Authorization": "Bearer test-key"},
        timeout=30,
    )

    client.get_advanced_game_stats(2024)
    session.get.assert_called_with(
        "https://api.collegefootballdata.com/stats/game/advanced",
        params={"year": 2024, "seasonType": "regular"},
        headers={"Authorization": "Bearer test-key"},
        timeout=30,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cfbd_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.cfbd_client'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/cfbd_client.py
import os

import requests


class CFBDClient:
    BASE_URL = "https://api.collegefootballdata.com"

    def __init__(self, api_key=None, session=None):
        self.api_key = api_key or os.environ.get("CFBD_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "CFBD_API_KEY not set. Get a free key at "
                "https://collegefootballdata.com/key and set it in your "
                "environment or a .env file."
            )
        self.session = session or requests.Session()

    def _get(self, path, params=None):
        response = self.session.get(
            f"{self.BASE_URL}{path}",
            params=params or {},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_fbs_teams(self, year):
        return self._get("/teams/fbs", {"year": year})

    def get_games(self, year, season_type="regular"):
        return self._get("/games", {"year": year, "seasonType": season_type})

    def get_lines(self, year, season_type="regular"):
        return self._get("/lines", {"year": year, "seasonType": season_type})

    def get_advanced_game_stats(self, year, season_type="regular"):
        return self._get(
            "/stats/game/advanced", {"year": year, "seasonType": season_type}
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cfbd_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/cfbd_client.py tests/test_cfbd_client.py
git commit -m "feat: add CFBD API client"
```

---

## Task 4: Team alias normalization

**Files:**
- Create: `cfb_picks/aliases.py`
- Test: `tests/test_aliases.py`

**Interfaces:**
- Consumes: `get_connection`, `init_db` from `cfb_picks.db`.
- Produces: `seed_aliases(conn, teams_json) -> None`, `add_alias(conn, alias, canonical_school) -> None`, `normalize_team_name(conn, name) -> str` (raises `ValueError` if unknown).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aliases.py
import pytest

from cfb_picks.aliases import add_alias, normalize_team_name, seed_aliases
from cfb_picks.db import get_connection, init_db


def _conn(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    return conn


def test_seed_aliases_and_normalize(tmp_path):
    conn = _conn(tmp_path)
    seed_aliases(conn, [{"school": "Ohio State", "abbreviation": "OSU"}])

    assert normalize_team_name(conn, "Ohio State") == "Ohio State"
    assert normalize_team_name(conn, "OSU") == "Ohio State"


def test_add_alias_and_unknown_name_raises(tmp_path):
    conn = _conn(tmp_path)
    add_alias(conn, "Ohio St.", "Ohio State")

    assert normalize_team_name(conn, "Ohio St.") == "Ohio State"
    with pytest.raises(ValueError):
        normalize_team_name(conn, "Nonexistent Team")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aliases.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.aliases'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/aliases.py
def seed_aliases(conn, teams_json):
    rows = []
    for team in teams_json:
        school = team["school"]
        rows.append((school, school))
        if team.get("abbreviation"):
            rows.append((team["abbreviation"], school))
    conn.executemany(
        "INSERT OR IGNORE INTO team_aliases (alias, canonical_school) VALUES (?, ?)",
        rows,
    )
    conn.commit()


def add_alias(conn, alias, canonical_school):
    conn.execute(
        "INSERT INTO team_aliases (alias, canonical_school) VALUES (?, ?) "
        "ON CONFLICT(alias) DO UPDATE SET canonical_school=excluded.canonical_school",
        (alias, canonical_school),
    )
    conn.commit()


def normalize_team_name(conn, name):
    row = conn.execute(
        "SELECT canonical_school FROM team_aliases WHERE alias = ?", (name,)
    ).fetchone()
    if row is None:
        raise ValueError(
            f"Unknown team name: {name!r}. Add an alias with add_alias()."
        )
    return row["canonical_school"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aliases.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/aliases.py tests/test_aliases.py
git commit -m "feat: add team alias normalization"
```

---

## Task 5: Data ingestion (fetch-data logic)

**Files:**
- Create: `cfb_picks/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `get_connection`/`init_db` (`cfb_picks.db`), `seed_aliases` (`cfb_picks.aliases`).
- Produces: `upsert_teams(conn, teams_json)`, `upsert_games(conn, games_json)`, `extract_consensus_spread(betting_game_entry) -> float | None`, `upsert_betting_lines(conn, lines_json)`, `upsert_team_game_stats(conn, stats_json)`, `fetch_data(conn, client, years) -> None` (calls `client.get_fbs_teams(year)`, `client.get_games(year)`, `client.get_lines(year)`, `client.get_advanced_game_stats(year)` for each year and persists everything; idempotent).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
import pytest

from cfb_picks.db import get_connection, init_db
from cfb_picks.ingest import (
    extract_consensus_spread,
    fetch_data,
    upsert_betting_lines,
    upsert_games,
    upsert_team_game_stats,
    upsert_teams,
)

SAMPLE_GAME = {
    "id": 401520145,
    "season": 2024,
    "week": 1,
    "seasonType": "regular",
    "startDate": "2024-08-31T00:00:00.000Z",
    "completed": True,
    "neutralSite": False,
    "homeTeam": "Ohio State",
    "awayTeam": "Akron",
    "homePoints": 52,
    "awayPoints": 6,
}


def _conn(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    return conn


def test_upsert_teams(tmp_path):
    conn = _conn(tmp_path)
    upsert_teams(conn, [{"school": "Ohio State", "conference": "Big Ten", "classification": "fbs"}])
    row = conn.execute("SELECT * FROM teams WHERE school = 'Ohio State'").fetchone()
    assert row["conference"] == "Big Ten"


def test_upsert_games_is_idempotent(tmp_path):
    conn = _conn(tmp_path)
    upsert_games(conn, [SAMPLE_GAME])
    upsert_games(conn, [SAMPLE_GAME])
    row = conn.execute("SELECT * FROM games WHERE id = ?", (401520145,)).fetchone()
    assert row["home_team"] == "Ohio State"
    assert row["home_points"] == 52
    assert conn.execute("SELECT COUNT(*) c FROM games").fetchone()["c"] == 1


def test_extract_consensus_spread_averages_providers():
    entry = {"lines": [{"provider": "DraftKings", "spread": -45.5}, {"provider": "Bovada", "spread": -44.0}]}
    assert extract_consensus_spread(entry) == pytest.approx(-44.75)


def test_extract_consensus_spread_none_when_missing():
    assert extract_consensus_spread({"lines": []}) is None


def test_upsert_betting_lines(tmp_path):
    conn = _conn(tmp_path)
    upsert_games(conn, [SAMPLE_GAME])
    upsert_betting_lines(
        conn, [{"id": 401520145, "lines": [{"provider": "DraftKings", "spread": -45.5}]}]
    )
    row = conn.execute(
        "SELECT spread FROM betting_lines WHERE game_id = ?", (401520145,)
    ).fetchone()
    assert row["spread"] == -45.5


def test_upsert_team_game_stats(tmp_path):
    conn = _conn(tmp_path)
    upsert_games(conn, [SAMPLE_GAME])
    upsert_team_game_stats(
        conn,
        [
            {
                "gameId": 401520145,
                "team": "Ohio State",
                "offense": {"successRate": 0.55, "ppa": 0.42},
            }
        ],
    )
    row = conn.execute(
        "SELECT success_rate, ppa FROM team_game_stats WHERE game_id = ?", (401520145,)
    ).fetchone()
    assert row["success_rate"] == 0.55
    assert row["ppa"] == 0.42


def test_fetch_data_calls_client_and_persists_everything(tmp_path):
    conn = _conn(tmp_path)

    class FakeClient:
        def get_fbs_teams(self, year):
            return [{"school": "Ohio State", "abbreviation": "OSU"}]

        def get_games(self, year):
            return [SAMPLE_GAME]

        def get_lines(self, year):
            return [{"id": 401520145, "lines": [{"provider": "DraftKings", "spread": -45.5}]}]

        def get_advanced_game_stats(self, year):
            return [
                {
                    "gameId": 401520145,
                    "team": "Ohio State",
                    "offense": {"successRate": 0.55, "ppa": 0.42},
                }
            ]

    fetch_data(conn, FakeClient(), [2024])

    assert conn.execute("SELECT COUNT(*) c FROM games").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM betting_lines").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM team_game_stats").fetchone()["c"] == 1
    assert conn.execute("SELECT COUNT(*) c FROM team_aliases").fetchone()["c"] >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.ingest'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/ingest.py
from .aliases import seed_aliases


def upsert_teams(conn, teams_json):
    rows = [
        (team["school"], team.get("conference"), team.get("classification"))
        for team in teams_json
    ]
    conn.executemany(
        "INSERT INTO teams (school, conference, classification) VALUES (?, ?, ?) "
        "ON CONFLICT(school) DO UPDATE SET "
        "conference=excluded.conference, classification=excluded.classification",
        rows,
    )
    conn.commit()


def upsert_games(conn, games_json):
    rows = [
        (
            game["id"],
            game["season"],
            game["week"],
            game["seasonType"],
            game.get("startDate"),
            1 if game.get("completed") else 0,
            1 if game.get("neutralSite") else 0,
            game["homeTeam"],
            game["awayTeam"],
            game.get("homePoints"),
            game.get("awayPoints"),
        )
        for game in games_json
    ]
    conn.executemany(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "completed=excluded.completed, home_points=excluded.home_points, "
        "away_points=excluded.away_points",
        rows,
    )
    conn.commit()


def extract_consensus_spread(betting_game_entry):
    spreads = [
        line["spread"]
        for line in betting_game_entry.get("lines", [])
        if line.get("spread") is not None
    ]
    if not spreads:
        return None
    return sum(spreads) / len(spreads)


def upsert_betting_lines(conn, lines_json):
    rows = []
    for game in lines_json:
        spread = extract_consensus_spread(game)
        if spread is None:
            continue
        rows.append((game["id"], "consensus", spread))
    conn.executemany(
        "INSERT INTO betting_lines (game_id, provider, spread) VALUES (?, ?, ?) "
        "ON CONFLICT(game_id, provider) DO UPDATE SET spread=excluded.spread",
        rows,
    )
    conn.commit()


def upsert_team_game_stats(conn, stats_json):
    rows = [
        (
            stat["gameId"],
            stat["team"],
            stat.get("offense", {}).get("successRate"),
            stat.get("offense", {}).get("ppa"),
        )
        for stat in stats_json
    ]
    conn.executemany(
        "INSERT INTO team_game_stats (game_id, team, success_rate, ppa) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(game_id, team) DO UPDATE SET "
        "success_rate=excluded.success_rate, ppa=excluded.ppa",
        rows,
    )
    conn.commit()


def fetch_data(conn, client, years):
    for year in years:
        teams = client.get_fbs_teams(year)
        upsert_teams(conn, teams)
        seed_aliases(conn, teams)

        games = client.get_games(year)
        upsert_games(conn, games)

        lines = client.get_lines(year)
        upsert_betting_lines(conn, lines)

        stats = client.get_advanced_game_stats(year)
        upsert_team_game_stats(conn, stats)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/ingest.py tests/test_ingest.py
git commit -m "feat: add CFBD data ingestion"
```

---

## Task 6: Elo rating engine

**Files:**
- Create: `cfb_picks/elo.py`
- Test: `tests/test_elo.py`

**Interfaces:**
- Consumes: `get_connection`/`init_db` (`cfb_picks.db`).
- Produces: constants `K_FACTOR=20.0`, `HOME_FIELD_ELO=65.0`, `BASELINE_RATING=1500.0`, `FCS_BASELINE_RATING=1200.0`, `REGRESSION_FACTOR=0.66`; functions `expected_win_prob(rating_diff) -> float`, `mov_multiplier(margin) -> float`, `update_ratings(home_rating, away_rating, home_points, away_points, neutral_site, k=K_FACTOR, home_field=HOME_FIELD_ELO) -> (float, float)`, `regress_to_mean(rating, factor=REGRESSION_FACTOR, mean=BASELINE_RATING) -> float`, `is_fbs_team(conn, team) -> bool` (looks up `teams.classification`; a team with no row or a non-"fbs" classification is treated as non-FBS), `compute_elo_history(conn, seasons) -> None` (new/unseen teams start at `BASELINE_RATING` if FBS, else `FCS_BASELINE_RATING`), `get_rating_as_of(conn, season, week, team) -> float` (same classification-aware fallback for teams with no rating history at all).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_elo.py
import math

import pytest

from cfb_picks.db import get_connection, init_db
from cfb_picks.elo import (
    BASELINE_RATING,
    FCS_BASELINE_RATING,
    compute_elo_history,
    expected_win_prob,
    get_rating_as_of,
    mov_multiplier,
    regress_to_mean,
    update_ratings,
)

GAME_COLUMNS = (
    "id, season, week, season_type, start_date, completed, neutral_site, "
    "home_team, away_team, home_points, away_points"
)


def _insert_game(conn, row):
    conn.execute(f"INSERT INTO games ({GAME_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)


def test_expected_win_prob_equal_ratings_is_half():
    assert expected_win_prob(0) == pytest.approx(0.5)


def test_mov_multiplier_increases_with_margin():
    assert mov_multiplier(28) > mov_multiplier(3)


def test_update_ratings_home_win_neutral_site():
    new_home, new_away = update_ratings(1500, 1500, 31, 17, neutral_site=True, k=20, home_field=65)
    expected_change = 20 * math.log(15) * 0.5
    assert new_home == pytest.approx(1500 + expected_change)
    assert new_away == pytest.approx(1500 - expected_change)


def test_update_ratings_accounts_for_home_field():
    new_home_neutral, _ = update_ratings(1500, 1500, 24, 20, neutral_site=True, k=20, home_field=65)
    new_home_with_field, _ = update_ratings(1500, 1500, 24, 20, neutral_site=False, k=20, home_field=65)
    assert new_home_with_field < new_home_neutral


def test_regress_to_mean_pulls_toward_baseline():
    assert regress_to_mean(1700, factor=0.66, mean=BASELINE_RATING) == pytest.approx(1500 + 0.66 * 200)


def test_compute_elo_history_updates_ratings(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO teams (school, classification) VALUES ('Ohio State', 'fbs'), ('Akron', 'fbs')"
    )
    _insert_game(conn, (1, 2024, 1, "regular", None, 1, 0, "Ohio State", "Akron", 52, 6))
    conn.commit()

    compute_elo_history(conn, [2024])

    assert get_rating_as_of(conn, 2024, 2, "Ohio State") > BASELINE_RATING
    assert get_rating_as_of(conn, 2024, 2, "Akron") < BASELINE_RATING


def test_compute_elo_history_starts_non_fbs_opponents_at_lower_baseline(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('Ohio State', 'fbs')")
    # 'Some FCS School' has no teams row, so it's treated as non-FBS.
    _insert_game(conn, (1, 2024, 1, "regular", None, 1, 0, "Ohio State", "Some FCS School", 45, 3))
    conn.commit()

    compute_elo_history(conn, [2024])

    preseason_row = conn.execute(
        "SELECT rating FROM elo_ratings WHERE season = 2024 AND week = 0 AND team = 'Some FCS School'"
    ).fetchone()
    assert preseason_row["rating"] == FCS_BASELINE_RATING


def test_get_rating_as_of_returns_fbs_baseline_for_known_fbs_team(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('Ohio State', 'fbs')")
    conn.commit()
    assert get_rating_as_of(conn, 2024, 1, "Ohio State") == BASELINE_RATING


def test_get_rating_as_of_returns_fcs_baseline_for_unknown_team(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    assert get_rating_as_of(conn, 2024, 1, "Random FCS School") == FCS_BASELINE_RATING


def test_get_rating_as_of_carries_prior_season_forward(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO teams (school, classification) VALUES ('Ohio State', 'fbs'), ('Akron', 'fbs')"
    )
    _insert_game(conn, (1, 2023, 15, "regular", None, 1, 0, "Ohio State", "Akron", 52, 6))
    conn.commit()

    compute_elo_history(conn, [2023])

    rating_end_2023 = get_rating_as_of(conn, 2023, 16, "Ohio State")
    rating_2024 = get_rating_as_of(conn, 2024, 1, "Ohio State")
    assert rating_2024 == pytest.approx(rating_end_2023)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_elo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.elo'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/elo.py
import math

K_FACTOR = 20.0
HOME_FIELD_ELO = 65.0
BASELINE_RATING = 1500.0
FCS_BASELINE_RATING = 1200.0
REGRESSION_FACTOR = 0.66


def expected_win_prob(rating_diff):
    return 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))


def mov_multiplier(margin):
    return math.log(abs(margin) + 1)


def update_ratings(
    home_rating,
    away_rating,
    home_points,
    away_points,
    neutral_site,
    k=K_FACTOR,
    home_field=HOME_FIELD_ELO,
):
    effective_diff = (
        (home_rating - away_rating)
        if neutral_site
        else (home_rating + home_field - away_rating)
    )
    expected_home = expected_win_prob(effective_diff)
    actual_home = 1.0 if home_points > away_points else 0.0
    margin = home_points - away_points
    change = k * mov_multiplier(margin) * (actual_home - expected_home)
    return home_rating + change, away_rating - change


def regress_to_mean(rating, factor=REGRESSION_FACTOR, mean=BASELINE_RATING):
    return mean + factor * (rating - mean)


def is_fbs_team(conn, team):
    row = conn.execute(
        "SELECT classification FROM teams WHERE school = ?", (team,)
    ).fetchone()
    return bool(row and row["classification"] == "fbs")


def _baseline_rating(conn, team):
    return BASELINE_RATING if is_fbs_team(conn, team) else FCS_BASELINE_RATING


def compute_elo_history(conn, seasons):
    conn.execute("DELETE FROM elo_ratings")
    ratings = {}
    for season in seasons:
        for team in list(ratings.keys()):
            ratings[team] = regress_to_mean(ratings[team])

        games = conn.execute(
            "SELECT * FROM games WHERE season = ? AND completed = 1 "
            "ORDER BY week ASC, id ASC",
            (season,),
        ).fetchall()

        for team, rating in ratings.items():
            conn.execute(
                "INSERT OR REPLACE INTO elo_ratings (season, week, team, rating) "
                "VALUES (?, 0, ?, ?)",
                (season, team, rating),
            )

        for game in games:
            home, away = game["home_team"], game["away_team"]
            ratings.setdefault(home, _baseline_rating(conn, home))
            ratings.setdefault(away, _baseline_rating(conn, away))
            new_home, new_away = update_ratings(
                ratings[home],
                ratings[away],
                game["home_points"],
                game["away_points"],
                bool(game["neutral_site"]),
            )
            ratings[home], ratings[away] = new_home, new_away
            conn.execute(
                "INSERT OR REPLACE INTO elo_ratings (season, week, team, rating) "
                "VALUES (?, ?, ?, ?)",
                (season, game["week"], home, new_home),
            )
            conn.execute(
                "INSERT OR REPLACE INTO elo_ratings (season, week, team, rating) "
                "VALUES (?, ?, ?, ?)",
                (season, game["week"], away, new_away),
            )
    conn.commit()


def get_rating_as_of(conn, season, week, team):
    row = conn.execute(
        "SELECT rating FROM elo_ratings WHERE season = ? AND week < ? AND team = ? "
        "ORDER BY week DESC LIMIT 1",
        (season, week, team),
    ).fetchone()
    if row:
        return row["rating"]

    row = conn.execute(
        "SELECT rating FROM elo_ratings WHERE season < ? AND team = ? "
        "ORDER BY season DESC, week DESC LIMIT 1",
        (season, team),
    ).fetchone()
    if row:
        return row["rating"]

    return _baseline_rating(conn, team)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_elo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/elo.py tests/test_elo.py
git commit -m "feat: add Elo rating engine with FCS-aware baselines"
```

---

## Task 7: Feature builder

**Files:**
- Create: `cfb_picks/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `get_rating_as_of` (`cfb_picks.elo`), `get_connection`/`init_db` (`cfb_picks.db`).
- Produces: `build_features(conn, season, week, home_team, away_team, neutral_site=False) -> dict` with keys `elo_diff`, `success_rate_diff`, `ppa_diff`, `neutral_site`. Uses only `team_game_stats` rows from games strictly before `(season, week)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import pytest

from cfb_picks.db import get_connection, init_db
from cfb_picks.features import build_features


def test_build_features_uses_only_prior_data(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (1, 2024, 1, 'regular', NULL, 1, 0, 'A', 'B', 30, 10)"
    )
    conn.execute(
        "INSERT INTO team_game_stats (game_id, team, success_rate, ppa) VALUES (1, 'A', 0.6, 0.5)"
    )
    conn.execute(
        "INSERT INTO elo_ratings (season, week, team, rating) VALUES "
        "(2024, 0, 'A', 1600), (2024, 0, 'B', 1400)"
    )
    conn.commit()

    features_week2 = build_features(conn, 2024, 2, "A", "B")
    assert features_week2["elo_diff"] == pytest.approx(200)
    assert features_week2["success_rate_diff"] == pytest.approx(0.6)
    assert features_week2["ppa_diff"] == pytest.approx(0.5)
    assert features_week2["neutral_site"] == 0.0

    # Week 1's own stats must not count toward its own pre-game features.
    features_week1 = build_features(conn, 2024, 1, "A", "B")
    assert features_week1["success_rate_diff"] == pytest.approx(0.0)
    assert features_week1["ppa_diff"] == pytest.approx(0.0)


def test_build_features_passes_through_neutral_site(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    features = build_features(conn, 2024, 1, "A", "B", neutral_site=True)
    assert features["neutral_site"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.features'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/features.py
from .elo import get_rating_as_of

TRAILING_GAMES = 6


def _trailing_offense_stats(conn, season, week, team, n=TRAILING_GAMES):
    rows = conn.execute(
        """
        SELECT s.success_rate, s.ppa
        FROM team_game_stats s
        JOIN games g ON g.id = s.game_id
        WHERE s.team = ? AND (g.season < ? OR (g.season = ? AND g.week < ?))
        ORDER BY g.season DESC, g.week DESC
        LIMIT ?
        """,
        (team, season, season, week, n),
    ).fetchall()
    if not rows:
        return 0.0, 0.0
    avg_success_rate = sum(row["success_rate"] for row in rows) / len(rows)
    avg_ppa = sum(row["ppa"] for row in rows) / len(rows)
    return avg_success_rate, avg_ppa


def build_features(conn, season, week, home_team, away_team, neutral_site=False):
    home_elo = get_rating_as_of(conn, season, week, home_team)
    away_elo = get_rating_as_of(conn, season, week, away_team)
    home_success_rate, home_ppa = _trailing_offense_stats(conn, season, week, home_team)
    away_success_rate, away_ppa = _trailing_offense_stats(conn, season, week, away_team)
    return {
        "elo_diff": home_elo - away_elo,
        "success_rate_diff": home_success_rate - away_success_rate,
        "ppa_diff": home_ppa - away_ppa,
        "neutral_site": 1.0 if neutral_site else 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/features.py tests/test_features.py
git commit -m "feat: add feature builder"
```

---

## Task 8: Regression model (train/predict/persist)

**Files:**
- Create: `cfb_picks/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `build_features` (`cfb_picks.features`), `is_fbs_team` (`cfb_picks.elo`).
- Produces: `FEATURE_ORDER = ["elo_diff", "success_rate_diff", "ppa_diff", "neutral_site"]`, `ModelWeights` dataclass (`intercept: float`, `coefficients: dict[str, float]`), `train_model(feature_dicts, margins, alpha=1.0) -> ModelWeights`, `predict_margin(weights, features) -> float`, `save_model(weights, path) -> None`, `load_model(path) -> ModelWeights`, `gather_training_data(conn, seasons) -> (list[dict], list[float])` (excludes any game where either team is not FBS, since CFBD games include FBS-vs-FCS matchups that would otherwise skew the regression target).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import pytest

from cfb_picks.db import get_connection, init_db
from cfb_picks.model import (
    FEATURE_ORDER,
    gather_training_data,
    load_model,
    predict_margin,
    save_model,
    train_model,
)


def test_train_and_predict_recovers_simple_relationship():
    feature_dicts = [
        {"elo_diff": 100, "success_rate_diff": 0, "ppa_diff": 0, "neutral_site": 0},
        {"elo_diff": -100, "success_rate_diff": 0, "ppa_diff": 0, "neutral_site": 0},
        {"elo_diff": 200, "success_rate_diff": 0, "ppa_diff": 0, "neutral_site": 0},
        {"elo_diff": 0, "success_rate_diff": 0, "ppa_diff": 0, "neutral_site": 0},
    ]
    margins = [10, -10, 20, 0]

    weights = train_model(feature_dicts, margins, alpha=0.001)

    assert weights.coefficients["elo_diff"] == pytest.approx(0.1, abs=0.02)
    predicted = predict_margin(
        weights, {"elo_diff": 150, "success_rate_diff": 0, "ppa_diff": 0, "neutral_site": 0}
    )
    assert predicted == pytest.approx(15, abs=3)


def test_save_and_load_model_roundtrip(tmp_path):
    weights = train_model(
        [{"elo_diff": 100, "success_rate_diff": 0.1, "ppa_diff": 0.2, "neutral_site": 0}],
        [10],
    )
    path = tmp_path / "model.json"

    save_model(weights, path)
    loaded = load_model(path)

    assert loaded.intercept == pytest.approx(weights.intercept)
    assert loaded.coefficients == pytest.approx(weights.coefficients)


def test_gather_training_data_builds_features_and_margins(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('A', 'fbs'), ('B', 'fbs')")
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (1, 2024, 1, 'regular', NULL, 1, 0, 'A', 'B', 30, 10)"
    )
    conn.commit()

    feature_dicts, margins = gather_training_data(conn, [2024])

    assert margins == [20]
    assert set(feature_dicts[0].keys()) == set(FEATURE_ORDER)


def test_gather_training_data_excludes_fbs_vs_fcs_games(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('A', 'fbs')")
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (1, 2024, 1, 'regular', NULL, 1, 0, 'A', 'Some FCS School', 50, 3)"
    )
    conn.commit()

    feature_dicts, margins = gather_training_data(conn, [2024])

    assert margins == []
    assert feature_dicts == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.model'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/model.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/model.py tests/test_model.py
git commit -m "feat: add ridge regression model training/persistence"
```

---

## Task 9: Calibration module

**Files:**
- Create: `cfb_picks/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `edge_bucket(abs_edge) -> str` (buckets: `"0-2"` for `< 2`, `"2-5"` for `< 5`, `"5-8"` for `< 8`, `"8+"` otherwise), `save_calibration(calibration, path) -> None`, `load_calibration(path) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibration.py
from cfb_picks.calibration import edge_bucket, load_calibration, save_calibration


def test_edge_bucket_boundaries():
    assert edge_bucket(0) == "0-2"
    assert edge_bucket(1.9) == "0-2"
    assert edge_bucket(2) == "2-5"
    assert edge_bucket(4.9) == "2-5"
    assert edge_bucket(5) == "5-8"
    assert edge_bucket(7.9) == "5-8"
    assert edge_bucket(8) == "8+"
    assert edge_bucket(100) == "8+"


def test_save_and_load_calibration_roundtrip(tmp_path):
    calibration = {"0-2": 0.51, "2-5": 0.55, "5-8": 0.61, "8+": 0.58}
    path = tmp_path / "calibration.json"

    save_calibration(calibration, path)

    assert load_calibration(path) == calibration
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.calibration'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/calibration.py
import json

EDGE_BUCKETS = [(2, "0-2"), (5, "2-5"), (8, "5-8"), (float("inf"), "8+")]


def edge_bucket(abs_edge):
    for upper, name in EDGE_BUCKETS:
        if abs_edge < upper:
            return name
    return EDGE_BUCKETS[-1][1]


def save_calibration(calibration, path):
    with open(path, "w") as f:
        json.dump(calibration, f)


def load_calibration(path):
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/calibration.py tests/test_calibration.py
git commit -m "feat: add edge calibration module"
```

---

## Task 10: Pick logic (edge + calibrated best pick)

**Files:**
- Create: `cfb_picks/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `edge_bucket` (`cfb_picks.calibration`).
- Produces: `Pick` dataclass (`home_team`, `away_team`, `spread`, `predicted_margin`, `edge`, `pick_team`, `is_best_pick`), `compute_edge(predicted_margin, spread) -> float`, `make_picks(games: list[dict], calibration: dict | None = None) -> list[Pick]` (each `games` dict has `home_team`, `away_team`, `spread`, `predicted_margin`). Best pick is the one maximizing `abs(edge) * calibration[edge_bucket(abs(edge))]` when `calibration` is provided and has a value for that bucket; otherwise falls back to plain `abs(edge)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predict.py
import pytest

from cfb_picks.predict import compute_edge, make_picks


def test_compute_edge_home_favorite():
    assert compute_edge(predicted_margin=10, spread=-3) == pytest.approx(7)


def test_compute_edge_home_underdog():
    assert compute_edge(predicted_margin=-1, spread=6) == pytest.approx(5)


def test_make_picks_selects_side_and_best_pick_by_raw_edge_without_calibration():
    games = [
        {"home_team": "A", "away_team": "B", "spread": -3, "predicted_margin": 10},
        {"home_team": "C", "away_team": "D", "spread": 6, "predicted_margin": -1},
    ]

    picks = make_picks(games)

    assert picks[0].pick_team == "A"
    assert picks[0].edge == pytest.approx(7)
    assert picks[1].pick_team == "C"
    assert picks[1].edge == pytest.approx(5)
    assert picks[0].is_best_pick is True
    assert picks[1].is_best_pick is False


def test_make_picks_best_pick_uses_calibration_when_provided():
    games = [
        {"home_team": "A", "away_team": "B", "spread": -1, "predicted_margin": 7},  # edge 6 -> "5-8"
        {"home_team": "C", "away_team": "D", "spread": -1, "predicted_margin": 10},  # edge 9 -> "8+"
    ]
    calibration = {"5-8": 0.9, "8+": 0.3}

    picks = make_picks(games, calibration=calibration)

    # Raw edge favors the second game (9 > 6), but calibrated confidence
    # (6*0.9=5.4 vs 9*0.3=2.7) favors the first.
    assert picks[0].is_best_pick is True
    assert picks[1].is_best_pick is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.predict'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/predict.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_predict.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/predict.py tests/test_predict.py
git commit -m "feat: add pick/edge logic with calibrated best-pick selection"
```

---

## Task 11: Walk-forward backtest harness

**Files:**
- Create: `cfb_picks/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `edge_bucket` (`cfb_picks.calibration`), `is_fbs_team` (`cfb_picks.elo`), `build_features` (`cfb_picks.features`), `gather_training_data`/`predict_margin`/`train_model` (`cfb_picks.model`), `compute_edge` (`cfb_picks.predict`).
- Produces: `run_backtest(conn, seasons) -> list[dict]` (each dict has `edge: float`, `correct: bool`). For each season being evaluated, a regression model is trained **only on strictly earlier seasons** in the given list (out-of-sample) — the earliest season in the list is skipped since it has no prior training data. `summarize_backtest(results) -> dict` (`overall_accuracy`, `by_edge_bucket` — only buckets that actually occurred, `n`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest.py
import pytest

from cfb_picks.backtest import run_backtest, summarize_backtest
from cfb_picks.db import get_connection, init_db
from cfb_picks.elo import compute_elo_history

GAME_COLUMNS = (
    "id, season, week, season_type, start_date, completed, neutral_site, "
    "home_team, away_team, home_points, away_points"
)


def _insert_game(conn, row):
    conn.execute(f"INSERT INTO games ({GAME_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)


def test_run_backtest_trains_only_on_prior_seasons(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('A', 'fbs'), ('B', 'fbs')")
    _insert_game(conn, (1, 2023, 1, "regular", None, 1, 0, "A", "B", 50, 0))
    _insert_game(conn, (2, 2024, 1, "regular", None, 1, 0, "A", "B", 30, 10))
    conn.execute("INSERT INTO betting_lines (game_id, provider, spread) VALUES (2, 'consensus', -3)")
    conn.commit()
    compute_elo_history(conn, [2023, 2024])

    results = run_backtest(conn, [2023, 2024])

    # 2023 has no prior season to train on and is skipped; only 2024 is evaluated.
    assert len(results) == 1


def test_run_backtest_skips_games_without_a_market_spread(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute("INSERT INTO teams (school, classification) VALUES ('A', 'fbs'), ('B', 'fbs')")
    _insert_game(conn, (1, 2023, 1, "regular", None, 1, 0, "A", "B", 50, 0))
    _insert_game(conn, (2, 2024, 1, "regular", None, 1, 0, "A", "B", 30, 10))
    conn.commit()
    compute_elo_history(conn, [2023, 2024])

    results = run_backtest(conn, [2023, 2024])

    assert results == []


def test_summarize_backtest_buckets_by_edge_size():
    results = [
        {"edge": 1, "correct": True},
        {"edge": 9, "correct": True},
        {"edge": 9, "correct": False},
    ]

    summary = summarize_backtest(results)

    assert summary["overall_accuracy"] == pytest.approx(2 / 3)
    assert summary["by_edge_bucket"]["0-2"] == pytest.approx(1.0)
    assert summary["by_edge_bucket"]["8+"] == pytest.approx(0.5)
    assert summary["n"] == 3


def test_summarize_backtest_handles_empty_results():
    summary = summarize_backtest([])
    assert summary["overall_accuracy"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.backtest'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/backtest.py
from .calibration import edge_bucket
from .elo import is_fbs_team
from .features import build_features
from .model import gather_training_data, predict_margin, train_model
from .predict import compute_edge


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
        home_covered = actual_margin > -spread
        picked_home = edge > 0
        results.append({"edge": edge, "correct": picked_home == home_covered})

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/backtest.py tests/test_backtest.py
git commit -m "feat: add out-of-sample walk-forward backtest harness"
```

---

## Task 12: Results grading

**Files:**
- Create: `cfb_picks/results.py`
- Test: `tests/test_results.py`

**Interfaces:**
- Consumes: `get_connection`/`init_db` (`cfb_picks.db`).
- Produces: `grade_week(conn, season, week) -> None` — for every row in `picks` matching `(season, week)`, finds the matching completed `games` row and sets `result` to `"win"` or `"loss"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_results.py
from cfb_picks.db import get_connection, init_db
from cfb_picks.results import grade_week


def test_grade_week_marks_win_and_loss(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (1, 2024, 1, 'regular', NULL, 1, 0, 'A', 'B', 30, 10)"
    )
    conn.execute(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) "
        "VALUES (2024, 1, 'A', 'B', -3, 10, 7, 'A', 1, NULL)"
    )
    conn.execute(
        "INSERT INTO games (id, season, week, season_type, start_date, completed, "
        "neutral_site, home_team, away_team, home_points, away_points) "
        "VALUES (2, 2024, 1, 'regular', NULL, 1, 0, 'C', 'D', 10, 30)"
    )
    conn.execute(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) "
        "VALUES (2024, 1, 'C', 'D', -3, 5, 8, 'C', 0, NULL)"
    )
    conn.commit()

    grade_week(conn, 2024, 1)

    win_row = conn.execute("SELECT result FROM picks WHERE home_team = 'A'").fetchone()
    loss_row = conn.execute("SELECT result FROM picks WHERE home_team = 'C'").fetchone()
    assert win_row["result"] == "win"
    assert loss_row["result"] == "loss"


def test_grade_week_skips_ungraded_games(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    conn.execute(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) "
        "VALUES (2024, 1, 'A', 'B', -3, 10, 7, 'A', 1, NULL)"
    )
    conn.commit()

    grade_week(conn, 2024, 1)

    row = conn.execute("SELECT result FROM picks WHERE home_team = 'A'").fetchone()
    assert row["result"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_results.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.results'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/results.py
def grade_week(conn, season, week):
    picks = conn.execute(
        "SELECT rowid, * FROM picks WHERE season = ? AND week = ?", (season, week)
    ).fetchall()

    for pick in picks:
        game = conn.execute(
            "SELECT * FROM games WHERE season = ? AND week = ? AND home_team = ? "
            "AND away_team = ? AND completed = 1",
            (season, week, pick["home_team"], pick["away_team"]),
        ).fetchone()
        if game is None:
            continue

        actual_margin = game["home_points"] - game["away_points"]
        home_covered = actual_margin > -pick["spread"]
        picked_home = pick["pick_team"] == pick["home_team"]
        result = "win" if picked_home == home_covered else "loss"
        conn.execute("UPDATE picks SET result = ? WHERE rowid = ?", (result, pick["rowid"]))

    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_results.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/results.py tests/test_results.py
git commit -m "feat: add weekly results grading"
```

---

## Task 13: Season history reporting

**Files:**
- Create: `cfb_picks/history.py`
- Test: `tests/test_history.py`

**Interfaces:**
- Consumes: `get_connection`/`init_db` (`cfb_picks.db`).
- Produces: `season_record(conn, season=None) -> dict` with keys `points`, `picks`, `wins`, `ats_pct`, `best_pick_record` (string `"W/N"`), computed only from graded (`result IS NOT NULL`) picks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_history.py
import pytest

from cfb_picks.db import get_connection, init_db
from cfb_picks.history import season_record


def test_season_record_computes_points_and_pct(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    init_db(conn)
    rows = [
        (2024, 1, "A", "B", -3, 10, 7, "A", 1, "win"),
        (2024, 1, "C", "D", 3, -1, -4, "D", 0, "loss"),
        (2024, 2, "E", "F", -3, 10, 7, "E", 0, "win"),
        (2024, 2, "G", "H", -3, 10, 7, "G", 0, None),
    ]
    conn.executemany(
        "INSERT INTO picks (season, week, home_team, away_team, spread, predicted_margin, "
        "edge, pick_team, is_best_pick, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    record = season_record(conn, 2024)

    assert record["picks"] == 3
    assert record["wins"] == 2
    assert record["points"] == 3
    assert record["ats_pct"] == pytest.approx(2 / 3)
    assert record["best_pick_record"] == "1/1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.history'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/history.py
def season_record(conn, season=None):
    query = "SELECT * FROM picks WHERE result IS NOT NULL"
    params = []
    if season is not None:
        query += " AND season = ?"
        params.append(season)

    rows = conn.execute(query, params).fetchall()

    points = 0
    wins = 0
    best_pick_wins = 0
    best_pick_total = 0

    for row in rows:
        if row["result"] == "win":
            wins += 1
            points += 2 if row["is_best_pick"] else 1
        if row["is_best_pick"]:
            best_pick_total += 1
            if row["result"] == "win":
                best_pick_wins += 1

    total = len(rows)
    return {
        "points": points,
        "picks": total,
        "wins": wins,
        "ats_pct": (wins / total) if total else None,
        "best_pick_record": f"{best_pick_wins}/{best_pick_total}",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_history.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/history.py tests/test_history.py
git commit -m "feat: add season history reporting"
```

---

## Task 14: Config module

**Files:**
- Create: `cfb_picks/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `DB_PATH: Path`, `MODEL_PATH: Path`, `CALIBRATION_PATH: Path`, `SEASONS: list[int]` — computed dynamically as every year from 2021 through the current calendar year, excluding 2020, so the range grows automatically each season without a code change. All three paths are overridable via `CFB_PICKS_DB` / `CFB_PICKS_MODEL` / `CFB_PICKS_CALIBRATION` env vars. Loads `.env` via `python-dotenv` if present.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# cfb_picks/config.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cfb_picks/config.py tests/test_config.py
git commit -m "feat: add config module with dynamic season range"
```

---

## Task 15: CLI wiring

**Files:**
- Create: `cfb_picks/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 2-14.
- Produces: a Click group `cli` with commands `fetch-data`, `build-ratings`, `backtest`, `predict`, `record-results`, `history`, and a `main()` entry point (matches `pyproject.toml`'s `[project.scripts]`).
- `fetch-data [--seasons ...]`: defaults to the full `SEASONS` range but accepts a comma-separated override for cheap incremental refreshes mid-season.
- `build-ratings`: **always** recomputes Elo + regression + calibration over the full `SEASONS` range (no override) — a partial range would truncate `elo_ratings` and corrupt season-to-season carryover. Also runs the walk-forward backtest and saves the resulting edge-bucket calibration to `CALIBRATION_PATH`.
- `backtest`: always runs over the full `SEASONS` range; no longer needs a saved model since `run_backtest` fits its own per-season out-of-sample models.
- `predict --input <csv> --season <int> --week <int>`: reads a CSV with columns `home_team,away_team,spread` and an optional `neutral_site` column (`1`/`true` = neutral site), normalizes team names via `normalize_team_name`, builds features, predicts margins, computes picks via `make_picks` (loading calibration if available), persists them to the `picks` table (`INSERT OR REPLACE`), and prints a table with the best pick flagged.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfb_picks.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
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
            neutral_site = row.get("neutral_site", "0").strip().lower() in ("1", "true")
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
    for pick in picks:
        conn.execute(
            "INSERT OR REPLACE INTO picks (season, week, home_team, away_team, spread, "
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
    grade_week(conn, season, week)
    click.echo(f"Graded week {week}, {season}")


@cli.command("history")
@click.option("--season", default=None, type=int)
def history_cmd(season):
    conn = get_connection(DB_PATH)
    click.echo(season_record(conn, season))


def main():
    cli()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: All tests across every task PASS.

- [ ] **Step 6: Commit**

```bash
git add cfb_picks/cli.py tests/test_cli.py
git commit -m "feat: wire up CLI commands"
```

---

## Task 16: README with setup and weekly workflow

**Files:**
- Create: `README.md`

**Interfaces:**
- None (documentation only).

- [ ] **Step 1: Write `README.md`**

```markdown
# cfb-picks

Power-rating model for picking college football games against the spread.

## Setup

1. Get a free API key at https://collegefootballdata.com/key
2. Copy `.env.example` to `.env` and fill in `CFBD_API_KEY`.
3. `python -m venv .venv && source .venv/Scripts/activate` (or `.venv/bin/activate` on macOS/Linux)
4. `pip install -e ".[dev]"`
5. `pytest` to confirm everything passes.
6. `cfb-picks fetch-data` to pull historical data for every season from 2021 through the current one (a few dozen API calls total, well under the free 1,000/month limit).
7. `cfb-picks build-ratings` to compute Elo history, fit the regression model, and produce the edge-size calibration table.
8. `cfb-picks backtest` to see out-of-sample historical ATS accuracy before picking live.

## Weekly workflow

1. `cfb-picks fetch-data --seasons <current_year>` to pull any newly completed games/lines/stats for the current season (cheap, incremental).
2. `cfb-picks build-ratings` to recompute Elo through the latest results and refit the regression + calibration on the full history. **Run this every week** — it's what lets ratings reflect the season so far.
3. Paste the Splash slate (text or screenshot) into chat; it gets converted into a CSV with columns `home_team,away_team,spread` (spread is home-team-relative: negative = home favored) and an optional `neutral_site` column for games at a neutral site.
4. `cfb-picks predict --input slate.csv --season <year> --week <n>` — prints ranked picks with the best pick flagged (chosen by calibrated confidence, not just raw edge size), and saves them.
5. After games finish: `cfb-picks record-results --season <year> --week <n>`.
6. `cfb-picks history --season <year>` to see your running point total and ATS record.

**Note:** the calibration table is built from CFBD's consensus closing lines, which can differ from Splash's contest line (often set earlier and off-market). Treat calibration hit-rates as directional guidance about which edge sizes tend to be trustworthy, not exact live win probabilities.

## Other commands

- `cfb-picks fetch-data [--seasons 2021,2022,...]` — refresh CFBD data; defaults to the full range, or pass specific seasons for a cheap incremental update.
- `cfb-picks build-ratings` — recompute Elo history, refit the regression weights, and rebuild the calibration table over the full season range. No `--seasons` override — a partial range would corrupt season-to-season Elo carryover.
- `cfb-picks backtest` — run out-of-sample historical validation over the full season range.

## Tuning

`K_FACTOR`, `HOME_FIELD_ELO`, and `REGRESSION_FACTOR` (in `cfb_picks/elo.py`) and the ridge `alpha` (in `cfb_picks/model.py`'s `train_model`) are hand-tunable constants, not auto-tuned. After running `backtest`, if overall accuracy or edge-bucket calibration looks off, adjust these and re-run `build-ratings` + `backtest` to compare.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add setup and weekly workflow README"
```

---

## Self-Review Notes

- **Spec coverage:** data layer (Tasks 2-5), Elo + feature blend + regression (Tasks 6-8), calibration + pick/edge/best-pick logic (Tasks 9-10), out-of-sample backtesting/calibration (Task 11), weekly workflow/CLI (Tasks 14-15), history tracking (Tasks 12-13, 15). All spec sections have a corresponding task.
- **Known deviation from spec (surfaced to user before planning):** efficiency features are CFBD's actual advanced-stats fields (`successRate`, `ppa`) rather than yards/play and turnover margin, which CFBD's advanced-stats endpoint does not expose. Same intent, real data.
- **Post-brainstorm advisor review caught and fixed:** (1) `SEASONS` is now computed dynamically so the tool works for the current/live season, not just a hardcoded 2021-2025 range; `build-ratings` always rebuilds the full contiguous range rather than accepting a partial-range override that would corrupt Elo carryover. (2) Backtesting is now walk-forward/out-of-sample (train only on strictly prior seasons) instead of fitting and evaluating on the same data. (3) Elo baselines are FCS-aware (non-FBS opponents start at a lower baseline, and FBS-vs-FCS games are excluded from the regression training target) so blowout wins over overmatched non-FBS opponents don't skew ratings. (4) The "best pick" is now actually selected using the backtested edge-size calibration table (via a new `calibration.py` module), not just raw edge magnitude. (5) The weekly slate CSV supports an optional `neutral_site` column.
- **Type consistency:** `ModelWeights`, `Pick`, and all function signatures are used identically across Tasks 8-15 (verified `FEATURE_ORDER`, `predict_margin`, `make_picks`, `compute_edge`, `edge_bucket` signatures match between definition and every call site; no import cycles — `calibration.py` has no dependencies on `predict.py` or `backtest.py`, avoiding the cycle that would otherwise arise from `backtest.py` needing bucket logic and `predict.py` needing it too).
- **No placeholders:** every step has runnable code and concrete assertions; no TBD/TODO markers.
