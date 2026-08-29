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
