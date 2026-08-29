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
