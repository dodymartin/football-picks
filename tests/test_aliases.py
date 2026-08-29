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
