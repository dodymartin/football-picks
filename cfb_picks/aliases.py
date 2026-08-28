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
