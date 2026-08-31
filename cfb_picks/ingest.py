from .aliases import seed_aliases


def upsert_teams(conn, teams_json):
    # Every team returned by /teams/fbs is FBS by construction, even though
    # the payload's `classification` field is nullable, so default it to
    # "fbs" when missing.
    rows = [
        (team["school"], team.get("conference"), team.get("classification") or "fbs")
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


def upsert_team_talent(conn, talent_json):
    rows = [(t["year"], t["team"], t.get("talent")) for t in talent_json]
    conn.executemany(
        "INSERT INTO team_talent (season, team, talent) VALUES (?, ?, ?) "
        "ON CONFLICT(season, team) DO UPDATE SET talent=excluded.talent",
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

        talent = client.get_talent(year)
        upsert_team_talent(conn, talent)
