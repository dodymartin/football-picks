from .elo import get_rating_as_of

TRAILING_GAMES = 6


def _trailing_offense_stats(conn, season, week, team, n=TRAILING_GAMES):
    rows = conn.execute(
        """
        SELECT s.success_rate, s.ppa
        FROM team_game_stats s
        JOIN games g ON g.id = s.game_id
        WHERE s.team = ? AND (g.season < ? OR (g.season = ? AND g.week < ?))
        AND s.success_rate IS NOT NULL AND s.ppa IS NOT NULL
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
