from .elo import get_rating_as_of

TRAILING_GAMES = 6
RECENCY_DECAY = 0.85


def _trailing_offense_stats(conn, season, week, team, n=TRAILING_GAMES, decay=RECENCY_DECAY):
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
    # Rows are ordered most-recent-first, so weight[0] (the most recent
    # game) gets the largest weight and it decays geometrically from there.
    weights = [decay**i for i in range(len(rows))]
    total_weight = sum(weights)
    avg_success_rate = sum(w * row["success_rate"] for w, row in zip(weights, rows)) / total_weight
    avg_ppa = sum(w * row["ppa"] for w, row in zip(weights, rows)) / total_weight
    return avg_success_rate, avg_ppa


def _talent(conn, season, team):
    row = conn.execute(
        "SELECT talent FROM team_talent WHERE season = ? AND team = ?", (season, team)
    ).fetchone()
    if row is None or row["talent"] is None:
        return 0.0
    return row["talent"]


def build_features(conn, season, week, home_team, away_team, neutral_site=False):
    home_elo = get_rating_as_of(conn, season, week, home_team)
    away_elo = get_rating_as_of(conn, season, week, away_team)
    home_success_rate, home_ppa = _trailing_offense_stats(conn, season, week, home_team)
    away_success_rate, away_ppa = _trailing_offense_stats(conn, season, week, away_team)
    home_talent = _talent(conn, season, home_team)
    away_talent = _talent(conn, season, away_team)
    return {
        "elo_diff": home_elo - away_elo,
        "success_rate_diff": home_success_rate - away_success_rate,
        "ppa_diff": home_ppa - away_ppa,
        "talent_diff": home_talent - away_talent,
        "neutral_site": 1.0 if neutral_site else 0.0,
    }
