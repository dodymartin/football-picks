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

        for game in games:
            ratings.setdefault(game["home_team"], _baseline_rating(conn, game["home_team"]))
            ratings.setdefault(game["away_team"], _baseline_rating(conn, game["away_team"]))

        for team, rating in ratings.items():
            conn.execute(
                "INSERT OR REPLACE INTO elo_ratings (season, week, team, rating) "
                "VALUES (?, 0, ?, ?)",
                (season, team, rating),
            )

        for game in games:
            home, away = game["home_team"], game["away_team"]
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
