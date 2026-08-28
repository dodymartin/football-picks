def grade_ats(actual_margin, spread, picked_home):
    """Grade a single against-the-spread pick.

    `actual_margin` is home_points - away_points. `spread` is home-team-relative
    (negative means home favored). Returns "win", "loss", or "push" when the
    actual margin exactly matches the spread (neither side covers).
    """
    if actual_margin == -spread:
        return "push"
    home_covered = actual_margin > -spread
    return "win" if picked_home == home_covered else "loss"


def grade_week(conn, season, week):
    picks = conn.execute(
        "SELECT rowid, * FROM picks WHERE season = ? AND week = ?", (season, week)
    ).fetchall()

    graded_count = 0
    for pick in picks:
        game = conn.execute(
            "SELECT * FROM games WHERE season = ? AND week = ? AND home_team = ? "
            "AND away_team = ? AND completed = 1",
            (season, week, pick["home_team"], pick["away_team"]),
        ).fetchone()
        if game is None:
            continue

        actual_margin = game["home_points"] - game["away_points"]
        picked_home = pick["pick_team"] == pick["home_team"]
        result = grade_ats(actual_margin, pick["spread"], picked_home)
        conn.execute("UPDATE picks SET result = ? WHERE rowid = ?", (result, pick["rowid"]))
        graded_count += 1

    conn.commit()
    return graded_count
