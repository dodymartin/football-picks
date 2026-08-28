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
