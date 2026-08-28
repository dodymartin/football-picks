def season_record(conn, season=None):
    query = "SELECT * FROM picks WHERE result IS NOT NULL"
    params = []
    if season is not None:
        query += " AND season = ?"
        params.append(season)

    rows = conn.execute(query, params).fetchall()

    points = 0
    wins = 0
    losses = 0
    best_pick_wins = 0
    best_pick_total = 0

    for row in rows:
        if row["result"] == "win":
            wins += 1
            points += 2 if row["is_best_pick"] else 1
        elif row["result"] == "loss":
            losses += 1
        # A push counts toward games played but contributes to neither the
        # win/loss tally nor points (including the best-pick bonus).
        if row["is_best_pick"] and row["result"] != "push":
            best_pick_total += 1
            if row["result"] == "win":
                best_pick_wins += 1

    total = len(rows)
    decided = wins + losses
    return {
        "points": points,
        "picks": total,
        "wins": wins,
        "ats_pct": (wins / decided) if decided else None,
        "best_pick_record": f"{best_pick_wins}/{best_pick_total}",
    }
