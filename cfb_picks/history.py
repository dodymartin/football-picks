def season_record(conn, season=None):
    query = "SELECT * FROM picks WHERE result IS NOT NULL"
    params = []
    if season is not None:
        query += " AND season = ?"
        params.append(season)

    rows = conn.execute(query, params).fetchall()

    points = 0
    wins = 0
    best_pick_wins = 0
    best_pick_total = 0

    for row in rows:
        if row["result"] == "win":
            wins += 1
            points += 2 if row["is_best_pick"] else 1
        if row["is_best_pick"]:
            best_pick_total += 1
            if row["result"] == "win":
                best_pick_wins += 1

    total = len(rows)
    return {
        "points": points,
        "picks": total,
        "wins": wins,
        "ats_pct": (wins / total) if total else None,
        "best_pick_record": f"{best_pick_wins}/{best_pick_total}",
    }
