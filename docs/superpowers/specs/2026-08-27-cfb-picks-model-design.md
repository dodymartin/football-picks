# College Football ATS Picks Model — Design

## Purpose

Help pick winners against the spread (ATS) for a weekly college football contest. Each week the contest ("Splash") presents a 13-game slate with a spread for each game; the goal is to pick the side of the spread most likely to cover, plus designate one "best pick" for double points.

**Scoring**: 1 point per correct ATS pick, 2 points if the designated "best pick" is correct (i.e., +1 bonus over a normal correct pick).

## Constraints

- No paid data/odds API access. Historical/team data will come from collegefootballdata.com (CFBD), a free API (requires a free registration key).
- Weekly slate + spreads come from Splash (a private contest site); no scraping of Splash — the user will paste the slate text or a screenshot into chat each week, and it will be manually converted into a structured input file.
- Single user, run locally. No hosting/deployment needed.
- Python, delivered as a CLI application.

## Architecture

```
cfb-picks/
  data/          # CFBD ingestion + SQLite storage
  ratings/       # Elo engine + feature building + regression model
  backtest/      # historical replay & calibration
  cli.py         # command entry points
  cfb_picks.db   # SQLite database (gitignored)
```

Pieces:
- **Data layer**: pulls historical games, scores, team efficiency stats, and historical betting lines from CFBD; stores everything in a local SQLite database.
- **Rating/prediction engine**: Elo-style ratings updated game-by-game, blended with a handful of efficiency stats via a regression model trained on historical results, to produce a predicted scoring margin for any matchup.
- **Backtesting harness**: replays past seasons week-by-week to validate ATS accuracy and calibrate confidence in the predicted edge before using it live.
- **CLI**: commands to fetch/update data, rebuild ratings, backtest, generate weekly predictions from a slate, and record results/track history.

## Data Layer

Source: CFBD API (free tier, API key required).

SQLite tables:
- `games` — every FBS game, 2021-2025 (2020 excluded as COVID-disrupted), with final scores, week, season, home/away teams, neutral-site flag.
- `betting_lines` — historical closing spreads per game, for backtesting our model against real Vegas numbers.
- `team_game_stats` — per-team, per-game efficiency stats (yards/play, success rate, turnover margin, etc.) from CFBD's advanced stats endpoints.
- `team_aliases` — lookup table mapping alternate team name spellings (e.g., "Ohio St." vs "Ohio State") to a canonical CFBD team name, since Splash's naming may not exactly match CFBD's. Seeded from the FBS team list; extended as mismatches are discovered.

`fetch-data` command pulls new/missing data — full historical load initially, then incremental updates as each week's games complete during the season. Idempotent — safe to re-run.

## Rating & Prediction Engine

- **Elo core**: each team has a rating, starting from a baseline (1500) on first appearance in the data, carried over between seasons with regression toward the mean. Ratings update after each game based on actual margin vs. expected margin, with a home-field bonus and a margin-of-victory dampener (to avoid overreacting to blowouts). Elo recomputes incrementally as the season progresses, giving current-form ratings each week.
- **Feature blend**: for an upcoming game, compute a feature vector: Elo rating differential (with home-field baked in), plus trailing-window (last ~4-6 games) differentials in yards/play, success rate, and turnover margin.
- **Regression layer**: a ridge regression trained on all historical games maps the feature vector to a **predicted scoring margin**. This combines Elo and efficiency stats into one number with data-driven (not hand-tuned) weights.
- **Pick logic**: `edge = predicted_margin - spread` (sign-adjusted to a consistent convention). The side favored by the edge is the pick; the game with the largest absolute edge, scaled by how reliable that edge size has been historically (from backtest calibration), is the "best pick."

All computation for a historical game must use only data available before that game was played (no lookahead), so backtests honestly reflect what would have been knowable picking live.

## Backtesting & Calibration

Replay 2021-2025 week-by-week using CFBD's historical closing spreads:
- Compute ratings/features using only data available up to that point in time.
- Generate a predicted margin, compare to the actual historical spread to get an edge.
- Compare the implied pick to the actual outcome (ATS win/loss).

Produces:
- Overall ATS accuracy (sanity-check against ~52.4% breakeven, though this contest scores straight win/loss with no vig).
- Calibration by edge size (e.g., "edge >= 7 points hits 61% of the time; edge < 2 points hits 51%"), which drives trustworthy "best pick" selection.
- A tuning surface for Elo K-factor, home-field constant, and regression regularization strength — all validated before ever picking live.

## Weekly Workflow & CLI

Each week:
1. User pastes the Splash slate (text or screenshot) into chat; it's converted into a small structured CSV (`home_team, away_team, spread, week`).
2. `cfb-picks predict --input slate.csv` — normalizes team names via the alias table, computes predicted margins/edges/picks, flags the best pick, prints a ranked table.
3. `cfb-picks record-results --week N` — pulls final scores from CFBD (manual entry fallback if needed) and grades the week's picks.

Other commands:
- `cfb-picks fetch-data` — refresh CFBD data.
- `cfb-picks build-ratings` — recompute Elo history + refit regression weights.
- `cfb-picks backtest` — run historical validation (see above).
- `cfb-picks history` — show running record (points, ATS %, best-pick hit rate) across the season.

## History Tracking

A `picks` table in SQLite logs every week's picks: predicted margin, spread, edge, whether it was the designated best pick, and the graded outcome once known. The `history` command surfaces the season-to-date record so the model's real performance (not just backtest performance) can be monitored and used to inform re-tuning.

## Out of Scope (for this iteration)

- Scraping Splash or any sportsbook directly.
- Machine learning beyond linear/ridge regression (e.g., gradient boosting) — revisit only if simpler approach underperforms in backtesting.
- Injury reports, weather, travel/rest-day adjustments, recruiting/returning-production data — could be added as future features but not in the initial build.
- Multi-user support, web UI, hosting.
