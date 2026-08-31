# cfb-picks

Power-rating model for picking college football games against the spread.

## Setup

1. Get a free API key at https://collegefootballdata.com/key
2. Copy `.env.example` to `.env` and fill in `CFBD_API_KEY`.
3. `python -m venv .venv && source .venv/Scripts/activate` (or `.venv/bin/activate` on macOS/Linux)
4. `pip install -e ".[dev]"`
5. `pytest` to confirm everything passes.
6. `cfb-picks fetch-data` to pull historical data for every season from 2021 through the current one (a few dozen API calls total, well under the free 1,000/month limit).
7. `cfb-picks build-ratings` to compute Elo history, fit the regression model, and produce the edge-size calibration table.
8. `cfb-picks backtest` to see out-of-sample historical ATS accuracy before picking live.

## Weekly workflow

1. `cfb-picks fetch-data --seasons <current_year>` to pull any newly completed games/lines/stats for the current season (cheap, incremental).
2. `cfb-picks build-ratings` to recompute Elo through the latest results and refit the regression + calibration on the full history. **Run this every week** — it's what lets ratings reflect the season so far.
3. Paste the Splash slate (text or screenshot) into chat; it gets converted into a CSV with columns `home_team,away_team,spread` (spread is home-team-relative: negative = home favored) and an optional `neutral_site` column for games at a neutral site.
4. `cfb-picks predict --input slate.csv --season <year> --week <n>` — prints ranked picks with the best pick flagged (chosen by calibrated confidence, not just raw edge size), and saves them.
5. After games finish: `cfb-picks record-results --season <year> --week <n>`.
6. `cfb-picks history --season <year>` to see your running point total and ATS record.

**Note:** the calibration table is built from CFBD's consensus closing lines, which can differ from Splash's contest line (often set earlier and off-market). Treat calibration hit-rates as directional guidance about which edge sizes tend to be trustworthy, not exact live win probabilities.

## Other commands

- `cfb-picks fetch-data [--seasons 2021,2022,...]` — refresh CFBD data; defaults to the full range, or pass specific seasons for a cheap incremental update.
- `cfb-picks build-ratings` — recompute Elo history, refit the regression weights, and rebuild the calibration table over the full season range. No `--seasons` override — a partial range would corrupt season-to-season Elo carryover.
- `cfb-picks backtest` — run out-of-sample historical validation over the full season range.
- `cfb-picks top-picks --input <csv> --season <year> --week <n> [--count 10] [--output <path>]` — for betting outside the Splash contest: paste in lines for as many games as you want scanned (same CSV format as `predict`), and it ranks them all by calibrated confidence, prints the top N (default 10), and writes the full ranked list to CSV (default `top_picks_<season>_week<week>.csv`) for cross-reference. Read-only — never touches the `picks` table, so it can freely overlap with the contest slate. Track win/loss yourself in your betting app; this doesn't grade results.

## Tuning

`K_FACTOR`, `HOME_FIELD_ELO`, and `REGRESSION_FACTOR` (in `cfb_picks/elo.py`), `RECENCY_DECAY` (in `cfb_picks/features.py`), and the ridge `alpha` (in `cfb_picks/model.py`'s `train_model`) are hand-tunable constants, not auto-tuned. After running `backtest`, if overall accuracy or edge-bucket calibration looks off, adjust these and re-run `build-ratings` + `backtest` to compare.
