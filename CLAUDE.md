# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A power-rating model for picking college football games against the spread (ATS), built for a weekly pick'em contest called "Lucky 13." Single user, run locally, Python CLI (`cfb-picks`). No web UI, no hosting, no multi-user support — don't add any of that speculatively.

## Commands

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash; .venv/bin/activate elsewhere
pip install -e ".[dev]"

pytest                                  # full suite
pytest tests/test_predict.py            # one file
pytest tests/test_predict.py::test_rank_by_confidence_orders_highest_confidence_first   # one test

cfb-picks fetch-data [--seasons 2021,2022,...]   # refresh CFBD data; no arg = full configured range (see SEASONS in config.py)
cfb-picks build-ratings                          # recompute Elo history, refit regression, rebuild calibration — run every week, no --seasons override
cfb-picks backtest                               # out-of-sample historical validation over the full season range
cfb-picks predict --input slate.csv --season <y> --week <n>       # contest picks: writes to the `picks` table, best-pick logic is stateful (see below)
cfb-picks top-picks --input slate.csv --season <y> --week <n> [--count 10] [--output path]  # read-only ranked report, doesn't touch `picks`
cfb-picks record-results --season <y> --week <n>  # grade a week's picks against final scores
cfb-picks history --season <y>                    # season-to-date points/ATS%/best-pick record
```

There's no lint/format command configured — don't invent one.

## Architecture

Pipeline for turning a slate CSV into ranked picks (`cli.py` orchestrates, wired top-down):

1. **`db.py`** — SQLite schema and connection. Tables: `teams`, `team_aliases`, `games`, `betting_lines`, `team_game_stats`, `team_talent`, `elo_ratings`, `picks`. `data/*.db` and `data/*.json` are gitignored — every user runs their own local copy.
2. **`ingest.py` / `cfbd_client.py`** — pull games, historical closing spreads, per-team-per-game efficiency stats, and talent composite from the CollegeFootballData.com API into SQLite. Idempotent, incremental.
3. **`aliases.py`** — `normalize_team_name(conn, name)` maps a slate's spelling of a team to CFBD's canonical name via `team_aliases`; raises `ValueError` on an unmapped name rather than guessing. Sportsbook/contest slates routinely use different spellings/abbreviations than CFBD (`UMass`→`Massachusetts`, `Miami FL`→`Miami`, etc.) — expect to hit unmapped names on every new slate source and add aliases rather than editing the input by hand.
4. **`elo.py`** — Elo ratings computed incrementally season-by-season (`compute_elo_history`), with a margin-of-victory dampener, home-field bonus, and season-to-season regression toward the mean. FCS opponents get a much lower baseline rating (1200 vs 1500) via `is_fbs_team` — but only teams present in the `teams` table are recognized as FBS at all; many FCS opponents aren't in `teams` and have essentially no signal.
5. **`features.py`** — for a given (season, week, home, away), builds the feature vector: Elo differential, recency-weighted trailing-game efficiency stats (success rate, PPA), season talent-composite differential, neutral-site flag. Every lookup is strictly "as of before this week" — no lookahead, so backtests stay honest.
6. **`model.py`** — ridge regression (scikit-learn) trained on the feature vectors to predict scoring margin. Weights persist to `data/model_weights.json`.
7. **`predict.py`** — `compute_edge = predicted_margin - (-spread)`; the picked side is whichever team the edge favors. **Confidence ranking is `(calibrated_win_rate, |edge|)`, not `|edge| × win_rate`.** This was deliberately changed (see git history / `confidence_score`, `_rank_key`) after backtesting showed edge size is a weak predictor of win rate outside the extremes — the 0–2 point edge bucket has both the best historical win rate and the largest sample, while very large edges (25+) are backed by too few historical games to trust. Win rate is always the primary sort key; edge only breaks ties within a bucket. Any future change to ranking/confidence should go through the same backtest-bucket lens, not intuition about edge size.
8. **`calibration.py`** — buckets `|edge|` into fixed ranges (`0-2, 2-5, 5-8, 8-15, 15-25, 25+`) and looks up each bucket's historical ATS win rate from `data/calibration.json`, produced by `backtest.py`.
9. **`backtest.py`** — walks seasons in order, training only on strictly earlier seasons for each season's evaluation (true out-of-sample), and buckets results by edge size to produce the calibration table `build-ratings` saves.
10. **`results.py` / `history.py`** — grade a graded game's ATS result (`win`/`loss`/`push`) and roll up season points (1 per win, 2 for a correct best pick).

### The `predict` command's best-pick statefulness

`predict_cmd` in `cli.py` is the most intricate piece of logic in the codebase — read it fully before touching it. The contest rule is "best pick is locked once any game that week has been graded," so re-running `predict` (e.g. correcting one game's line) must not silently reassign the best pick after grading has started, must recompute it correctly across *all* of a week's ungraded picks (not just the ones in the current CSV) when nothing is graded yet, and must never touch already-graded rows. The DELETE/INSERT-OR-IGNORE dance and the `prior_best_pick`/`other_picks` bookkeeping all exist to satisfy those constraints — the inline comments explain each piece, but the constraints themselves aren't obvious from the code shape alone.

### Week numbering

`--week` must match CFBD's own week numbers, not calendar weeks or a slate's filename — a single CFBD week can span more than seven days (e.g. week 1 of a season can run from the preceding Thursday through the following Monday, covering what looks like two separate "weekends" of games). Before running `predict`/`top-picks` on a new slate, sanity-check a couple of its games against `SELECT week, MIN(start_date), MAX(start_date) FROM games WHERE season=? GROUP BY week` rather than assuming.

### Reading slate PDFs

When a slate arrives as a PDF (sportsbook lines dump or a contest app export), `pdftotext -layout` reflows odds-board columns unreliably. Use PyMuPDF (`import fitz`, `page.get_text('dict')`) for line-level text with bounding boxes instead, and validate every extracted spread with a sum-to-zero check (`away_spread + home_spread ≈ 0`) before feeding it to `predict`/`top-picks` — a silent pairing error produces a plausible-looking but wrong spread rather than an obvious failure. `pick-data/lucky-13/` holds the contest's own slate PDFs and derived pick CSVs; `pick-data/bets/` holds any separate sportsbook-lines analysis (via `top-picks`) done outside the contest.
