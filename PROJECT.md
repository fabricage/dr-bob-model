# Master context — NFL predictive line model

You are building a local Python application that models NFL point spreads.
It is research/decision-support only — no bet placement, no accounts, no
hosting. Everything runs on localhost.

## Methodology (non-negotiable)

1. No stat enters the model until its predictive stability has been measured
   (odd/even split-half reliability and first-N-games → rest-of-season
   correlation). Turnovers are expected to fail these tests; efficiency
   stats (EPA/play, success rate, yards per play) are expected to pass.
2. All stats are opponent-adjusted ("compensated") before use, via ridge
   regression solving offense effect + opponent defense effect + home field
   simultaneously.
3. The blend weight between preseason prior and current-season data is
   derived empirically from the first-N study — it grows as games accumulate.
4. Exclude Week 18 from all training and study data (teams rest starters).
   Pre-2021 seasons had 16 games: exclude week 17 for those.
5. Injury/lineup adjustments are explicit point overrides applied AFTER the
   model line, with a reason and timestamp — never hidden inside ratings.

## Stack

Python 3.12, Polars, scikit-learn + scipy, nflreadpy (nflverse), Streamlit,
pytest, plain parquet files in `data/` (no database).

## Sign conventions

- `predicted_margin = home − away`
- `model_home_spread = −predicted_margin`
- `home_spread_edge = market_home_spread − model_home_spread`
- nflverse `spread_line` is PFR-style (positive = home favored).
  `market_home_spread = −spread_line`.

## Data

nflverse via `nflreadpy`. Play-by-play includes precomputed `epa`, `success`,
`air_yards`. Schedules include `spread_line`, `total_line`, and results.
Seasons 2015–2025.

## Rules for every change

- Every feature must be computable as-of the game date. Never use future
  data to predict past games.
- Keep functions small. No premature abstraction.
- Each new piece should have a runnable command, printed output, and a test.
