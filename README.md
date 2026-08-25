# NFL Predictive Line Model

Local Python research tool that models NFL point spreads. It is
**decision-support only** — no bet placement, no accounts, no hosting.
Everything runs on localhost.

`predicted_margin = home − away`  
`model_home_spread = −predicted_margin`  
`home_spread_edge = market_home_spread − model_home_spread`

nflverse stores `spread_line` in Pro-Football-Reference style (positive
means the **home** team is favored). The model converts that to a betting
shop number with `market_home_spread = −spread_line`.

---

## How the model thinks (read this once)

1. **No stat enters the ratings until it has been shown to be stable.**
   We measure odd/even split-half reliability and first-N-games →
   rest-of-season correlation. Efficiency stats (EPA/play, success rate)
   usually pass. Turnovers usually fail — they are in the pipeline as
   *noise controls*, not as weapons.
2. **Every stat is opponent-adjusted** with ridge regression before it is
   used: offense effect + opponent defense effect + home field, fit
   simultaneously.
3. **The blend weight** between last year's prior and this year's games
   is taken from the first-N study. It grows as games accumulate.
4. **Week 18 is never used** (week 17 before 2021). Teams rest starters.
5. **Injury/lineup adjustments are explicit point overrides** applied
   *after* the model line, with a reason and a timestamp. They are never
   hidden inside the ratings.

---

## First-time setup on a Mac

Homebrew Python will not let `pip install` write into the system. That
error (`externally-managed-environment`) is a safety feature, not a
broken install. This project keeps its packages in a private folder
called `.venv` instead.

```bash
make setup      # once: creates .venv and installs packages
make app        # waits until the server is up, then opens http://127.0.0.1:8501
```

Leave that terminal window open. It is only the **server**. The model
itself is the web page at **http://127.0.0.1:8501**.

If a tab opens too early and says the site cannot be reached, wait until
the terminal prints `Server is up`, then refresh, or paste
`http://127.0.0.1:8501` into the address bar (use `127.0.0.1`, not
`localhost`).

If the board is empty the first time, use **Build data** in the sidebar
and click **First-time build**. That download can take several minutes.

If you ever run Python by hand in a new terminal:

```bash
source .venv/bin/activate
```

Python 3.12+ is fine (including Homebrew 3.14). Stack: Polars,
scikit-learn, nflreadpy, Streamlit, pytest.

---

## Weekly in-season routine

After the Sunday/Monday games are in nflverse (usually Monday night /
Tuesday):

```bash
make week
```

That command:

1. Re-downloads the **current** season's play-by-play and the full
   schedule file (lines move; scores land).
2. Rebuilds team-game stats and opponent-adjusted stats.
3. Rebuilds walk-forward model lines and writes `data/gold/board.parquet`.

Then open the dashboard:

```bash
make app
```

Add an injury override when a starter is ruled out (example: backup QB):

```bash
python -m model.overrides add \
  --team KC --season 2025 --week 6 \
  --points -6.5 --position QB \
  --reason "starter out, backup QB" \
  --source report --confidence high

python -m model.overrides list
python -m model.overrides expire --id <id>
```

`points` is that **team's** impact on the game. A `-6.5` on the home
team pulls the home margin down 6.5 points; on the away team it pushes
the home margin up 6.5 points. Raw and adjusted lines are both kept.

---

## Offseason routine

When a season is complete and you want the stability studies to include
it:

```bash
python run_week.py --full --force-pull
```

That re-runs odd/even reliability, the first-N weight curve, and the
shrunk priors, then refreshes the backtest.

Piece by piece (same as `make all`):

```bash
make pull        # nflverse -> data/bronze
make features    # team-game stats + ridge compensation
make research    # stability studies
make model       # priors, weights, sample board
make backtest    # walk-forward ATS / MAE report
make test        # pytest
make lint        # ruff
```

---

## Project layout

```
config.py              # seasons, week filter, candidate stats, paths
ingest/                # nflverse pulls -> bronze parquet
features/              # team-game aggregation, compensated stats
research/              # odd/even + first-N studies
model/                 # priors, blend weights, spread, overrides
backtest/              # walk-forward evaluation
app.py                 # Streamlit dashboard
run_week.py            # one-command refresh
data/                  # parquet + overrides.jsonl (gitignored)
tests/
```

`data/` is rebuilt from nflverse. Do not commit parquet files.

---

## Tests

```bash
python -m pytest
```

Unit tests use a tiny synthetic league so they do not need a download.
Once you have run `make pull`, an extra test asserts regular-season
`spread_line` is present after 2015.

---

## What the studies showed (2015–2025 nflverse)

These are the numbers the pipeline printed on real data — the reason
the methodology exists:

| Stat | Split-half r (Spearman–Brown) | First-N r (N=8) |
| --- | --- | --- |
| Offensive success rate | 0.77 | 0.58 |
| Offensive EPA/play | 0.74 | 0.56 |
| Turnover margin | 0.28 | 0.17 |
| Fumbles lost | ~0 | ~0 |

Efficiency is a trait. Turnovers are mostly luck. The 2025 preseason
prior top-10 by net EPA was BAL, BUF, DET, PHI, GB — last year's good
teams, shrunk toward average. Walk-forward MAE vs the closing spread
is about 2.6 points; ATS when the model disagrees by 1.5+ points sits
near 49% pooled, which is what you should expect from a first EPA-only
line against a sharp market.

---

## Not built yet

- Live odds feed (The Odds API) replacing schedule lines on the board
- Closing-line-value tracking per model pick
- QB-specific rating adjustment learned from historical QB out/in games
- Totals model reusing the same compensated-stat machinery
