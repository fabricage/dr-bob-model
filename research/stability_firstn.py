"""First-N-games → rest-of-season correlations (the blend-weight curve).

Question this script answers: "After N games, how much should I trust
this year's numbers vs last year's prior?"

For each N in {3,4,5,6,8,10,12} we correlate:
    team's compensated average over games 1..N
with
    the same team's average over games N+1 .. end of usable season

That r(N) is the current-season reliability used in Phase 5:
    weight = r_current² / (r_current² + r_prior²)

r should rise with N — more games, more signal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

import config
from helpers import pearson_r, print_frame, print_header, write_parquet


def load_compensated() -> pl.DataFrame:
    path = config.team_game_stats_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python features/compensated.py")
    return pl.read_parquet(path)


def _order_games(games: pl.DataFrame) -> pl.DataFrame:
    """Number each team's games 1..G in chronological order (week, then date)."""
    sort_cols = ["season", "team", "week"]
    if "game_date" in games.columns:
        sort_cols.append("game_date")
    sort_cols.append("game_id")
    return games.sort(sort_cols).with_columns(
        (pl.int_range(pl.len()).over(["season", "team"]) + 1).alias("game_number")
    )


def firstn_grid(
    games: pl.DataFrame,
    ns: list[int] | None = None,
    stats: list[str] | None = None,
) -> pl.DataFrame:
    ns = ns or list(config.FIRST_N_VALUES)
    numbered = _order_games(games)
    stats = stats or [
        c for c in config.all_model_stat_columns() if f"{c}_comp" in numbered.columns
    ]
    rows = []
    for stat in stats:
        col = f"{stat}_comp"
        for n in ns:
            early = (
                numbered.filter(pl.col("game_number") <= n)
                .group_by(["season", "team"])
                .agg(
                    pl.col(col).mean().alias("early_mean"),
                    pl.len().alias("early_n"),
                )
                .filter(pl.col("early_n") == n)
            )
            late = (
                numbered.filter(pl.col("game_number") > n)
                .group_by(["season", "team"])
                .agg(
                    pl.col(col).mean().alias("late_mean"),
                    pl.len().alias("late_n"),
                )
                .filter(pl.col("late_n") >= 3)
            )
            paired = early.join(late, on=["season", "team"])
            r = pearson_r(
                paired.get_column("early_mean").to_numpy(),
                paired.get_column("late_mean").to_numpy(),
            )
            rows.append(
                {
                    "stat": stat,
                    "n_games": n,
                    "n_team_seasons": paired.height,
                    "r_first_n": r,
                }
            )
    return pl.DataFrame(rows).sort(["stat", "n_games"])


def print_grid(grid: pl.DataFrame) -> None:
    """Wide table: rows = stats, columns = N, values = r."""
    wide = (
        grid.with_columns(pl.col("r_first_n").round(3))
        .pivot(values="r_first_n", index="stat", on="n_games", sort_columns=True)
        .sort("stat")
    )
    print_frame(wide, "r(first N games, rest of season) by stat")
    print(
        "\nExpected shape: correlations should generally INCREASE as N grows,\n"
        "and EPA / success rate should beat fumbles_lost / turnover_margin at\n"
        "every N. Those r values become the in-season blend-weight curve."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="First-N stability / weight curve.")
    parser.parse_args(argv)
    games = load_compensated()
    grid = firstn_grid(games)
    write_parquet(grid, config.weight_curve_path())
    print_header("First-N → rest-of-season correlations")
    print_grid(grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
