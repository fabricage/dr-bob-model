"""Odd/even split-half reliability for every candidate stat.

Question this script answers: "If I only saw a team's odd-numbered weeks,
how well would that predict even-numbered weeks in the SAME season?"

A stable, skill-like stat (EPA/play, success rate) should correlate.
A noisy stat (fumbles lost, turnover margin) should not.

We then apply the Spearman–Brown correction, which estimates how reliable
the FULL season average would be if the two halves correlate at r:
    r_full = 2r / (1 + r)
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
from helpers import pearson_r, print_frame, print_header, spearman_brown, write_parquet


def load_compensated() -> pl.DataFrame:
    path = config.team_game_stats_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python features/compensated.py")
    return pl.read_parquet(path)


def split_half_table(games: pl.DataFrame, stats: list[str] | None = None) -> pl.DataFrame:
    stats = stats or [
        c for c in config.all_model_stat_columns() if f"{c}_comp" in games.columns
    ]
    rows = []
    for stat in stats:
        col = f"{stat}_comp" if f"{stat}_comp" in games.columns else stat
        odd = (
            games.filter((pl.col("week") % 2) == 1)
            .group_by(["season", "team"])
            .agg(
                pl.col(col).mean().alias("odd_mean"),
                pl.len().alias("odd_n"),
            )
        )
        even = (
            games.filter((pl.col("week") % 2) == 0)
            .group_by(["season", "team"])
            .agg(
                pl.col(col).mean().alias("even_mean"),
                pl.len().alias("even_n"),
            )
        )
        paired = odd.join(even, on=["season", "team"]).filter(
            (pl.col("odd_n") >= 3) & (pl.col("even_n") >= 3)
        )
        r = pearson_r(
            paired.get_column("odd_mean").to_numpy(),
            paired.get_column("even_mean").to_numpy(),
        )
        rows.append(
            {
                "stat": stat,
                "n_team_seasons": paired.height,
                "r_split_half": r,
                "r_spearman_brown": spearman_brown(r),
            }
        )
    return pl.DataFrame(rows).sort("r_spearman_brown", descending=True, nulls_last=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Odd/even stability study.")
    parser.parse_args(argv)
    games = load_compensated()
    table = split_half_table(games)
    write_parquet(table, config.stability_oddeven_path())
    print_header("Odd/even split-half reliability (compensated stats)")
    printable = table.with_columns(
        pl.col("r_split_half").round(3),
        pl.col("r_spearman_brown").round(3),
    )
    print_frame(printable)
    print(
        "\nExpected shape: epa_per_play and success_rate should sit WELL ABOVE\n"
        "fumbles_lost and turnover_margin. That is the whole point of this study:\n"
        "efficiency is a trait; turnovers are mostly luck until a huge sample."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
