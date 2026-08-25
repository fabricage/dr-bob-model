"""One-command weekly refresh: pull latest data, rebuild features, refresh the board.

In-season you usually:
  1. pull the current season's play-by-play / schedules (force, because
     nflverse updates after each game)
  2. rebuild team-game stats and opponent-adjusted stats
  3. rebuild model lines (priors stay put unless you are in the offseason)

Offseason you re-run the stability studies after the new season is in
the bronze layer (see README).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

import config
from backtest.walk_forward import run_backtest
from features.compensated import compensate_all, load_raw_team_games, print_verify
from features.team_games import build_team_game_stats, sanity_print
from helpers import print_frame, print_header, write_parquet
from ingest.pull_nflverse import download_pbp, download_schedules, print_summary
from model.priors import build_priors, load_effects, load_split_half
from model.priors import print_verify as print_priors
from model.spread import walk_forward_games, week_board
from model.weights import load_prior_r, load_weight_curve, weight_curve_table
from research.stability_firstn import firstn_grid, load_compensated, print_grid
from research.stability_oddeven import split_half_table


def pull(seasons: list[int], force: bool) -> None:
    config.ensure_data_dirs()
    download_schedules(seasons, force=force)
    for season in seasons:
        download_pbp(season, force=force)
    print_summary()


def rebuild_features(seasons: list[int] | None = None) -> None:
    df = build_team_game_stats(seasons)
    write_parquet(df, config.team_game_stats_raw_path())
    sanity_print(df)
    games, effects = compensate_all(load_raw_team_games())
    write_parquet(games, config.team_game_stats_path())
    write_parquet(effects, config.team_effects_path())
    print_verify(load_raw_team_games(), effects)


def rebuild_research() -> None:
    games = load_compensated()
    odd = split_half_table(games)
    write_parquet(odd, config.stability_oddeven_path())
    print_frame(odd.with_columns(pl.col(pl.Float64).round(3)), "Odd/even reliability")
    grid = firstn_grid(games)
    write_parquet(grid, config.weight_curve_path())
    print_grid(grid)


def rebuild_priors() -> None:
    priors, prior_r = build_priors(load_effects(), load_split_half())
    write_parquet(priors, config.priors_path())
    write_parquet(prior_r, config.prior_reliability_path())
    print_priors(priors, prior_r)
    table = weight_curve_table(load_weight_curve(), load_prior_r(), "off_epa_per_play")
    print_frame(
        table.with_columns(pl.col(pl.Float64).round(3)),
        "Blend weight curve (off_epa_per_play)",
    )


def rebuild_board(season: int | None, week: int | None) -> None:
    scored, coefs = walk_forward_games()
    write_parquet(scored, config.backtest_games_path())
    write_parquet(coefs, config.spread_coefficients_path())
    _, report = run_backtest()
    write_parquet(report, config.backtest_path())
    if season is None:
        season = int(scored.get_column("season").max())
    if week is None:
        week = int(scored.filter(pl.col("season") == season).get_column("week").max())
    board = week_board(season, week, scored=scored)
    write_parquet(board, config.board_path())
    keep = [
        c
        for c in (
            "away_team",
            "home_team",
            "market_home_spread",
            "model_home_spread",
            "adjusted_home_spread",
            "home_spread_edge",
            "n_overrides",
            "override_reasons",
            "actual_margin",
        )
        if c in board.columns
    ]
    print_frame(
        board.select(keep).with_columns(pl.col(pl.Float64).round(2)),
        f"Board {season} week {week}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weekly (or full) NFL model refresh.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-run stability studies and priors (offseason routine).",
    )
    parser.add_argument(
        "--force-pull",
        action="store_true",
        help="Re-download nflverse files even if they exist.",
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=None,
        help="Seasons to pull/rebuild (default: current max configured season for weekly, all for --full).",
    )
    parser.add_argument("--board-season", type=int, default=None)
    parser.add_argument("--board-week", type=int, default=None)
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Rebuild from local bronze files only.",
    )
    args = parser.parse_args(argv)

    print_header("run_week")
    if args.full:
        seasons = args.seasons or list(config.SEASONS)
    else:
        seasons = args.seasons or [max(config.SEASONS)]

    if not args.skip_pull:
        # Always refresh schedules (small). Force PBP for the seasons we care about.
        pull(seasons if not args.full else list(config.SEASONS), force=args.force_pull or not args.full)

    # Features: weekly can rebuild just the pulled seasons, but compensated ridge
    # is per-season so rebuilding only those seasons would drop others from the
    # parquet. Rebuild ALL seasons that have bronze pbp on disk.
    rebuild_features(None)
    if args.full or not config.stability_oddeven_path().exists() or not config.priors_path().exists():
        rebuild_research()
        rebuild_priors()
    rebuild_board(args.board_season, args.board_week)
    print("\nDone. Launch the dashboard with:  python -m streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
