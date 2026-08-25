"""Download nflverse play-by-play and schedules into local parquet files.

Bronze layer = "as downloaded." We do not clean or aggregate here; that
happens later so we can rerun features without hitting the network.

Idempotent: if a season's parquet already exists, we skip it unless you
pass --force.
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
from helpers import print_frame, print_header, write_parquet


def _load_nflreadpy():
    """Import nflreadpy only when we actually need to download."""
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise SystemExit(
            "nflreadpy is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return nfl


def download_pbp(season: int, force: bool = False) -> Path:
    """Save one season of play-by-play to data/bronze/pbp/pbp_YYYY.parquet."""
    out = config.pbp_path(season)
    if out.exists() and not force:
        print(f"  skip pbp {season} (exists): {out}")
        return out
    print(f"  downloading pbp {season} ...")
    nfl = _load_nflreadpy()
    df = nfl.load_pbp(season)
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    write_parquet(df, out)
    print(f"  wrote {out} ({df.height:,} rows, {len(df.columns)} cols)")
    return out


def download_schedules(seasons: list[int], force: bool = False) -> Path:
    """Save schedules (lines + scores) for every configured season."""
    out = config.schedules_path()
    if out.exists() and not force:
        print(f"  skip schedules (exists): {out}")
        return out
    print("  downloading schedules ...")
    nfl = _load_nflreadpy()
    df = nfl.load_schedules(seasons)
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    # Keep the seasons we asked for even if the source returns extra years.
    df = df.filter(pl.col("season").is_in(seasons))
    write_parquet(df, out)
    print(f"  wrote {out} ({df.height:,} games)")
    return out


def load_schedules() -> pl.DataFrame:
    path = config.schedules_path()
    if not path.exists():
        raise FileNotFoundError(f"Schedules not found at {path}. Run ingest first.")
    return pl.read_parquet(path)


def pbp_row_counts() -> pl.DataFrame:
    """Count saved play-by-play rows per season (no full concat)."""
    rows = []
    for season in config.SEASONS:
        path = config.pbp_path(season)
        if not path.exists():
            rows.append({"season": season, "pbp_rows": None, "pbp_file": "MISSING"})
            continue
        n = pl.scan_parquet(path).select(pl.len()).collect().item()
        rows.append({"season": season, "pbp_rows": n, "pbp_file": path.name})
    return pl.DataFrame(rows)


def games_per_season(schedules: pl.DataFrame) -> pl.DataFrame:
    """Count regular-season games (all weeks, including the excluded finale)."""
    if "game_type" in schedules.columns:
        reg = schedules.filter(pl.col("game_type") == "REG")
    elif "season_type" in schedules.columns:
        reg = schedules.filter(pl.col("season_type") == "REG")
    else:
        reg = schedules
    return (
        reg.group_by("season")
        .agg(pl.len().alias("reg_games"))
        .sort("season")
    )


def week1_2024(schedules: pl.DataFrame) -> pl.DataFrame:
    """Printable 2024 Week 1 card so we can eyeball spread signs."""
    needed = ["season", "week", "game_id", "home_team", "away_team", "spread_line"]
    optional = [
        col
        for col in (
            "total_line",
            "home_score",
            "away_score",
            "gameday",
            "gametime",
            "result",
        )
        if col in schedules.columns
    ]
    week1 = (
        schedules.filter((pl.col("season") == 2024) & (pl.col("week") == 1))
        .select([c for c in needed + optional if c in schedules.columns])
        .sort("game_id")
    )
    # nflverse/PFR: positive spread_line means the HOME team is favored.
    # Betting shops print the opposite (home favorite = negative number).
    # We keep both so you can confirm the convention with your own eyes.
    if "spread_line" in week1.columns:
        week1 = week1.with_columns(
            (-pl.col("spread_line")).alias("market_home_spread"),
        )
    return week1


def print_summary() -> None:
    """Human-readable ingest report (the Phase 1 verify step)."""
    counts = pbp_row_counts()
    schedules = load_schedules()
    games = games_per_season(schedules)
    summary = counts.join(games, on="season", how="left")
    print_frame(summary, "Play-by-play rows and regular-season games per season")
    print_frame(
        week1_2024(schedules),
        "2024 Week 1 schedule with market spreads",
    )
    print(
        "\nSpread convention: nflverse spread_line > 0 means HOME is favored\n"
        "(Pro-Football-Reference style). market_home_spread = -spread_line\n"
        "is the betting-shop number used later in the model\n"
        "(predicted_margin = home - away; model_home_spread = -predicted_margin)."
    )


def missing_spread_after_2015(schedules: pl.DataFrame) -> pl.DataFrame:
    """Regular-season games from 2016+ with a null spread_line."""
    df = schedules.filter(pl.col("season") >= 2016)
    if "game_type" in df.columns:
        df = df.filter(pl.col("game_type") == "REG")
    return df.filter(pl.col("spread_line").is_null())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download nflverse bronze data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local parquet files already exist.",
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of seasons (default: all configured seasons).",
    )
    parser.add_argument(
        "--schedules-only",
        action="store_true",
        help="Skip play-by-play (useful for a quick line check).",
    )
    args = parser.parse_args(argv)

    seasons = args.seasons or list(config.SEASONS)
    config.ensure_data_dirs()
    print_header("NFL model — bronze ingest")
    print(f"seasons: {seasons}")
    download_schedules(seasons, force=args.force)
    if not args.schedules_only:
        for season in seasons:
            download_pbp(season, force=args.force)
    print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
