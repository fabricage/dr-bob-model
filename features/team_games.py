"""Collapse play-by-play into one row per team per game.

Each NFL game becomes TWO rows (home and away) so later steps can treat
"what a team did" and "what a team allowed" as regular columns.

We drop kneels, spikes, and no-play penalties before averaging. Those
plays are not real attempts to gain yards, so they would distort EPA/play
and success rate. Playoff games and the final regular-season week are
also dropped (see config.last_usable_week).
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
from ingest.pull_nflverse import load_schedules

# Columns we actually use. Reading only these keeps memory small enough to
# process one season at a time on a laptop.
PBP_COLUMNS = [
    "season",
    "week",
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "success",
    "yards_gained",
    "rushing_yards",
    "passing_yards",
    "sack",
    "interception",
    "fumble_lost",
    "qb_kneel",
    "qb_spike",
    "two_point_attempt",
    "season_type",
]


def usable_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Regular-season games in weeks we are allowed to train on."""
    df = schedules
    if "game_type" in df.columns:
        df = df.filter(pl.col("game_type") == "REG")
    elif "season_type" in df.columns:
        df = df.filter(pl.col("season_type") == "REG")
    last_week = pl.when(pl.col("season") < config.SEVENTEEN_GAME_SEASON_START).then(
        16
    ).otherwise(17)
    df = df.filter(pl.col("week") <= last_week)
    keep = [
        c
        for c in (
            "game_id",
            "season",
            "week",
            "gameday",
            "game_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
            "total_line",
            "result",
        )
        if c in df.columns
    ]
    return df.select(keep).with_columns(
        pl.col("home_team").map_elements(config.franchise, return_dtype=pl.String).alias(
            "home_team"
        ),
        pl.col("away_team").map_elements(config.franchise, return_dtype=pl.String).alias(
            "away_team"
        ),
    )


def _available_columns(path: Path) -> list[str]:
    schema = pl.scan_parquet(path).collect_schema()
    return [c for c in PBP_COLUMNS if c in schema.names()]


def load_pbp_season(season: int) -> pl.DataFrame:
    path = config.pbp_path(season)
    if not path.exists():
        raise FileNotFoundError(f"Missing PBP for {season}: {path}")
    cols = _available_columns(path)
    return pl.read_parquet(path, columns=cols)


def filter_scrimmage_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Keep designed rush/pass plays; drop kneels, spikes, and no-plays."""
    df = pbp.filter(pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null())
    if "play_type" in df.columns:
        df = df.filter(pl.col("play_type").is_in(["pass", "run"]))
    for flag, keep_if in (("qb_kneel", 0), ("qb_spike", 0), ("two_point_attempt", 0)):
        if flag in df.columns:
            df = df.filter(pl.col(flag).fill_null(0) == keep_if)
    if "epa" in df.columns:
        df = df.filter(pl.col("epa").is_not_null())
    return df


def _play_flags(plays: pl.DataFrame) -> pl.DataFrame:
    """Add helper columns used in the group-by aggregations."""
    rush_yards = (
        pl.col("rushing_yards") if "rushing_yards" in plays.columns else pl.col("yards_gained")
    )
    pass_yards = (
        pl.col("passing_yards") if "passing_yards" in plays.columns else pl.col("yards_gained")
    )
    is_run = pl.col("play_type").eq("run")
    is_pass = pl.col("play_type").eq("pass")
    explosive = (
        (is_run & (pl.col("yards_gained") >= config.EXPLOSIVE_RUSH_YARDS))
        | (is_pass & (pl.col("yards_gained") >= config.EXPLOSIVE_PASS_YARDS))
    ).cast(pl.Float64)
    return plays.with_columns(
        is_run.alias("is_run"),
        is_pass.alias("is_pass"),
        explosive.alias("is_explosive"),
        rush_yards.alias("rush_yards"),
        pass_yards.alias("dropback_yards"),
        pl.col("interception").fill_null(0).cast(pl.Float64).alias("is_int"),
        pl.col("fumble_lost").fill_null(0).cast(pl.Float64).alias("is_fumble_lost"),
    )


def _offense_agg(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, possession team): what that offense produced."""
    return plays.group_by(["game_id", "posteam"]).agg(
        pl.len().alias("off_plays"),
        pl.col("epa").mean().alias("off_epa_per_play"),
        pl.col("success").mean().alias("off_success_rate"),
        pl.when(pl.col("is_run").sum() > 0)
        .then(pl.col("rush_yards").filter(pl.col("is_run")).mean())
        .otherwise(None)
        .alias("off_yards_per_rush"),
        pl.when(pl.col("is_pass").sum() > 0)
        .then(pl.col("dropback_yards").filter(pl.col("is_pass")).mean())
        .otherwise(None)
        .alias("off_yards_per_dropback"),
        pl.col("is_explosive").mean().alias("off_explosive_play_rate"),
        pl.col("is_int").sum().alias("giveaway_int"),
        pl.col("is_fumble_lost").sum().alias("fumbles_lost"),
    )


def _defense_agg(plays: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, defending team): what that defense allowed."""
    return plays.group_by(["game_id", "defteam"]).agg(
        pl.len().alias("def_plays"),
        pl.col("epa").mean().alias("def_epa_per_play"),
        pl.col("success").mean().alias("def_success_rate"),
        pl.when(pl.col("is_run").sum() > 0)
        .then(pl.col("rush_yards").filter(pl.col("is_run")).mean())
        .otherwise(None)
        .alias("def_yards_per_rush"),
        pl.when(pl.col("is_pass").sum() > 0)
        .then(pl.col("dropback_yards").filter(pl.col("is_pass")).mean())
        .otherwise(None)
        .alias("def_yards_per_dropback"),
        pl.col("is_explosive").mean().alias("def_explosive_play_rate"),
        pl.col("is_int").sum().alias("takeaway_int"),
        pl.col("is_fumble_lost").sum().alias("takeaway_fumbles"),
    )


def _team_rows_from_schedule(games: pl.DataFrame) -> pl.DataFrame:
    """Stack home and away into a long team-game table."""
    game_date_col = "gameday" if "gameday" in games.columns else "game_date"
    home = games.select(
        "season",
        "week",
        "game_id",
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("home"),
        pl.col(game_date_col).alias("game_date") if game_date_col in games.columns else pl.lit(None).alias("game_date"),
        pl.col("home_score").alias("points_for") if "home_score" in games.columns else pl.lit(None).alias("points_for"),
        pl.col("away_score").alias("points_against")
        if "away_score" in games.columns
        else pl.lit(None).alias("points_against"),
        "spread_line",
        "total_line" if "total_line" in games.columns else pl.lit(None).alias("total_line"),
    )
    away = games.select(
        "season",
        "week",
        "game_id",
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("home"),
        pl.col(game_date_col).alias("game_date") if game_date_col in games.columns else pl.lit(None).alias("game_date"),
        pl.col("away_score").alias("points_for") if "away_score" in games.columns else pl.lit(None).alias("points_for"),
        pl.col("home_score").alias("points_against")
        if "home_score" in games.columns
        else pl.lit(None).alias("points_against"),
        "spread_line",
        "total_line" if "total_line" in games.columns else pl.lit(None).alias("total_line"),
    )
    return pl.concat([home, away], how="vertical").with_columns(
        pl.col("team").map_elements(config.franchise, return_dtype=pl.String),
        pl.col("opponent").map_elements(config.franchise, return_dtype=pl.String),
    )


def aggregate_season(pbp: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Build team-game stat lines for one season."""
    plays = _play_flags(filter_scrimmage_plays(pbp))
    offense = _offense_agg(plays).rename({"posteam": "team"})
    defense = _defense_agg(plays).rename({"defteam": "team"})
    teams = _team_rows_from_schedule(games)
    out = (
        teams.join(offense, on=["game_id", "team"], how="left")
        .join(defense, on=["game_id", "team"], how="left")
        .with_columns(
            (
                (pl.col("takeaway_int").fill_null(0) + pl.col("takeaway_fumbles").fill_null(0))
                - (pl.col("giveaway_int").fill_null(0) + pl.col("fumbles_lost").fill_null(0))
            ).alias("turnover_margin"),
        )
    )
    return out


def build_team_game_stats(seasons: list[int] | None = None) -> pl.DataFrame:
    seasons = seasons or list(config.SEASONS)
    schedules = usable_games(load_schedules())
    parts: list[pl.DataFrame] = []
    for season in seasons:
        path = config.pbp_path(season)
        if not path.exists():
            print(f"  skip {season}: no pbp file")
            continue
        print(f"  aggregating season {season} ...")
        pbp = load_pbp_season(season)
        games = schedules.filter(pl.col("season") == season)
        game_ids = set(games.get_column("game_id").to_list())
        pbp = pbp.filter(pl.col("game_id").is_in(list(game_ids)))
        parts.append(aggregate_season(pbp, games))
    if not parts:
        raise FileNotFoundError("No play-by-play seasons found. Run ingest first.")
    return pl.concat(parts, how="diagonal_relaxed").sort(["season", "week", "game_id", "home"])


def sanity_print(df: pl.DataFrame) -> None:
    """Phase 2 verify: known teams, league EPA, and 2×games row count."""
    print_header("Team-game stat lines")
    n_rows = df.height
    n_games = df.get_column("game_id").n_unique()
    print(f"rows={n_rows:,}  unique_games={n_games:,}  2×games={2 * n_games:,}")

    known = ["KC", "SF", "BUF", "PHI", "CHI"]
    have = set(df.get_column("team").unique().to_list())
    teams = [t for t in known if t in have]
    if not teams:
        teams = sorted(have)[:5]

    avg = (
        df.filter((pl.col("season") == 2024) & pl.col("team").is_in(teams))
        .group_by("team")
        .agg(
            pl.len().alias("games"),
            pl.col("off_epa_per_play").mean().round(3).alias("off_epa"),
            pl.col("def_epa_per_play").mean().round(3).alias("def_epa"),
            pl.col("off_success_rate").mean().round(3).alias("off_success"),
            pl.col("off_yards_per_rush").mean().round(2).alias("yp_rush"),
            pl.col("off_yards_per_dropback").mean().round(2).alias("yp_dropback"),
            pl.col("points_for").mean().round(1).alias("pts_for"),
            pl.col("points_against").mean().round(1).alias("pts_against"),
        )
        .sort("off_epa", descending=True)
    )
    print_frame(avg, "2024 season averages (usable weeks only) for well-known teams")

    league_epa = df.select(pl.col("off_epa_per_play").mean().round(4).alias("league_off_epa"))
    print_frame(league_epa, "League-average offensive EPA/play (should sit near 0)")
    print(
        "\nNote: averages here exclude the rest week and playoffs, so they will\n"
        "not match public 'full season including week 18' tables exactly."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build team-game stat lines.")
    parser.add_argument("--seasons", nargs="*", type=int, default=None)
    args = parser.parse_args(argv)
    config.ensure_data_dirs()
    df = build_team_game_stats(args.seasons)
    out = write_parquet(df, config.team_game_stats_raw_path())
    print(f"wrote {out} ({df.height:,} rows)")
    sanity_print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
