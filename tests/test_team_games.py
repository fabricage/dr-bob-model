"""Phase 2 tests: one row per team per game, home/away points mirror."""

from __future__ import annotations

import polars as pl

from features.team_games import build_team_game_stats, filter_scrimmage_plays


def test_kneels_spikes_no_plays_are_dropped() -> None:
    pbp = pl.DataFrame(
        {
            "posteam": ["KC", "KC", "KC", "KC"],
            "defteam": ["BUF", "BUF", "BUF", "BUF"],
            "play_type": ["pass", "qb_kneel", "qb_spike", "no_play"],
            "epa": [0.1, 0.0, 0.0, 0.5],
            "qb_kneel": [0, 1, 0, 0],
            "qb_spike": [0, 0, 1, 0],
            "two_point_attempt": [0, 0, 0, 0],
        }
    )
    kept = filter_scrimmage_plays(pbp)
    assert kept.height == 1
    assert kept["play_type"][0] == "pass"


def test_two_rows_per_game_and_points_mirror(mini_data) -> None:
    df = build_team_game_stats(seasons=[2024])
    counts = df.group_by("game_id").agg(pl.len().alias("n"))
    assert counts.filter(pl.col("n") != 2).height == 0
    assert df.height == 2 * df.get_column("game_id").n_unique()

    home = df.filter(pl.col("home")).select(
        "game_id",
        pl.col("points_for").alias("h_for"),
        pl.col("points_against").alias("h_against"),
    )
    away = df.filter(~pl.col("home")).select(
        "game_id",
        pl.col("points_for").alias("a_for"),
        pl.col("points_against").alias("a_against"),
    )
    joined = home.join(away, on="game_id")
    assert joined.filter(pl.col("h_for") != pl.col("a_against")).height == 0
    assert joined.filter(pl.col("h_against") != pl.col("a_for")).height == 0
    assert df["off_epa_per_play"].null_count() == 0
