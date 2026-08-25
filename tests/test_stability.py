"""Phase 4 tests: efficiency stats should out-reliability the noise controls."""

from __future__ import annotations

import polars as pl

import config
from features.compensated import compensate_all
from features.team_games import build_team_game_stats
from helpers import write_parquet
from research.stability_firstn import firstn_grid
from research.stability_oddeven import split_half_table


def _prepare(mini_data) -> pl.DataFrame:
    raw = build_team_game_stats()
    write_parquet(raw, config.team_game_stats_raw_path())
    games, effects = compensate_all(raw, alpha=10.0)
    write_parquet(games, config.team_game_stats_path())
    write_parquet(effects, config.team_effects_path())
    return games


def test_epa_more_reliable_than_fumbles(mini_data) -> None:
    games = _prepare(mini_data)
    table = split_half_table(games)
    write_parquet(table, config.stability_oddeven_path())
    by_stat = {r["stat"]: r["r_spearman_brown"] for r in table.to_dicts()}
    epa = by_stat["off_epa_per_play"]
    success = by_stat["off_success_rate"]
    fumbles = by_stat["fumbles_lost"]
    turnovers = by_stat["turnover_margin"]
    assert epa > fumbles
    assert success > turnovers
    assert epa > 0.15


def test_firstn_r_generally_rises_for_epa(mini_data) -> None:
    games = _prepare(mini_data)
    grid = firstn_grid(games, ns=[3, 4, 5])
    write_parquet(grid, config.weight_curve_path())
    epa = grid.filter(pl.col("stat") == "off_epa_per_play").sort("n_games")
    rs = [r for r in epa["r_first_n"].to_list() if r is not None]
    assert len(rs) >= 2
    # Loose: last r should not be dramatically worse than first (planted skill).
    assert rs[-1] >= rs[0] - 0.25
    fumbles = [x or 0.0 for x in grid.filter(pl.col("stat") == "fumbles_lost")["r_first_n"].to_list()]
    assert max(rs) > max(fumbles)
