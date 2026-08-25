"""Phase 6 tests: no look-ahead in as-of ratings; conversion is finite."""

from __future__ import annotations

import math

import config
from features.compensated import compensate_all
from features.team_games import build_team_game_stats
from helpers import write_parquet
from model.priors import build_priors
from model.spread import walk_forward_games
from model.weights import blended_rating
from research.stability_firstn import firstn_grid
from research.stability_oddeven import split_half_table


def _full_mini_pipeline(mini_data) -> None:
    raw = build_team_game_stats()
    write_parquet(raw, config.team_game_stats_raw_path())
    games, effects = compensate_all(raw, alpha=10.0)
    write_parquet(games, config.team_game_stats_path())
    write_parquet(effects, config.team_effects_path())
    odd = split_half_table(games)
    write_parquet(odd, config.stability_oddeven_path())
    grid = firstn_grid(games)
    write_parquet(grid, config.weight_curve_path())
    priors, prior_r = build_priors(effects, odd)
    write_parquet(priors, config.priors_path())
    write_parquet(prior_r, config.prior_reliability_path())


def test_blended_rating_excludes_same_week_and_future(mini_data) -> None:
    _full_mini_pipeline(mini_data)
    info = blended_rating("KC", "off_epa_per_play", 2024, 5)
    assert info["games_played"] >= 1
    assert info["max_week_used"] is not None
    assert info["max_week_used"] < 5


def test_walk_forward_produces_spreads(mini_data) -> None:
    _full_mini_pipeline(mini_data)
    scored, coefs = walk_forward_games()
    assert scored.height > 0
    assert "predicted_margin" in scored.columns
    assert "model_home_spread" in scored.columns
    assert "home_spread_edge" in scored.columns
    for row in coefs.to_dicts():
        assert math.isfinite(row["points_per_epa"])
        assert math.isfinite(row["hfa"])
