"""Phase 3 tests: ridge effects sit near mean-zero; HFA is finite."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

import config
from features.compensated import RIDGE_TARGETS, compensate_all
from features.team_games import build_team_game_stats
from helpers import write_parquet


def test_effects_mean_zero_and_hfa_finite(mini_data) -> None:
    raw = build_team_game_stats()
    write_parquet(raw, config.team_game_stats_raw_path())
    games, effects = compensate_all(raw, alpha=10.0)
    write_parquet(games, config.team_game_stats_path())
    write_parquet(effects, config.team_effects_path())

    for stat in RIDGE_TARGETS:
        chunk = effects.filter(pl.col("stat") == stat)
        off_mean = float(chunk["offense_effect"].mean())
        def_mean = float(chunk["defense_effect"].mean())
        assert abs(off_mean) < 1e-8, (stat, off_mean)
        assert abs(def_mean) < 1e-8, (stat, def_mean)
        for hfa in chunk["hfa"].to_list():
            assert math.isfinite(hfa)

    epa_hfa = float(
        effects.filter(pl.col("stat") == "off_epa_per_play")["hfa"].unique()[0]
    )
    # Home offense should have a small positive EPA bump in the planted data.
    assert math.isfinite(epa_hfa)
    assert epa_hfa > 0

    assert "off_epa_per_play_comp" in games.columns
    assert "def_epa_per_play_comp" in games.columns
    assert games.height == raw.height
    assert np.isfinite(games["off_epa_per_play_comp"].drop_nulls().to_numpy()).all()
