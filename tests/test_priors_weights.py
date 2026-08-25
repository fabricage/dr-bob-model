"""Phase 5 tests: blend weights are in [0, 1] and non-decreasing in N."""

from __future__ import annotations

import polars as pl

from model.weights import blend_weight, weight_curve_table


def test_weights_monotone_and_bounded() -> None:
    curve = pl.DataFrame(
        {
            "stat": ["off_epa_per_play"] * 5,
            "n_games": [3, 4, 6, 8, 12],
            "r_first_n": [0.25, 0.32, 0.40, 0.38, 0.55],  # dip at 8 must not lower weight
        }
    )
    prior_r = pl.DataFrame({"stat": ["off_epa_per_play"], "r_prior": [0.30]})
    table = weight_curve_table(curve, prior_r, "off_epa_per_play")
    weights = table["weight_current"].to_list()
    assert all(0.0 <= w <= 1.0 for w in weights)
    assert weights == sorted(weights)
    assert weights[0] == 0.0
    # Midseason (N=8) should already lean on current data.
    w8 = blend_weight(8, "off_epa_per_play", curve, prior_r)
    assert 0.5 <= w8 <= 1.0
