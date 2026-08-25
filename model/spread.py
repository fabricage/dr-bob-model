"""Turn blended net EPA into a predicted margin and a model home spread.

    predicted_margin = (home_net_epa − away_net_epa) × points_per_epa + hfa
    model_home_spread = −predicted_margin
    home_spread_edge = market_home_spread − model_home_spread

points_per_epa and hfa are FIT, not guessed. We regress actual (home − away)
margins on the walk-forward net-EPA difference so a typical EPA gap maps
onto a typical point gap in historical data.

Sign conventions (betting shop, not PFR):
    market_home_spread = −nflverse_spread_line
    because nflverse/PFR stores a positive number when HOME is favored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl
from sklearn.linear_model import LinearRegression

import config
from helpers import print_frame, print_header, write_parquet
from model.overrides import apply_overrides, load_overrides
from model.weights import (
    expanding_game_ratings,
    load_compensated,
    load_prior_r,
    load_priors,
    load_weight_curve,
)


def to_market_home_spread(spread_line: float | None) -> float | None:
    """Convert nflverse/PFR spread_line into a betting-shop home spread."""
    if spread_line is None:
        return None
    return float(-spread_line)


def pair_home_away(ratings: pl.DataFrame) -> pl.DataFrame:
    """One row per game with home and away blended net EPA."""
    home = ratings.filter(pl.col("home")).select(
        "season",
        "week",
        "game_id",
        pl.col("team").alias("home_team"),
        pl.col("opponent").alias("away_team"),
        pl.col("net_epa_blended").alias("home_net_epa"),
        pl.col("games_played").alias("home_games_played"),
        pl.col("points_for").alias("home_score"),
        pl.col("points_against").alias("away_score"),
        pl.col(f"{config.PRIMARY_OFF_STAT}_weight").alias("home_weight_current"),
    )
    away = ratings.filter(~pl.col("home")).select(
        "game_id",
        pl.col("net_epa_blended").alias("away_net_epa"),
        pl.col("games_played").alias("away_games_played"),
        pl.col(f"{config.PRIMARY_OFF_STAT}_weight").alias("away_weight_current"),
    )
    paired = home.join(away, on="game_id")
    return paired.with_columns(
        (pl.col("home_net_epa") - pl.col("away_net_epa")).alias("net_epa_diff"),
        (pl.col("home_score") - pl.col("away_score")).alias("actual_margin"),
    )


def attach_market_lines(paired: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    lines = (
        games.filter(pl.col("home"))
        .select("game_id", "spread_line", "total_line", "game_date")
        .unique(subset=["game_id"])
    )
    out = paired.join(lines, on="game_id", how="left")
    return out.with_columns(
        (-pl.col("spread_line")).alias("market_home_spread"),
    )


def fit_conversion(train: pl.DataFrame) -> tuple[float, float]:
    """OLS: actual_margin = points_per_epa * net_epa_diff + hfa."""
    work = train.filter(
        pl.col("net_epa_diff").is_not_null() & pl.col("actual_margin").is_not_null()
    )
    x = work.get_column("net_epa_diff").to_numpy().reshape(-1, 1)
    y = work.get_column("actual_margin").to_numpy()
    if len(y) < 10:
        return 25.0, 2.0  # fallback only if the sample is empty (should not happen)
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    return float(model.coef_[0]), float(model.intercept_)


def apply_conversion(df: pl.DataFrame, points_per_epa: float, hfa: float) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("net_epa_diff") * points_per_epa + hfa).alias("predicted_margin"),
    ).with_columns(
        (-pl.col("predicted_margin")).alias("model_home_spread"),
    ).with_columns(
        (pl.col("market_home_spread") - pl.col("model_home_spread")).alias("home_spread_edge"),
    )


def walk_forward_games() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build every usable game's model line with no future data in the ratings.

    Conversion coefficients for season S are fit on seasons < S only.
    """
    games = load_compensated()
    priors = load_priors()
    curve = load_weight_curve()
    prior_r = load_prior_r()
    ratings = expanding_game_ratings(games, priors, curve, prior_r)
    paired = attach_market_lines(pair_home_away(ratings), games)

    coef_rows = []
    scored_parts = []
    seasons = sorted(paired.get_column("season").unique().to_list())
    for season in seasons:
        history = paired.filter(pl.col("season") < season)
        if history.height < 50:
            # First year or two: fit on whatever earlier games exist, else skip coef save.
            train = history if history.height >= 10 else paired.filter(pl.col("season") == season)
        else:
            train = history
        ppe, hfa = fit_conversion(train)
        coef_rows.append(
            {
                "season": season,
                "points_per_epa": ppe,
                "hfa": hfa,
                "train_games": train.height,
            }
        )
        part = apply_conversion(paired.filter(pl.col("season") == season), ppe, hfa)
        scored_parts.append(part)
    scored = pl.concat(scored_parts) if scored_parts else paired
    coefs = pl.DataFrame(coef_rows)
    return scored, coefs


def predict_game(
    game_id: str,
    apply_injury_overrides: bool = True,
    scored: pl.DataFrame | None = None,
) -> dict:
    """Look up one game's raw model line and optional override-adjusted line."""
    if scored is None:
        scored, _ = walk_forward_games()
    row = scored.filter(pl.col("game_id") == game_id)
    if row.height == 0:
        raise KeyError(f"No model line for game_id={game_id}")
    rec = row.to_dicts()[0]
    rec["adjusted_margin"] = rec["predicted_margin"]
    rec["adjusted_home_spread"] = rec["model_home_spread"]
    rec["active_overrides"] = []
    if apply_injury_overrides:
        adj = apply_overrides(
            rec["predicted_margin"],
            game_id=rec["game_id"],
            home_team=rec["home_team"],
            away_team=rec["away_team"],
            season=rec["season"],
            week=rec["week"],
            overrides=load_overrides(),
        )
        rec["adjusted_margin"] = adj["adjusted_margin"]
        rec["adjusted_home_spread"] = -adj["adjusted_margin"]
        rec["active_overrides"] = adj["applied"]
        rec["home_spread_edge_adjusted"] = rec["market_home_spread"] - rec["adjusted_home_spread"]
    return rec


def week_board(
    season: int,
    week: int,
    scored: pl.DataFrame | None = None,
    apply_injury_overrides: bool = True,
) -> pl.DataFrame:
    if scored is None:
        scored, _ = walk_forward_games()
    board = scored.filter((pl.col("season") == season) & (pl.col("week") == week))
    if apply_injury_overrides:
        rows = []
        for rec in board.to_dicts():
            adj = apply_overrides(
                rec["predicted_margin"],
                game_id=rec["game_id"],
                home_team=rec["home_team"],
                away_team=rec["away_team"],
                season=rec["season"],
                week=rec["week"],
                overrides=load_overrides(),
            )
            rec["adjusted_margin"] = adj["adjusted_margin"]
            rec["adjusted_home_spread"] = -adj["adjusted_margin"]
            rec["override_points"] = adj["points"]
            rec["n_overrides"] = len(adj["applied"])
            rec["override_reasons"] = "; ".join(
                f"{o.get('reason', '')} ({o.get('points', 0):+g})" for o in adj["applied"]
            )
            rec["home_spread_edge_adjusted"] = rec["market_home_spread"] - rec["adjusted_home_spread"]
            rec["abs_edge"] = abs(rec.get("home_spread_edge_adjusted") or rec["home_spread_edge"])
            rows.append(rec)
        board = pl.DataFrame(rows) if rows else board
    else:
        board = board.with_columns(pl.col("home_spread_edge").abs().alias("abs_edge"))
    if board.height:
        board = board.sort("abs_edge", descending=True)
    return board


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit spread conversion and print a sample board.")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--week", type=int, default=5)
    args = parser.parse_args(argv)
    config.ensure_data_dirs()
    print_header("Fitting points_per_epa and HFA (walk-forward, no look-ahead)")
    scored, coefs = walk_forward_games()
    write_parquet(scored, config.backtest_games_path())
    write_parquet(coefs, config.spread_coefficients_path())
    print_frame(coefs.with_columns(pl.col("points_per_epa").round(2), pl.col("hfa").round(2)))
    board = week_board(args.season, args.week, scored=scored)
    keep = [
        c
        for c in (
            "game_id",
            "home_team",
            "away_team",
            "market_home_spread",
            "model_home_spread",
            "adjusted_home_spread",
            "home_spread_edge",
            "predicted_margin",
            "actual_margin",
            "n_overrides",
        )
        if c in board.columns
    ]
    print_frame(
        board.select(keep).with_columns(
            pl.col(pl.Float64).round(2)
        ),
        f"Model board {args.season} week {args.week} (sorted by |edge| when available)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
