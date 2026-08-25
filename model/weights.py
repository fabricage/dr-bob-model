"""Blend weights: how fast we trust this year's numbers.

    weight(N) = r_current(N)² / (r_current(N)² + r_prior²)

r_current(N) comes from the first-N study (Phase 4). r_prior comes from
how well last year's shrunk effect predicted this year (Phase 5).

Weight is 0 before any games (pure prior) and climbs as N grows. We
interpolate between the studied N values and force the curve to be
monotone non-decreasing so a random dip at N=8 cannot make week 8
trust the data LESS than week 6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl

import config
from helpers import print_frame, print_header
from research.stability_firstn import _order_games


def load_weight_curve() -> pl.DataFrame:
    path = config.weight_curve_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python research/stability_firstn.py")
    return pl.read_parquet(path)


def load_prior_r() -> pl.DataFrame:
    path = config.prior_reliability_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python model/priors.py")
    return pl.read_parquet(path)


def load_priors() -> pl.DataFrame:
    path = config.priors_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python model/priors.py")
    return pl.read_parquet(path)


def load_compensated() -> pl.DataFrame:
    path = config.team_game_stats_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python features/compensated.py")
    return pl.read_parquet(path)


def _r_current_interpolator(curve: pl.DataFrame, stat: str) -> callable:
    """r(N) for one stat, with r(0)=0 and monotone non-decreasing r."""
    chunk = curve.filter(pl.col("stat") == stat).sort("n_games")
    ns = chunk.get_column("n_games").to_numpy().astype(float)
    rs = chunk.get_column("r_first_n").to_numpy().astype(float)
    rs = np.where(np.isfinite(rs), np.maximum(rs, 0.0), 0.0)
    # Prepend N=0.
    ns = np.concatenate([[0.0], ns])
    rs = np.concatenate([[0.0], rs])
    rs = np.maximum.accumulate(rs)

    def r_at(n: float) -> float:
        n = max(0.0, float(n))
        return float(np.interp(n, ns, rs, left=0.0, right=float(rs[-1])))

    return r_at


def blend_weight(games_played: float, stat: str, curve: pl.DataFrame, prior_r: pl.DataFrame) -> float:
    """Share of the blended rating that comes from THIS season (0..1)."""
    r_fn = _r_current_interpolator(curve, stat)
    rc = r_fn(games_played)
    match = prior_r.filter(pl.col("stat") == stat)
    rp = float(match.get_column("r_prior")[0]) if match.height else 0.0
    rp = max(0.0, rp)
    denom = rc * rc + rp * rp
    if denom <= 0:
        return 0.0 if games_played <= 0 else 1.0
    w = (rc * rc) / denom
    return float(min(1.0, max(0.0, w)))


def weight_curve_table(curve: pl.DataFrame, prior_r: pl.DataFrame, stat: str = "off_epa_per_play") -> pl.DataFrame:
    """Week-by-week (N=0..17) blend weights for one stat, for the verify print."""
    rows = []
    for n in range(0, 18):
        rows.append(
            {
                "games_played": n,
                "stat": stat,
                "weight_current": blend_weight(n, stat, curve, prior_r),
                "weight_prior": 1.0 - blend_weight(n, stat, curve, prior_r),
            }
        )
    return pl.DataFrame(rows)


def blended_rating(
    team: str,
    stat: str,
    season: int,
    as_of_week: int,
    games: pl.DataFrame | None = None,
    priors: pl.DataFrame | None = None,
    curve: pl.DataFrame | None = None,
    prior_r: pl.DataFrame | None = None,
) -> dict:
    """Blend prior + current-season games strictly BEFORE `as_of_week`.

    Returns a small dict so callers (and tests) can see the ingredients,
    not just the final number. Never uses same-week or future games.
    """
    games = games if games is not None else load_compensated()
    priors = priors if priors is not None else load_priors()
    curve = curve if curve is not None else load_weight_curve()
    prior_r = prior_r if prior_r is not None else load_prior_r()

    col = f"{stat}_comp" if f"{stat}_comp" in games.columns else stat
    current = games.filter(
        (pl.col("season") == season)
        & (pl.col("team") == team)
        & (pl.col("week") < as_of_week)
        & pl.col(col).is_not_null()
    )
    n = current.height
    current_mean = float(current.get_column(col).mean()) if n else 0.0
    prior_row = priors.filter(
        (pl.col("for_season") == season) & (pl.col("team") == team) & (pl.col("stat") == stat)
    )
    prior_val = float(prior_row.get_column("prior")[0]) if prior_row.height else 0.0
    w = blend_weight(n, stat, curve, prior_r)
    blended = w * current_mean + (1.0 - w) * prior_val
    return {
        "team": team,
        "stat": stat,
        "season": season,
        "as_of_week": as_of_week,
        "games_played": n,
        "current_mean": current_mean,
        "prior": prior_val,
        "weight_current": w,
        "blended": blended,
        "max_week_used": int(current.get_column("week").max()) if n else None,
    }


def ratings_as_of(
    season: int,
    as_of_week: int,
    games: pl.DataFrame | None = None,
    priors: pl.DataFrame | None = None,
    curve: pl.DataFrame | None = None,
    prior_r: pl.DataFrame | None = None,
    stats: list[str] | None = None,
) -> pl.DataFrame:
    """Blended ratings for every team as of the start of `as_of_week`."""
    games = games if games is not None else load_compensated()
    priors = priors if priors is not None else load_priors()
    curve = curve if curve is not None else load_weight_curve()
    prior_r = prior_r if prior_r is not None else load_prior_r()
    stats = stats or config.all_model_stat_columns()

    prior_s = priors.filter(pl.col("for_season") == season)
    teams = sorted(
        set(games.filter(pl.col("season") == season).get_column("team").to_list())
        | set(prior_s.get_column("team").to_list())
    )
    rows = []
    for team in teams:
        rec = {"season": season, "as_of_week": as_of_week, "team": team}
        for stat in stats:
            d = blended_rating(team, stat, season, as_of_week, games, priors, curve, prior_r)
            rec[stat] = d["blended"]
            if stat == config.PRIMARY_OFF_STAT:
                rec["games_played"] = d["games_played"]
                rec["weight_current"] = d["weight_current"]
        rec["net_epa"] = rec.get(config.PRIMARY_OFF_STAT, 0.0) - rec.get(
            config.PRIMARY_DEF_STAT, 0.0
        )
        rows.append(rec)
    return pl.DataFrame(rows)


def expanding_game_ratings(
    games: pl.DataFrame,
    priors: pl.DataFrame,
    curve: pl.DataFrame,
    prior_r: pl.DataFrame,
    stats: list[str] | None = None,
) -> pl.DataFrame:
    """For every team-game, ratings using ONLY earlier games that season.

    This is the walk-forward feature table. Same-week and future games are
    never in the average: we use the expanding mean *before* the current
    game number.
    """
    stats = stats or [config.PRIMARY_OFF_STAT, config.PRIMARY_DEF_STAT]
    numbered = _order_games(games)
    current_exprs = []
    for stat in stats:
        col = f"{stat}_comp"
        csum = pl.col(col).cum_sum().over(["season", "team"])
        prior_sum = csum - pl.col(col)
        prior_n = pl.col("game_number") - 1
        current_exprs.append(
            pl.when(prior_n > 0).then(prior_sum / prior_n).otherwise(None).alias(f"{stat}_current")
        )
    base = numbered.with_columns(current_exprs)

    out_rows = []
    # Small loop over team-games (~5k) is fine and keeps the blend formula obvious.
    prior_lookup = {
        (r["for_season"], r["team"], r["stat"]): r["prior"]
        for r in priors.iter_rows(named=True)
    }
    for row in base.iter_rows(named=True):
        rec = {
            "season": row["season"],
            "week": row["week"],
            "game_id": row["game_id"],
            "team": row["team"],
            "opponent": row["opponent"],
            "home": row["home"],
            "game_number": row["game_number"],
            "games_played": int(row["game_number"] - 1),
            "points_for": row.get("points_for"),
            "points_against": row.get("points_against"),
        }
        for stat in stats:
            n = rec["games_played"]
            current = row.get(f"{stat}_current")
            if current is None:
                current = 0.0
            prior = prior_lookup.get((row["season"], row["team"], stat), 0.0) or 0.0
            w = blend_weight(n, stat, curve, prior_r)
            rec[f"{stat}_blended"] = w * float(current) + (1.0 - w) * float(prior)
            rec[f"{stat}_current"] = float(current)
            rec[f"{stat}_prior"] = float(prior)
            rec[f"{stat}_weight"] = w
        rec["net_epa_blended"] = rec[f"{config.PRIMARY_OFF_STAT}_blended"] - rec[
            f"{config.PRIMARY_DEF_STAT}_blended"
        ]
        out_rows.append(rec)
    return pl.DataFrame(out_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print blend-weight curve.")
    parser.parse_args(argv)
    curve = load_weight_curve()
    prior_r = load_prior_r()
    table = weight_curve_table(curve, prior_r, "off_epa_per_play")
    print_header("Blend weight curve for off_epa_per_play")
    print_frame(
        table.with_columns(
            pl.col("weight_current").round(3),
            pl.col("weight_prior").round(3),
        )
    )
    print(
        "\nThe current-season weight should start near 0 (or low) and climb\n"
        "toward roughly 0.7–0.9 by midseason. Exact values depend on how\n"
        "stable EPA was in YOUR data vs how sticky the prior was."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
