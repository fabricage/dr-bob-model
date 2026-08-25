"""Preseason priors: last year, shrunk toward the league mean.

A prior is not "I think this team is good." It is last season's opponent-
adjusted effect, pulled toward zero by how UNreliable that stat was in the
odd/even study:

    prior = split_half_reliability * last_season_effect
            + (1 - reliability) * league_mean

Because effects are already mean-zero, league mean is 0, so this is simply
reliability × last_year. We still write it out the long way so the formula
stays obvious.

We then measure how well THAT prior predicts the NEXT season's actual
effect. That predictive r becomes r_prior in the blend-weight formula.
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
from helpers import pearson_r, print_frame, print_header, write_parquet


def load_effects() -> pl.DataFrame:
    path = config.team_effects_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python features/compensated.py")
    return pl.read_parquet(path)


def load_split_half() -> pl.DataFrame:
    path = config.stability_oddeven_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python research/stability_oddeven.py")
    return pl.read_parquet(path)


def _reliability_map(split_half: pl.DataFrame) -> dict[str, float]:
    """stat -> Spearman-Brown reliability, floored at 0 (never reverse a prior)."""
    out = {}
    for row in split_half.iter_rows(named=True):
        r = row["r_spearman_brown"]
        if r is None or r != r:  # NaN
            r = 0.0
        out[row["stat"]] = float(max(0.0, min(1.0, r)))
    return out


def _season_stat_as_ratings(effects: pl.DataFrame) -> pl.DataFrame:
    """One row per season-team-stat with the effect we will shrink.

    Efficiency: offense_effect of off_* is the off rating; defense_effect of
    the matching off_* row is the def rating. Noise stats use offense_effect
    as a single team effect.
    """
    rows = []
    for row in effects.iter_rows(named=True):
        stat = row["stat"]
        if stat.startswith("off_"):
            rows.append(
                {
                    "season": row["season"],
                    "team": row["team"],
                    "stat": stat,
                    "effect": row["offense_effect"],
                }
            )
            rows.append(
                {
                    "season": row["season"],
                    "team": row["team"],
                    "stat": "def_" + stat[len("off_") :],
                    "effect": row["defense_effect"],
                }
            )
        else:
            rows.append(
                {
                    "season": row["season"],
                    "team": row["team"],
                    "stat": stat,
                    "effect": row["offense_effect"],
                }
            )
    return pl.DataFrame(rows)


def build_priors(effects: pl.DataFrame, split_half: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (prior rows keyed by the season they apply to, r_prior by stat)."""
    ratings = _season_stat_as_ratings(effects)
    rel = _reliability_map(split_half)
    # Default reliability if a stat was missing from the study.
    for stat in config.all_model_stat_columns():
        rel.setdefault(stat, 0.0)

    prior_frames = []
    seasons = sorted(ratings.get_column("season").unique().to_list())
    for season in seasons:
        nxt = season + 1
        last = ratings.filter(pl.col("season") == season).with_columns(
            pl.col("stat")
            .map_elements(lambda s: rel.get(s, 0.0), return_dtype=pl.Float64)
            .alias("reliability")
        )
        last = last.with_columns(
            (pl.col("reliability") * pl.col("effect") + (1 - pl.col("reliability")) * 0.0).alias(
                "prior"
            ),
            pl.lit(nxt).alias("for_season"),
            pl.lit(season).alias("from_season"),
        )
        prior_frames.append(last.select(["for_season", "from_season", "team", "stat", "effect", "reliability", "prior"]))
    priors = pl.concat(prior_frames)

    # Predictive r: prior for season S vs actual effect in season S.
    actual = ratings.rename({"season": "for_season", "effect": "actual"})
    paired = priors.join(actual, on=["for_season", "team", "stat"], how="inner")
    rel_rows = []
    for stat in sorted(paired.get_column("stat").unique().to_list()):
        chunk = paired.filter(pl.col("stat") == stat)
        r = pearson_r(chunk.get_column("prior").to_numpy(), chunk.get_column("actual").to_numpy())
        rel_rows.append(
            {
                "stat": stat,
                "r_prior": 0.0 if r != r else float(max(0.0, r)),
                "n_team_seasons": chunk.height,
            }
        )
    prior_r = pl.DataFrame(rel_rows).sort("r_prior", descending=True)
    return priors, prior_r


def print_verify(priors: pl.DataFrame, prior_r: pl.DataFrame) -> None:
    print_frame(
        prior_r.with_columns(pl.col("r_prior").round(3)),
        "How well does last year's shrunk effect predict this year?",
    )
    season = 2025
    have = set(priors.get_column("for_season").unique().to_list())
    if season not in have:
        season = max(have)
    net = (
        priors.filter((pl.col("for_season") == season) & (pl.col("stat") == "off_epa_per_play"))
        .join(
            priors.filter(
                (pl.col("for_season") == season) & (pl.col("stat") == "def_epa_per_play")
            ).select("team", pl.col("prior").alias("def_prior")),
            on="team",
        )
        .with_columns((pl.col("prior") - pl.col("def_prior")).alias("net_epa_prior"))
        .select(
            "team",
            pl.col("prior").round(4).alias("off_epa_prior"),
            pl.col("def_prior").round(4).alias("def_epa_prior"),
            pl.col("net_epa_prior").round(4),
        )
        .sort("net_epa_prior", descending=True)
    )
    print_frame(net.head(10), f"{season} preseason top 10 by prior net EPA")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build shrunk preseason priors.")
    parser.parse_args(argv)
    config.ensure_data_dirs()
    priors, prior_r = build_priors(load_effects(), load_split_half())
    write_parquet(priors, config.priors_path())
    write_parquet(prior_r, config.prior_reliability_path())
    print_header("Preseason priors")
    print(f"wrote {config.priors_path()} ({priors.height:,} rows)")
    print_verify(priors, prior_r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
