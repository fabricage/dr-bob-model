"""Opponent-adjust ("compensate") team-game stats with ridge regression.

Raw EPA against a cupcake defense looks great; raw EPA against a shutdown
front looks awful. We do not want the model to reward a soft schedule.

For each season and each candidate stat we fit:

    observed_stat ≈ offense_effect[team]
                  + defense_effect[opponent]
                  + hfa * is_home

Ridge (alpha=50 by default) keeps the 32 + 32 team effects from flying
off to infinity. After fitting we subtract the mean from the offense
effects and from the defense effects so they sit near zero — "average"
is literally average.

Compensated per-game value = observed minus opponent effect minus HFA.
That is "how did this team play against a league-average opponent on a
neutral field," plus leftover noise.
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
from sklearn.linear_model import Ridge

import config
from helpers import print_frame, print_header, write_parquet

# Ridge is fit on these observed columns. For efficiency stats we use the
# OFFENSE version (what the team produced). Defense-allowed columns are
# then neutralized with the same opponent/venue effects.
RIDGE_TARGETS: list[str] = [f"off_{s}" for s in config.EFFICIENCY_STATS] + list(
    config.NOISE_CONTROL_STATS
)


def load_raw_team_games() -> pl.DataFrame:
    path = config.team_game_stats_raw_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python features/team_games.py")
    return pl.read_parquet(path)


def _hfa_code(is_home: bool) -> float:
    """+0.5 home / −0.5 away so HFA cannot swallow the grand mean of the stat."""
    return 0.5 if bool(is_home) else -0.5


def _design_matrix(
    teams: list[str],
    team: np.ndarray,
    opponent: np.ndarray,
    home: np.ndarray,
) -> np.ndarray:
    """Dummy-code offense team + defense opponent + home-field contrast.

    Columns: [off_team_1 ... off_team_k | def_opp_1 ... def_opp_k | hfa]
    Ridge does not need us to drop a dummy; shrinkage pins the scale.
    HFA is contrast-coded so it is the home-minus-away gap, not the
    league average of the stat (success rate is ~0.43; EPA is ~0).
    """
    index = {t: i for i, t in enumerate(teams)}
    n = len(team)
    k = len(teams)
    x = np.zeros((n, 2 * k + 1), dtype=float)
    for i in range(n):
        x[i, index[str(team[i])]] = 1.0
        x[i, k + index[str(opponent[i])]] = 1.0
        x[i, -1] = _hfa_code(home[i])
    return x


def fit_stat_season(
    season_df: pl.DataFrame,
    stat: str,
    alpha: float,
) -> tuple[dict[str, float], dict[str, float], float, np.ndarray]:
    """Fit one season of one stat. Returns off effects, def effects, hfa, residual."""
    work = season_df.filter(pl.col(stat).is_not_null())
    teams = sorted(set(work.get_column("team").to_list()) | set(work.get_column("opponent").to_list()))
    y = work.get_column(stat).to_numpy().astype(float)
    x = _design_matrix(
        teams,
        work.get_column("team").to_numpy(),
        work.get_column("opponent").to_numpy(),
        work.get_column("home").to_numpy(),
    )
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(x, y)
    coef = model.coef_
    k = len(teams)
    off = coef[:k].copy()
    deff = coef[k : 2 * k].copy()
    hfa = float(coef[-1])
    # Center so "0" means league-average unit, not an arbitrary dummy.
    off -= off.mean()
    deff -= deff.mean()
    off_map = {t: float(off[i]) for i, t in enumerate(teams)}
    def_map = {t: float(deff[i]) for i, t in enumerate(teams)}
    # Compensated offense ≈ offense_effect + residual (opponent/HFA stripped).
    compensated = y - np.array(
        [
            def_map[str(o)] + hfa * _hfa_code(h)
            for o, h in zip(
                work.get_column("opponent").to_list(),
                work.get_column("home").to_list(),
                strict=True,
            )
        ]
    )
    return (
        off_map,
        def_map,
        hfa,
        compensated,
        work.get_column("game_id").to_list(),
        work.get_column("team").to_list(),
    )


def _defense_compensated(
    season_df: pl.DataFrame,
    def_stat: str,
    off_map: dict[str, float],
    hfa: float,
) -> pl.DataFrame:
    """Strip the opponent's offense and the HFA that applied to that opponent."""
    def_comp = []
    game_ids = []
    teams = []
    for row in season_df.select(["game_id", "team", "opponent", "home", def_stat]).iter_rows(
        named=True
    ):
        y_i = row[def_stat]
        game_ids.append(row["game_id"])
        teams.append(row["team"])
        if y_i is None or (isinstance(y_i, float) and not np.isfinite(y_i)):
            def_comp.append(None)
            continue
        opp_home = not bool(row["home"])
        def_comp.append(float(y_i) - off_map.get(str(row["opponent"]), 0.0) - hfa * _hfa_code(opp_home))
    return pl.DataFrame(
        {"game_id": game_ids, "team": teams, f"{def_stat}_comp": def_comp}
    )


def compensate_all(raw: pl.DataFrame, alpha: float = config.RIDGE_ALPHA) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (per-game compensated stats, season-level team effects)."""
    effect_rows: list[dict] = []
    season_frames: list[pl.DataFrame] = []
    for season in sorted(raw.get_column("season").unique().to_list()):
        season_df = raw.filter(pl.col("season") == season)
        print(f"  ridge season {season} ({season_df.height} team-games) ...")
        for stat in RIDGE_TARGETS:
            off_map, def_map, hfa, compensated, game_ids, teams = fit_stat_season(
                season_df, stat, alpha
            )
            piece = pl.DataFrame(
                {"game_id": game_ids, "team": teams, f"{stat}_comp": compensated}
            )
            season_df = season_df.join(piece, on=["game_id", "team"], how="left")
            for team in sorted(set(off_map) | set(def_map)):
                effect_rows.append(
                    {
                        "season": season,
                        "team": team,
                        "stat": stat,
                        "offense_effect": off_map.get(team, 0.0),
                        "defense_effect": def_map.get(team, 0.0),
                        "hfa": hfa,
                        "alpha": alpha,
                    }
                )
            if stat.startswith("off_"):
                def_stat = "def_" + stat[len("off_") :]
                if def_stat in season_df.columns:
                    season_df = season_df.join(
                        _defense_compensated(season_df, def_stat, off_map, hfa),
                        on=["game_id", "team"],
                        how="left",
                    )
        season_frames.append(season_df)

    effects = pl.DataFrame(effect_rows)
    games = pl.concat(season_frames, how="diagonal_relaxed") if season_frames else raw
    return games, effects


def net_epa_rankings(effects: pl.DataFrame, season: int) -> pl.DataFrame:
    epa = effects.filter((pl.col("season") == season) & (pl.col("stat") == "off_epa_per_play"))
    return epa.with_columns(
        (pl.col("offense_effect") - pl.col("defense_effect")).alias("net_epa")
    ).sort("net_epa", descending=True)


def biggest_movers(raw: pl.DataFrame, effects: pl.DataFrame, season: int, n: int = 8) -> pl.DataFrame:
    """Teams whose ranking changes most after opponent adjustment."""
    raw_net = (
        raw.filter(pl.col("season") == season)
        .group_by("team")
        .agg(
            (pl.col("off_epa_per_play").mean() - pl.col("def_epa_per_play").mean()).alias(
                "raw_net_epa"
            )
        )
        .with_columns(pl.col("raw_net_epa").rank(descending=True).alias("raw_rank"))
    )
    adj = net_epa_rankings(effects, season).select(
        "team",
        pl.col("net_epa").alias("adj_net_epa"),
        pl.col("net_epa").rank(descending=True).alias("adj_rank"),
    )
    moved = raw_net.join(adj, on="team").with_columns(
        (pl.col("raw_rank") - pl.col("adj_rank")).alias("rank_change_raw_minus_adj")
    )
    # Positive rank_change: looked better on raw (soft schedule) than adjusted.
    return moved.sort(pl.col("rank_change_raw_minus_adj").abs(), descending=True).head(n)


def print_verify(raw: pl.DataFrame, effects: pl.DataFrame) -> None:
    season = 2024 if 2024 in set(effects.get_column("season").to_list()) else int(
        effects.get_column("season").max()
    )
    ranked = net_epa_rankings(effects, season).select(
        "team",
        pl.col("offense_effect").round(4).alias("off_epa"),
        pl.col("defense_effect").round(4).alias("def_epa"),
        pl.col("net_epa").round(4),
        pl.col("hfa").round(4).alias("hfa_epa"),
    )
    print_frame(ranked.head(10), f"{season} net EPA effect — top 10 (good teams should be here)")
    print_frame(ranked.tail(10).sort("net_epa"), f"{season} net EPA effect — bottom 10")
    print_frame(
        biggest_movers(raw, effects, season).with_columns(
            pl.col("raw_net_epa").round(4),
            pl.col("adj_net_epa").round(4),
            pl.col("raw_rank").round(0),
            pl.col("adj_rank").round(0),
        ),
        f"{season} biggest movers raw vs opponent-adjusted (soft-schedule check)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opponent-adjust team-game stats.")
    parser.add_argument("--alpha", type=float, default=config.RIDGE_ALPHA)
    args = parser.parse_args(argv)
    config.ensure_data_dirs()
    raw = load_raw_team_games()
    print_header(f"Compensated stats (ridge alpha={args.alpha})")
    games, effects = compensate_all(raw, alpha=args.alpha)
    write_parquet(games, config.team_game_stats_path())
    write_parquet(effects, config.team_effects_path())
    print(f"wrote {config.team_game_stats_path()} ({games.height:,} rows)")
    print(f"wrote {config.team_effects_path()} ({effects.height:,} rows)")
    print_verify(raw, effects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
