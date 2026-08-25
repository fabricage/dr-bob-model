"""Walk-forward backtest of model spreads vs closing market lines.

For each season 2019–2025 and each week 3+:
  * ratings use priors from previous seasons + current-season games
    already played (never the same week, never the future)
  * conversion (points per EPA, HFA) is fit on earlier seasons only
  * we compare predicted margin to actual margin, and model spread to
    the schedule's spread_line

ATS (against the spread) is reported only when the model disagrees with
the market by at least 1.5 / 2.5 / 3.5 points. This is a research
metric, not a betting ticket.
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
from model.spread import walk_forward_games
from model.weights import (
    blended_rating,
    load_compensated,
    load_prior_r,
    load_priors,
    load_weight_curve,
)


def ats_side(edge: float) -> str | None:
    """If edge > 0 the model likes the home team vs the market line."""
    if edge is None:
        return None
    if edge > 0:
        return "home"
    if edge < 0:
        return "away"
    return None


def grade_ats(actual_margin: float, market_home_spread: float, side: str) -> str:
    """Home covers when actual_margin + market_home_spread > 0."""
    margin_vs_line = actual_margin + market_home_spread
    if abs(margin_vs_line) < 1e-9:
        return "push"
    home_covers = margin_vs_line > 0
    won = home_covers if side == "home" else (not home_covers)
    return "win" if won else "loss"


def summarize(games: pl.DataFrame, label: str) -> dict:
    work = games.filter(
        pl.col("predicted_margin").is_not_null() & pl.col("actual_margin").is_not_null()
    )
    mae_margin = (work.get_column("predicted_margin") - work.get_column("actual_margin")).abs().mean()
    mae_line = (work.get_column("model_home_spread") - work.get_column("market_home_spread")).abs().mean()
    rec = {
        "label": label,
        "n_games": work.height,
        "mae_margin": float(mae_margin) if mae_margin is not None else None,
        "mae_vs_market": float(mae_line) if mae_line is not None else None,
    }
    for thresh in config.EDGE_THRESHOLDS:
        picks = []
        for row in work.iter_rows(named=True):
            edge = row["home_spread_edge"]
            if edge is None or abs(edge) < thresh:
                continue
            if row["actual_margin"] is None or row["market_home_spread"] is None:
                continue
            side = ats_side(edge)
            result = grade_ats(row["actual_margin"], row["market_home_spread"], side)
            if result != "push":
                picks.append(result)
        wins = sum(1 for r in picks if r == "win")
        n = len(picks)
        rec[f"ats_n_{thresh}"] = n
        rec[f"ats_win_{thresh}"] = wins
        rec[f"ats_pct_{thresh}"] = (wins / n) if n else None
    return rec


def run_backtest() -> tuple[pl.DataFrame, pl.DataFrame]:
    scored, _coefs = walk_forward_games()
    scored = scored.filter(
        pl.col("season").is_in(config.BACKTEST_SEASONS) & (pl.col("week") >= config.BACKTEST_MIN_WEEK)
    )
    rows = []
    for season in config.BACKTEST_SEASONS:
        chunk = scored.filter(pl.col("season") == season)
        if chunk.height:
            rows.append(summarize(chunk, str(season)))
    rows.append(summarize(scored, "pooled"))
    report = pl.DataFrame(rows)
    return scored, report


def print_report(report: pl.DataFrame) -> None:
    cols = ["label", "n_games", "mae_margin", "mae_vs_market"]
    for thresh in config.EDGE_THRESHOLDS:
        cols += [f"ats_n_{thresh}", f"ats_win_{thresh}", f"ats_pct_{thresh}"]
    pretty = report.select(cols).with_columns(
        pl.col("mae_margin", "mae_vs_market", *[f"ats_pct_{t}" for t in config.EDGE_THRESHOLDS]).round(3)
    )
    print_frame(pretty, "Walk-forward backtest (week 3+, usable regular season)")
    print(
        "\nATS columns are research diagnostics. This program does not place bets.\n"
        "mae_margin = mean |predicted home margin − actual home margin|.\n"
        "mae_vs_market = mean |model home spread − market home spread|."
    )


def lookahead_spotcheck(season: int = 2024, week: int = 5, team: str = "KC") -> dict:
    """Prove that ratings as of week W only use games with week < W."""
    games = load_compensated()
    info = blended_rating(
        team,
        config.PRIMARY_OFF_STAT,
        season,
        week,
        games=games,
        priors=load_priors(),
        curve=load_weight_curve(),
        prior_r=load_prior_r(),
    )
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward spread backtest.")
    parser.parse_args(argv)
    config.ensure_data_dirs()
    print_header("Walk-forward backtest")
    scored, report = run_backtest()
    write_parquet(scored, config.backtest_games_path())
    write_parquet(report, config.backtest_path())
    print_report(report)
    check = lookahead_spotcheck()
    print_header("Look-ahead spot-check")
    print(
        f"team={check['team']} season={check['season']} as_of_week={check['as_of_week']}\n"
        f"games_played={check['games_played']} max_week_used={check['max_week_used']}\n"
        f"(max_week_used must be < as_of_week; None means prior only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
