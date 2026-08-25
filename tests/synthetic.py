"""Build a tiny but realistic NFL-like dataset for tests.

Four teams, several seasons, eight weeks each. Offensive EPA is planted
with real team skill so stability tests can see the difference between
efficiency (signal) and fumbles (noise).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from helpers import write_parquet

TEAMS = ["KC", "BUF", "SF", "PHI"]

# True talent: (off_epa, def_epa_allowed). Lower def = better defense.
TALENT = {
    "KC": (0.16, -0.08),
    "BUF": (0.10, -0.02),
    "SF": (0.12, -0.10),
    "PHI": (0.04, 0.03),
}

# Home/away pairs for 8 weeks (each team once per week).
PAIRINGS = [
    [("KC", "BUF"), ("SF", "PHI")],
    [("KC", "SF"), ("BUF", "PHI")],
    [("KC", "PHI"), ("BUF", "SF")],
    [("BUF", "KC"), ("PHI", "SF")],
    [("SF", "KC"), ("PHI", "BUF")],
    [("PHI", "KC"), ("SF", "BUF")],
    [("KC", "BUF"), ("PHI", "SF")],
    [("SF", "BUF"), ("PHI", "KC")],
]


def _plays_for_game(
    rng: np.random.Generator,
    game_id: str,
    season: int,
    week: int,
    home: str,
    away: str,
) -> list[dict]:
    rows = []
    for posteam, defteam, is_home in ((home, away, True), (away, home, False)):
        off, _ = TALENT[posteam]
        _, opp_def = TALENT[defteam]
        mean_epa = off + opp_def + (0.04 if is_home else 0.0)
        for i in range(30):
            is_pass = i % 2 == 0
            epa = float(rng.normal(mean_epa, 0.35))
            yards = float(max(-10.0, rng.normal(7.0 if is_pass else 4.2, 6.0)))
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "game_id": game_id,
                    "game_date": f"{season}-09-{week:02d}",
                    "home_team": home,
                    "away_team": away,
                    "posteam": posteam,
                    "defteam": defteam,
                    "play_type": "pass" if is_pass else "run",
                    "epa": epa,
                    "success": 1.0 if epa > 0 else 0.0,
                    "yards_gained": yards,
                    "rushing_yards": 0.0 if is_pass else yards,
                    "passing_yards": yards if is_pass else 0.0,
                    "sack": 0.0,
                    "interception": float(rng.random() < 0.02),
                    "fumble_lost": float(rng.random() < 0.02),
                    "qb_kneel": 0,
                    "qb_spike": 0,
                    "two_point_attempt": 0,
                    "season_type": "REG",
                }
            )
    return rows


def build_mini_league(seasons: list[int] | None = None, seed: int = 7) -> tuple[dict[int, pl.DataFrame], pl.DataFrame]:
    seasons = seasons or [2021, 2022, 2023, 2024]
    rng = np.random.default_rng(seed)
    pbp_by_season: dict[int, pl.DataFrame] = {}
    schedule_rows = []
    for season in seasons:
        plays = []
        for week, pairs in enumerate(PAIRINGS, start=1):
            for home, away in pairs:
                game_id = f"{season}_{week:02d}_{away}_{home}"
                off_h, def_h = TALENT[home]
                off_a, def_a = TALENT[away]
                # Rough margin from net talent + HFA so the spread model has a target.
                net = (off_h - def_h) - (off_a - def_a)
                margin = 25.0 * net + 2.0 + float(rng.normal(0, 6))
                home_score = int(round(24 + margin / 2))
                away_score = int(round(24 - margin / 2))
                spread_line = 25.0 * net + 2.0  # PFR: positive = home favored
                plays.extend(_plays_for_game(rng, game_id, season, week, home, away))
                schedule_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "gameday": f"{season}-09-{week:02d}",
                        "home_team": home,
                        "away_team": away,
                        "home_score": home_score,
                        "away_score": away_score,
                        "spread_line": spread_line,
                        "total_line": 47.0,
                        "result": home_score - away_score,
                    }
                )
        pbp_by_season[season] = pl.DataFrame(plays)
    return pbp_by_season, pl.DataFrame(schedule_rows)


def write_mini_league(data_root: Path, seasons: list[int] | None = None) -> None:
    """Write bronze parquet files under an isolated data root."""
    pbp_by_season, schedules = build_mini_league(seasons)
    pbp_dir = data_root / "bronze" / "pbp"
    pbp_dir.mkdir(parents=True, exist_ok=True)
    for season, df in pbp_by_season.items():
        write_parquet(df, pbp_dir / f"pbp_{season}.parquet")
    write_parquet(schedules, data_root / "bronze" / "schedules.parquet")
