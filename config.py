"""Project-wide settings for the NFL spread model.

Think of this file as the model's "rulebook." Every script reads the same
seasons, week filters, stat names, and file locations from here so we never
accidentally train on Week 18 in one place and exclude it in another.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Tests can point the whole pipeline at a temporary folder by setting
# NFL_MODEL_DATA_DIR. Production runs use ./data next to this file.
ROOT = Path(__file__).resolve().parent


def data_dir() -> Path:
    """Root folder for parquet outputs (bronze / silver / gold)."""
    override = os.environ.get("NFL_MODEL_DATA_DIR")
    return Path(override) if override else ROOT / "data"


def bronze_dir() -> Path:
    return data_dir() / "bronze"


def silver_dir() -> Path:
    return data_dir() / "silver"


def gold_dir() -> Path:
    return data_dir() / "gold"


def pbp_dir() -> Path:
    return bronze_dir() / "pbp"


def pbp_path(season: int) -> Path:
    return pbp_dir() / f"pbp_{season}.parquet"


def schedules_path() -> Path:
    return bronze_dir() / "schedules.parquet"


def team_game_stats_raw_path() -> Path:
    return silver_dir() / "team_game_stats_raw.parquet"


def team_game_stats_path() -> Path:
    return silver_dir() / "team_game_stats.parquet"


def team_effects_path() -> Path:
    return silver_dir() / "team_effects.parquet"


def stability_oddeven_path() -> Path:
    return gold_dir() / "stability_oddeven.parquet"


def weight_curve_path() -> Path:
    return gold_dir() / "weight_curve.parquet"


def priors_path() -> Path:
    return gold_dir() / "priors.parquet"


def prior_reliability_path() -> Path:
    return gold_dir() / "prior_reliability.parquet"


def spread_coefficients_path() -> Path:
    return gold_dir() / "spread_coefficients.parquet"


def backtest_path() -> Path:
    return gold_dir() / "backtest.parquet"


def backtest_games_path() -> Path:
    return gold_dir() / "backtest_games.parquet"


def board_path() -> Path:
    return gold_dir() / "board.parquet"


def overrides_path() -> Path:
    return data_dir() / "overrides.jsonl"


# ---------------------------------------------------------------------------
# Seasons and week filters
# ---------------------------------------------------------------------------
# The 17-game regular season started in 2021. Before that, teams played 16
# regular-season games (weeks 1–17 with one bye). We always drop the FINAL
# regular-season week because starters are often rested.
SEASONS: list[int] = list(range(2015, 2026))  # 2015 through 2025 inclusive
SEVENTEEN_GAME_SEASON_START = 2021

# "Usable" means: include in training / stability studies / ratings.
# Pre-2021: weeks 1–16 (exclude week 17). 2021+: weeks 1–17 (exclude week 18).
REGULAR_SEASON_WEEKS = 17


def last_usable_week(season: int) -> int:
    """Last regular-season week to KEEP (the finale is excluded)."""
    if season < SEVENTEEN_GAME_SEASON_START:
        return 16
    return 17


def is_usable_regular_week(season: int, week: int) -> bool:
    """True for regular-season weeks that are allowed in the study."""
    return 1 <= int(week) <= last_usable_week(int(season))


# ---------------------------------------------------------------------------
# Candidate stats
# ---------------------------------------------------------------------------
# Efficiency stats are expected to PASS the stability tests. Turnover stats
# are included on purpose as "noise controls" — they should look unstable.
# Each efficiency stat is stored as both offense (produced) and defense
# (allowed) on the team-game file.
EFFICIENCY_STATS: list[str] = [
    "epa_per_play",
    "success_rate",
    "yards_per_rush",
    "yards_per_dropback",
    "explosive_play_rate",
]

NOISE_CONTROL_STATS: list[str] = [
    "turnover_margin",
    "fumbles_lost",
]

# Ridge is fit on the offense (or team-level) observation:
#   observed = offense_effect[team] + defense_effect[opponent] + hfa * home
RIDGE_ALPHA = 50.0

# How many rushing yards / passing yards count as "explosive."
EXPLOSIVE_RUSH_YARDS = 10
EXPLOSIVE_PASS_YARDS = 20

# First-N study grid (games already played vs rest of season).
FIRST_N_VALUES: list[int] = [3, 4, 5, 6, 8, 10, 12]

# Walk-forward backtest window. Week 1–2 lines are mostly the prior; we
# start reporting at week 3 once a little current-season data exists.
BACKTEST_SEASONS: list[int] = list(range(2019, 2026))
BACKTEST_MIN_WEEK = 3
EDGE_THRESHOLDS: tuple[float, ...] = (1.5, 2.5, 3.5)

# The spread model is built off net EPA (offense EPA − defense EPA allowed).
PRIMARY_OFF_STAT = "off_epa_per_play"
PRIMARY_DEF_STAT = "def_epa_per_play"

# Teams that moved cities keep one franchise id so last year's prior still
# attaches to this year's roster.
FRANCHISE_MAP: dict[str, str] = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
}


def franchise(team: str | None) -> str | None:
    """Map a historical abbreviation onto the current franchise id."""
    if team is None:
        return None
    return FRANCHISE_MAP.get(str(team), str(team))


def efficiency_columns(side: str) -> list[str]:
    """Return column names like off_epa_per_play or def_epa_per_play."""
    if side not in {"off", "def"}:
        raise ValueError(f"side must be 'off' or 'def', got {side!r}")
    return [f"{side}_{name}" for name in EFFICIENCY_STATS]


def all_model_stat_columns() -> list[str]:
    """Every stat we run stability / priors / blending on."""
    return efficiency_columns("off") + efficiency_columns("def") + list(NOISE_CONTROL_STATS)


def ensure_data_dirs() -> None:
    """Create bronze/silver/gold folders if they do not exist yet."""
    for path in (pbp_dir(), silver_dir(), gold_dir()):
        path.mkdir(parents=True, exist_ok=True)
