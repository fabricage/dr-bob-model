"""Phase 1 tests: config week rules and schedule spread completeness."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import config
from ingest.pull_nflverse import load_schedules, missing_spread_after_2015


def test_last_usable_week_excludes_finale() -> None:
    assert config.last_usable_week(2019) == 16  # 16-game era: drop week 17
    assert config.last_usable_week(2021) == 17  # 17-game era: drop week 18
    assert config.is_usable_regular_week(2019, 16)
    assert not config.is_usable_regular_week(2019, 17)
    assert config.is_usable_regular_week(2024, 17)
    assert not config.is_usable_regular_week(2024, 18)


def test_candidate_stats_include_noise_controls() -> None:
    names = config.all_model_stat_columns()
    assert "off_epa_per_play" in names
    assert "def_success_rate" in names
    assert "turnover_margin" in names
    assert "fumbles_lost" in names


@pytest.mark.skipif(
    not config.schedules_path().exists(),
    reason="bronze schedules not downloaded yet",
)
def test_schedules_have_no_missing_spread_line_after_2015() -> None:
    missing = missing_spread_after_2015(load_schedules())
    assert missing.height == 0, missing.head(10)


def test_missing_spread_helper_finds_nulls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NFL_MODEL_DATA_DIR", str(tmp_path))
    sched = pl.DataFrame(
        {
            "season": [2016, 2016, 2015],
            "game_type": ["REG", "REG", "REG"],
            "spread_line": [3.0, None, None],
            "home_team": ["KC", "BUF", "SF"],
            "away_team": ["BUF", "KC", "PHI"],
        }
    )
    config.bronze_dir().mkdir(parents=True)
    sched.write_parquet(config.schedules_path())
    missing = missing_spread_after_2015(load_schedules())
    assert missing.height == 1
    assert missing["home_team"][0] == "BUF"
