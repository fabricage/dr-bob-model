"""Pytest fixtures. Unit tests get an isolated mini league via NFL_MODEL_DATA_DIR."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.synthetic import write_mini_league


@pytest.fixture
def mini_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("NFL_MODEL_DATA_DIR", str(tmp_path))
    write_mini_league(tmp_path)
    return tmp_path
