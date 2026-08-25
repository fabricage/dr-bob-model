"""Phase 7 tests: expired overrides do not apply; expiry keeps the audit row."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from model.overrides import add_override, apply_overrides, expire_override, load_overrides


def test_expire_keeps_audit_and_stops_applying(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NFL_MODEL_DATA_DIR", str(tmp_path))
    rec = add_override(
        points=-6.5,
        reason="backup QB",
        position="QB",
        team="KC",
        season=2024,
        week=5,
        source="test",
        confidence="high",
    )
    applied = apply_overrides(
        3.0,
        game_id="2024_05_BUF_KC",
        home_team="KC",
        away_team="BUF",
        season=2024,
        week=5,
    )
    assert applied["points"] == -6.5
    assert applied["adjusted_margin"] == 3.0 - 6.5

    expired = expire_override(rec["id"])
    assert expired["status"] == "expired"
    assert expired["expired_at"]
    # Original fields stay on the row (audit trail).
    assert expired["reason"] == "backup QB"
    assert expired["points"] == -6.5
    trail = load_overrides()
    assert any(r["id"] == rec["id"] and r["status"] == "expired" for r in trail)

    applied2 = apply_overrides(
        3.0,
        game_id="2024_05_BUF_KC",
        home_team="KC",
        away_team="BUF",
        season=2024,
        week=5,
    )
    assert applied2["points"] == 0.0
    assert applied2["adjusted_margin"] == 3.0


def test_past_expires_timestamp_does_not_apply(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NFL_MODEL_DATA_DIR", str(tmp_path))
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    add_override(
        points=-3.0,
        reason="expired by timestamp",
        game_id="2024_05_BUF_KC",
        expires=past,
    )
    applied = apply_overrides(
        1.0,
        game_id="2024_05_BUF_KC",
        home_team="KC",
        away_team="BUF",
        season=2024,
        week=5,
    )
    assert applied["points"] == 0.0


def test_away_team_tag_flips_sign(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NFL_MODEL_DATA_DIR", str(tmp_path))
    add_override(
        points=-6.5,
        reason="away backup QB",
        position="QB",
        team="BUF",
        season=2024,
        week=5,
    )
    applied = apply_overrides(
        0.0,
        game_id="2024_05_BUF_KC",
        home_team="KC",
        away_team="BUF",
        season=2024,
        week=5,
    )
    # BUF is away, so -6.5 to BUF is +6.5 to the home margin.
    assert applied["points"] == 6.5
