"""Explicit injury / lineup overrides applied AFTER the model line.

Overrides never hide inside EPA ratings. Each one is a documented point
adjustment to the home margin, with a reason, a timestamp, and an expiry.

    adjusted_margin = model_margin + sum(active overrides in home-margin units)

If you tag a TEAM (not a game_id), `points` is that team's impact
("backup QB, −6.5" means that team is 6.5 points worse). We flip the
sign when the tagged team is the away side so the math stays in
home-margin units.

Storage is a JSONL file (one JSON object per line) so the audit trail is
just the file. Expiring a record sets status=expired and expired_at; the
original row is not deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from helpers import print_header


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_overrides(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or config.overrides_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def save_overrides(rows: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or config.overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=True) for r in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")
    return path


def add_override(
    *,
    points: float,
    reason: str,
    source: str = "manual",
    confidence: str = "medium",
    position: str | None = None,
    game_id: str | None = None,
    team: str | None = None,
    season: int | None = None,
    week: int | None = None,
    expires: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    if game_id is None and (team is None or season is None or week is None):
        raise ValueError("Provide game_id, or team + season + week.")
    rec = {
        "id": str(uuid.uuid4())[:8],
        "game_id": game_id,
        "team": team,
        "season": season,
        "week": week,
        "points": float(points),
        "position": position,
        "reason": reason,
        "source": source,
        "confidence": confidence,
        "created_at": _now(),
        "expires": expires,
        "expired_at": None,
        "status": "active",
    }
    rows = load_overrides(path)
    rows.append(rec)
    save_overrides(rows, path)
    return rec


def expire_override(override_id: str, path: Path | None = None) -> dict[str, Any]:
    rows = load_overrides(path)
    found = None
    for rec in rows:
        if rec.get("id") == override_id:
            rec["status"] = "expired"
            rec["expired_at"] = _now()
            found = rec
    if found is None:
        raise KeyError(f"No override with id={override_id}")
    save_overrides(rows, path)
    return found


def is_active(rec: dict[str, Any], now: datetime | None = None) -> bool:
    if rec.get("status") != "active":
        return False
    now = now or datetime.now(timezone.utc)
    expires = _parse_dt(rec.get("expires"))
    if expires is not None and expires <= now:
        return False
    return True


def matches(
    rec: dict[str, Any],
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    season: int,
    week: int,
) -> bool:
    if rec.get("game_id") and rec["game_id"] == game_id:
        return True
    if rec.get("team") and rec.get("season") == season and rec.get("week") == week:
        return rec["team"] in {home_team, away_team}
    return False


def home_margin_delta(
    rec: dict[str, Any],
    *,
    home_team: str,
    away_team: str,
) -> float:
    """Convert a stored override into home-margin points."""
    pts = float(rec["points"])
    team = rec.get("team")
    if not team:
        # game_id overrides are already stored in home-margin units.
        return pts
    if team == home_team:
        return pts
    if team == away_team:
        return -pts
    return 0.0


def apply_overrides(
    model_margin: float,
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    season: int,
    week: int,
    overrides: list[dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    overrides = overrides if overrides is not None else load_overrides()
    applied = []
    total = 0.0
    for rec in overrides:
        if not is_active(rec, now=now):
            continue
        if not matches(
            rec,
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            season=season,
            week=week,
        ):
            continue
        delta = home_margin_delta(rec, home_team=home_team, away_team=away_team)
        total += delta
        applied.append({**rec, "home_margin_delta": delta})
    return {
        "adjusted_margin": float(model_margin) + total,
        "points": total,
        "applied": applied,
    }


def _print_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(none)")
        return
    for rec in rows:
        print(
            f"{rec.get('id')}  {rec.get('status'):8}  pts={rec.get('points'):+g}  "
            f"pos={rec.get('position') or '-'}  "
            f"game={rec.get('game_id') or '-'}  "
            f"team={rec.get('team') or '-'} s={rec.get('season')} w={rec.get('week')}  "
            f"{rec.get('reason')}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage injury/lineup overrides.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    add_p = sub.add_parser("add", help="Add an override.")
    add_p.add_argument("--points", type=float, required=True)
    add_p.add_argument("--reason", required=True)
    add_p.add_argument("--source", default="manual")
    add_p.add_argument("--confidence", default="medium")
    add_p.add_argument("--position", default=None, help="Position tag. Use QB for quarterbacks.")
    add_p.add_argument("--game-id", dest="game_id", default=None)
    add_p.add_argument("--team", default=None)
    add_p.add_argument("--season", type=int, default=None)
    add_p.add_argument("--week", type=int, default=None)
    add_p.add_argument("--expires", default=None, help="ISO-8601 timestamp.")

    sub.add_parser("list", help="List all overrides (including expired).")

    exp = sub.add_parser("expire", help="Expire an override by id (keeps the audit row).")
    exp.add_argument("--id", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "add":
        rec = add_override(
            points=args.points,
            reason=args.reason,
            source=args.source,
            confidence=args.confidence,
            position=args.position,
            game_id=args.game_id,
            team=args.team,
            season=args.season,
            week=args.week,
            expires=args.expires,
        )
        print_header("Override added")
        _print_rows([rec])
        print(f"stored in {config.overrides_path()}")
        return 0
    if args.cmd == "list":
        print_header("Overrides")
        _print_rows(load_overrides())
        return 0
    if args.cmd == "expire":
        rec = expire_override(args.id)
        print_header("Override expired (row kept for audit)")
        _print_rows([rec])
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
