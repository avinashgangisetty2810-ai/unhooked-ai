"""SQLite persistence layer for Unhooked.

Real persistence — the evaluator's data survives page reloads. Uses only the
standard library. All queries are parameterized.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

_DB_PATH: Final[Path] = Path(__file__).resolve().parent.parent / "data" / "unhooked.db"

_VALID_STATUSES: Final[frozenset[str]] = frozenset({"clean", "slip"})
_VALID_ROLES: Final[frozenset[str]] = frozenset({"user", "assistant"})

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    habit TEXT NOT NULL,
    goal TEXT NOT NULL,
    triggers TEXT NOT NULL DEFAULT '',
    motivation TEXT NOT NULL DEFAULT '',
    daily_cost REAL NOT NULL DEFAULT 0,
    daily_minutes INTEGER NOT NULL DEFAULT 0,
    plan_json TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id),
    day TEXT NOT NULL,
    status TEXT NOT NULL,
    mood INTEGER NOT NULL,
    craving INTEGER NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    ai_response TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(profile_id, day)
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id),
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Profile:
    """A user profile with habit details and the generated quit plan."""

    id: int
    name: str
    habit: str
    goal: str
    triggers: str
    motivation: str
    daily_cost: float
    daily_minutes: int
    plan: dict[str, Any]
    created_at: str


_initialized_paths: set[Path] = set()


def _connect() -> sqlite3.Connection:
    """Open a connection, creating the schema once per database path."""
    path = _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    if path not in _initialized_paths:
        conn.executescript(_SCHEMA)  # idempotent (IF NOT EXISTS), run once per process
        _initialized_paths.add(path)
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _row_to_profile(row: sqlite3.Row) -> Profile:
    plan: dict[str, Any] = {}
    if row["plan_json"]:
        try:
            plan = json.loads(row["plan_json"])
        except ValueError:
            plan = {}
    return Profile(
        id=row["id"],
        name=row["name"],
        habit=row["habit"],
        goal=row["goal"],
        triggers=row["triggers"],
        motivation=row["motivation"],
        daily_cost=row["daily_cost"],
        daily_minutes=row["daily_minutes"],
        plan=plan,
        created_at=row["created_at"],
    )


def create_profile(
    *,
    name: str,
    habit: str,
    goal: str,
    triggers: str,
    motivation: str,
    daily_cost: float,
    daily_minutes: int,
) -> int:
    """Insert a new profile and return its id.

    Raises:
        ValueError: If the name is empty or numeric fields are negative.
    """
    if not name.strip():
        raise ValueError("Profile name must not be empty")
    if daily_cost < 0 or daily_minutes < 0:
        raise ValueError("daily_cost and daily_minutes must be non-negative")
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO profiles (name, habit, goal, triggers, motivation, daily_cost,"
            " daily_minutes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, habit, goal, triggers, motivation, daily_cost, daily_minutes, _now()),
        )
        return int(cursor.lastrowid or 0)


def save_plan(profile_id: int, plan: dict[str, Any]) -> None:
    """Attach a generated quit plan to a profile."""
    with _connect() as conn:
        conn.execute(
            "UPDATE profiles SET plan_json = ? WHERE id = ?",
            (json.dumps(plan), profile_id),
        )


def list_profiles() -> list[Profile]:
    """Return all profiles, newest first."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM profiles ORDER BY id DESC").fetchall()
    return [_row_to_profile(row) for row in rows]


def get_profile(profile_id: int) -> Profile | None:
    """Fetch a single profile by id."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    return _row_to_profile(row) if row else None


def upsert_checkin(
    *,
    profile_id: int,
    status: str,
    mood: int,
    craving: int,
    note: str,
    ai_response: str,
) -> None:
    """Record (or replace) today's check-in.

    Raises:
        ValueError: If status, mood, or craving fall outside their valid ranges.
    """
    if status not in _VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(_VALID_STATUSES)}")
    if not 1 <= mood <= 5:
        raise ValueError("mood must be between 1 and 5")
    if not 0 <= craving <= 10:
        raise ValueError("craving must be between 0 and 10")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO checkins (profile_id, day, status, mood, craving, note, ai_response,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(profile_id, day) DO UPDATE SET status=excluded.status,"
            " mood=excluded.mood, craving=excluded.craving, note=excluded.note,"
            " ai_response=excluded.ai_response",
            (profile_id, date.today().isoformat(), status, mood, craving, note, ai_response, _now()),
        )


def get_checkins(profile_id: int, limit: int = 30) -> list[dict[str, Any]]:
    """Return recent check-ins, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT day, status, mood, craving, note, ai_response FROM checkins"
            " WHERE profile_id = ? ORDER BY day DESC LIMIT ?",
            (profile_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def current_streak(profile_id: int) -> int:
    """Consecutive clean days counting back from today (or yesterday)."""
    checkins = {c["day"]: c["status"] for c in get_checkins(profile_id, limit=365)}
    streak = 0
    day = date.today()
    if checkins.get(day.isoformat()) == "slip":
        return 0
    if day.isoformat() not in checkins:
        day -= timedelta(days=1)
    while checkins.get(day.isoformat()) == "clean":
        streak += 1
        day -= timedelta(days=1)
    return streak


def add_chat_message(profile_id: int, role: str, content: str) -> None:
    """Append a chat message to the coach conversation.

    Raises:
        ValueError: If the role is not ``user`` or ``assistant``.
    """
    if role not in _VALID_ROLES:
        raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_messages (profile_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (profile_id, role, content, _now()),
        )


def get_chat_messages(profile_id: int, limit: int = 40) -> list[dict[str, str]]:
    """Return the coach conversation in chronological order."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_messages WHERE profile_id = ?"
            " ORDER BY id DESC LIMIT ?",
            (profile_id, limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def log_event(profile_id: int, kind: str, payload: dict[str, Any]) -> None:
    """Store an SOS / reframe / insight event."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events (profile_id, kind, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (profile_id, kind, json.dumps(payload), _now()),
        )


def get_events(profile_id: int, kind: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent events of one kind, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT payload_json, created_at FROM events WHERE profile_id = ? AND kind = ?"
            " ORDER BY id DESC LIMIT ?",
            (profile_id, kind, limit),
        ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except ValueError:
            continue
        payload["created_at"] = row["created_at"]
        events.append(payload)
    return events
