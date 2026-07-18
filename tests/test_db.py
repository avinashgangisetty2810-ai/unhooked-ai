"""Unit tests for the SQLite persistence layer (core.db)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from core import db


def _make_profile(**overrides: object) -> int:
    kwargs: dict = {
        "name": "Asha",
        "habit": "Doomscrolling",
        "goal": "Quit completely",
        "triggers": "boredom, late nights",
        "motivation": "sleep better",
        "daily_cost": 0.0,
        "daily_minutes": 90,
    }
    kwargs.update(overrides)
    return db.create_profile(**kwargs)


class TestProfiles:
    def test_create_and_get_roundtrip(self) -> None:
        profile_id = _make_profile()
        profile = db.get_profile(profile_id)
        assert profile is not None
        assert profile.name == "Asha"
        assert profile.habit == "Doomscrolling"
        assert profile.daily_minutes == 90
        assert profile.plan == {}

    def test_get_missing_profile_returns_none(self) -> None:
        assert db.get_profile(9999) is None

    def test_list_profiles_newest_first(self) -> None:
        first = _make_profile(name="First")
        second = _make_profile(name="Second")
        profiles = db.list_profiles()
        assert [p.id for p in profiles] == [second, first]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            _make_profile(name="   ")

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _make_profile(daily_cost=-5.0)

    def test_save_and_load_plan(self) -> None:
        profile_id = _make_profile()
        plan = {"summary": "s", "weeks": [{"week": 1}], "mantra": "m"}
        db.save_plan(profile_id, plan)
        profile = db.get_profile(profile_id)
        assert profile is not None
        assert profile.plan == plan

    def test_corrupt_plan_json_yields_empty_plan(self) -> None:
        profile_id = _make_profile()
        with db._connect() as conn:
            conn.execute("UPDATE profiles SET plan_json = ? WHERE id = ?", ("{not json", profile_id))
        profile = db.get_profile(profile_id)
        assert profile is not None
        assert profile.plan == {}


class TestCheckins:
    def test_upsert_inserts_then_updates_same_day(self) -> None:
        profile_id = _make_profile()
        db.upsert_checkin(profile_id=profile_id, status="clean", mood=4, craving=2, note="ok", ai_response="nice")
        db.upsert_checkin(profile_id=profile_id, status="slip", mood=2, craving=8, note="bad", ai_response="hang on")
        checkins = db.get_checkins(profile_id)
        assert len(checkins) == 1
        assert checkins[0]["status"] == "slip"
        assert checkins[0]["craving"] == 8

    @pytest.mark.parametrize(
        ("status", "mood", "craving"),
        [("partying", 3, 3), ("clean", 0, 3), ("clean", 6, 3), ("clean", 3, -1), ("clean", 3, 11)],
    )
    def test_invalid_checkin_values_rejected(self, status: str, mood: int, craving: int) -> None:
        profile_id = _make_profile()
        with pytest.raises(ValueError):
            db.upsert_checkin(profile_id=profile_id, status=status, mood=mood, craving=craving, note="", ai_response="")

    def test_get_checkins_respects_limit(self) -> None:
        profile_id = _make_profile()
        _seed_checkins(profile_id, ["clean"] * 5)
        assert len(db.get_checkins(profile_id, limit=3)) == 3


def _seed_checkins(profile_id: int, statuses: list[str], *, start_today: bool = True) -> None:
    """Insert one check-in per day counting back from today (or yesterday)."""
    day = db.local_today() if start_today else db.local_today() - timedelta(days=1)
    with db._connect() as conn:
        for status in statuses:
            conn.execute(
                "INSERT INTO checkins (profile_id, day, status, mood, craving, note, ai_response,"
                " created_at) VALUES (?, ?, ?, 3, 3, '', '', ?)",
                (profile_id, day.isoformat(), status, db._now()),
            )
            day -= timedelta(days=1)


class TestStreak:
    def test_no_checkins_means_zero(self) -> None:
        assert db.current_streak(_make_profile()) == 0

    def test_consecutive_clean_days(self) -> None:
        profile_id = _make_profile()
        _seed_checkins(profile_id, ["clean", "clean", "clean"])
        assert db.current_streak(profile_id) == 3

    def test_slip_today_resets_to_zero(self) -> None:
        profile_id = _make_profile()
        _seed_checkins(profile_id, ["slip", "clean", "clean"])
        assert db.current_streak(profile_id) == 0

    def test_missing_today_counts_from_yesterday(self) -> None:
        profile_id = _make_profile()
        _seed_checkins(profile_id, ["clean", "clean"], start_today=False)
        assert db.current_streak(profile_id) == 2

    def test_streak_broken_by_gap(self) -> None:
        profile_id = _make_profile()
        with db._connect() as conn:
            today = db.local_today()
            for offset in (0, 2, 3):  # gap at day -1
                conn.execute(
                    "INSERT INTO checkins (profile_id, day, status, mood, craving, note,"
                    " ai_response, created_at) VALUES (?, ?, 'clean', 3, 3, '', '', ?)",
                    (profile_id, (today - timedelta(days=offset)).isoformat(), db._now()),
                )
        assert db.current_streak(profile_id) == 1


class TestChatMessages:
    def test_roundtrip_in_chronological_order(self) -> None:
        profile_id = _make_profile()
        db.add_chat_message(profile_id, "user", "hi")
        db.add_chat_message(profile_id, "assistant", "hello")
        messages = db.get_chat_messages(profile_id)
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "hi"

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValueError, match="role"):
            db.add_chat_message(_make_profile(), "system", "nope")


class TestEvents:
    def test_log_and_get_events(self) -> None:
        profile_id = _make_profile()
        db.log_event(profile_id, "risk", {"level": "low", "reason": "steady"})
        events = db.get_events(profile_id, "risk")
        assert len(events) == 1
        assert events[0]["level"] == "low"
        assert "created_at" in events[0]

    def test_events_filtered_by_kind(self) -> None:
        profile_id = _make_profile()
        db.log_event(profile_id, "risk", {"level": "low"})
        db.log_event(profile_id, "sos", {"trigger": "stress"})
        assert len(db.get_events(profile_id, "risk")) == 1
        assert len(db.get_events(profile_id, "sos")) == 1

    def test_corrupt_event_payload_skipped(self) -> None:
        profile_id = _make_profile()
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO events (profile_id, kind, payload_json, created_at) VALUES (?, 'risk', '{broken', ?)",
                (profile_id, db._now()),
            )
        assert db.get_events(profile_id, "risk") == []
