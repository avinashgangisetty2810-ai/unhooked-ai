"""Unit tests for GenAI coaching features (core.features) — LLM calls mocked."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from core import db, features


@pytest.fixture()
def profile() -> db.Profile:
    profile_id = db.create_profile(
        name="Ravi",
        habit="Smoking",
        goal="Quit completely",
        triggers="stress, chai breaks",
        motivation="my daughter",
        daily_cost=200.0,
        daily_minutes=45,
    )
    result = db.get_profile(profile_id)
    assert result is not None
    return result


class TestCrisisDetection:
    @pytest.mark.parametrize(
        "text",
        ["I want to kill myself", "thinking about SUICIDE", "I might hurt myself tonight"],
    )
    def test_crisis_language_detected(self, text: str) -> None:
        assert features.is_crisis(text) is True

    @pytest.mark.parametrize(
        "text",
        ["I really want a cigarette", "today was hard", "this habit is killing my savings"],
    )
    def test_normal_language_not_flagged(self, text: str) -> None:
        assert features.is_crisis(text) is False

    def test_coach_reply_short_circuits_on_crisis(self, profile: db.Profile) -> None:
        with patch.object(features, "chat") as chat:
            reply = features.coach_reply(profile, [], "I want to end my life")
        chat.assert_not_called()
        assert reply == features.CRISIS_MESSAGE
        assert "AASRA" in reply


class TestGeneratePlan:
    def test_valid_plan_returned(self, profile: db.Profile) -> None:
        plan = {"summary": "s", "weeks": [{"week": 1}], "mantra": "m"}
        with patch.object(features, "chat_json", return_value=plan):
            assert features.generate_plan(profile) == plan

    @pytest.mark.parametrize("bad_plan", [{}, {"weeks": []}, {"weeks": "not a list"}])
    def test_plan_without_weeks_rejected(self, profile: db.Profile, bad_plan: dict[str, Any]) -> None:
        with patch.object(features, "chat_json", return_value=bad_plan):
            with pytest.raises(ValueError, match="weekly stages"):
                features.generate_plan(profile)

    def test_prompt_is_personalized(self, profile: db.Profile) -> None:
        with patch.object(features, "chat_json", return_value={"weeks": [{"week": 1}]}) as chat_json:
            features.generate_plan(profile)
        prompt = chat_json.call_args.args[0][0]["content"]
        assert "Smoking" in prompt
        assert "my daughter" in prompt


class TestRelapseRisk:
    def test_no_checkins_returns_unknown_without_llm(self, profile: db.Profile) -> None:
        with patch.object(features, "chat_json") as chat_json:
            risk = features.relapse_risk(profile)
        chat_json.assert_not_called()
        assert risk["level"] == "unknown"

    def test_invalid_level_coerced_to_watch(self, profile: db.Profile) -> None:
        db.upsert_checkin(
            profile_id=profile.id, status="clean", mood=3, craving=3, note="", ai_response=""
        )
        with patch.object(features, "chat_json", return_value={"level": "catastrophic", "reason": "r"}):
            assert features.relapse_risk(profile)["level"] == "watch"

    def test_valid_level_passes_through(self, profile: db.Profile) -> None:
        db.upsert_checkin(
            profile_id=profile.id, status="clean", mood=4, craving=1, note="", ai_response=""
        )
        with patch.object(features, "chat_json", return_value={"level": "low", "reason": "steady"}):
            assert features.relapse_risk(profile)["level"] == "low"


class TestContextAndReplies:
    def test_profile_context_includes_checkins(self, profile: db.Profile) -> None:
        db.upsert_checkin(
            profile_id=profile.id, status="slip", mood=2, craving=9, note="rough day", ai_response=""
        )
        context = features._profile_context(profile)
        assert "Ravi" in context
        assert "rough day" in context
        assert "craving 9/10" in context

    def test_checkin_nudge_passes_todays_data(self, profile: db.Profile) -> None:
        with patch.object(features, "chat", return_value="proud of you") as chat:
            nudge = features.checkin_nudge(profile, status="clean", mood=5, craving=1, note="gym")
        assert nudge == "proud of you"
        prompt = chat.call_args.args[0][0]["content"]
        assert "status=clean" in prompt
        assert "gym" in prompt

    def test_sos_intervention_returns_llm_payload(self, profile: db.Profile) -> None:
        payload = {"urge_surf": "u", "distraction": "d", "reframe": "r", "future_you": "f", "grounding": "g"}
        with patch.object(features, "chat_json", return_value=payload):
            assert features.sos_intervention(profile, trigger="stress", intensity=8) == payload

    def test_sos_prompt_demands_varied_personalized_grounding(self, profile: db.Profile) -> None:
        with patch.object(features, "chat_json", return_value={}) as cj:
            features.sos_intervention(profile, trigger="bored at home", intensity=9)
        prompt = cj.call_args.args[0][0]["content"]
        assert "grounding_title" in prompt
        assert "Do NOT" in prompt and "box breathing" in prompt  # forbids the default technique
        assert "bored at home" in prompt
        assert "Smoking" in prompt  # habit is injected verbatim
        assert "never the same wording for different triggers" in prompt

    def test_reframe_thought_includes_thought_in_prompt(self, profile: db.Profile) -> None:
        with patch.object(features, "chat_json", return_value={"distortion": "permission-giving"}) as cj:
            features.reframe_thought(profile, "just one won't hurt")
        assert "just one won't hurt" in cj.call_args.args[0][0]["content"]

    def test_coach_reply_uses_history_window(self, profile: db.Profile) -> None:
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        with patch.object(features, "chat", return_value="reply") as chat:
            features.coach_reply(profile, history, "help me")
        messages = chat.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert len(messages) == 1 + 12 + 1  # system + last 12 history + new message
        assert messages[-1]["content"] == "help me"

    def test_weekly_insights_returns_llm_payload(self, profile: db.Profile) -> None:
        payload = {"headline": "h", "wins": ["w"], "watch_outs": ["x"], "next_week_focus": "f"}
        with patch.object(features, "chat_json", return_value=payload):
            assert features.weekly_insights(profile) == payload


class TestStreamingVariants:
    def test_coach_reply_stream_short_circuits_on_crisis(self, profile: db.Profile) -> None:
        with patch.object(features, "chat_stream") as cs:
            chunks = list(features.coach_reply_stream(profile, [], "I want to end my life"))
        cs.assert_not_called()
        assert chunks == [features.CRISIS_MESSAGE]

    def test_coach_reply_stream_delegates_with_same_messages(self, profile: db.Profile) -> None:
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        with patch.object(features, "chat_stream", return_value=iter(["a", "b"])) as cs:
            chunks = list(features.coach_reply_stream(profile, history, "help me"))
        assert chunks == ["a", "b"]
        messages = cs.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert len(messages) == 1 + 12 + 1
        assert messages[-1]["content"] == "help me"

    def test_checkin_nudge_stream_delegates_with_todays_data(self, profile: db.Profile) -> None:
        with patch.object(features, "chat_stream", return_value=iter(["nice"])) as cs:
            chunks = list(features.checkin_nudge_stream(profile, status="clean", mood=5, craving=1, note="gym"))
        assert chunks == ["nice"]
        prompt = cs.call_args.args[0][0]["content"]
        assert "status=clean" in prompt
        assert "gym" in prompt

    def test_json_features_forward_on_progress_for_live_ui(self, profile: db.Profile) -> None:
        db.upsert_checkin(profile_id=profile.id, status="clean", mood=4, craving=3, note="", ai_response="")
        callback = lambda _text: None  # noqa: E731
        plan = {"weeks": [{"week": 1}]}
        calls = [
            lambda: features.generate_plan(profile, on_progress=callback),
            lambda: features.sos_intervention(profile, trigger="stress", intensity=8, on_progress=callback),
            lambda: features.reframe_thought(profile, "just one", on_progress=callback),
            lambda: features.relapse_risk(profile, on_progress=callback),
            lambda: features.weekly_insights(profile, on_progress=callback),
        ]
        for call in calls:
            with patch.object(features, "chat_json", return_value=plan) as cj:
                call()
            assert cj.call_args.kwargs["on_delta"] is callback


class TestToneContract:
    """Every user-facing AI prompt must carry the motivating/plain-language tone rules."""

    def test_tone_rules_demand_motivating_plain_interactive_voice(self) -> None:
        assert "Motivating" in features._TONE_RULES
        assert "Plain, simple English" in features._TONE_RULES
        assert "question" in features._TONE_RULES  # interactive: invite a reply/action

    def test_json_feature_prompts_include_tone_rules(self, profile: db.Profile) -> None:
        db.upsert_checkin(profile_id=profile.id, status="clean", mood=4, craving=3, note="", ai_response="")
        plan = {"weeks": [{"week": 1}]}
        calls = [
            lambda: features.generate_plan(profile),
            lambda: features.sos_intervention(profile, trigger="stress", intensity=8),
            lambda: features.reframe_thought(profile, "just one"),
            lambda: features.relapse_risk(profile),
            lambda: features.weekly_insights(profile),
        ]
        for call in calls:
            with patch.object(features, "chat_json", return_value=plan) as cj:
                call()
            assert features._TONE_RULES in cj.call_args.args[0][0]["content"]

    def test_text_feature_prompts_include_tone_rules(self, profile: db.Profile) -> None:
        with patch.object(features, "chat", return_value="ok") as chat:
            features.checkin_nudge(profile, status="clean", mood=5, craving=1, note="")
        assert features._TONE_RULES in chat.call_args.args[0][0]["content"]
        with patch.object(features, "chat", return_value="ok") as chat:
            features.coach_reply(profile, [], "help")
        assert features._TONE_RULES in chat.call_args.args[0][0]["content"]  # system prompt
