"""GenAI coaching features for Unhooked.

Every function here makes a real LLM call through :mod:`core.llm` and returns
structured, validated data for the UI. Prompts are grounded in CBT and
habit-science techniques (urge surfing, implementation intentions,
cognitive-distortion reframing).
"""

from __future__ import annotations

from typing import Any, Final

from core import db
from core.llm import chat, chat_json

_CRISIS_KEYWORDS: Final[tuple[str, ...]] = (
    "kill myself",
    "suicide",
    "end my life",
    "self harm",
    "self-harm",
    "hurt myself",
    "want to die",
)

CRISIS_MESSAGE: Final[str] = (
    "It sounds like you're going through something serious. You deserve real support "
    "right now — please reach out to a professional:\n\n"
    "- **India**: AASRA 24x7 helpline — 91-9820466726, or iCall — 9152987821\n"
    "- **International**: https://findahelpline.com\n\n"
    "I'm here for habit coaching, but a trained counsellor can help far better with this."
)


def is_crisis(text: str) -> bool:
    """Detect crisis language that must be routed to helpline info."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in _CRISIS_KEYWORDS)


def _profile_context(profile: db.Profile) -> str:
    streak = db.current_streak(profile.id)
    recent = db.get_checkins(profile.id, limit=7)
    lines = [
        f"User: {profile.name}",
        f"Habit they are breaking: {profile.habit}",
        f"Goal: {profile.goal}",
        f"Known triggers: {profile.triggers or 'not specified'}",
        f"Their motivation: {profile.motivation or 'not specified'}",
        f"Current clean streak: {streak} day(s)",
    ]
    if recent:
        lines.append("Recent check-ins (newest first):")
        for c in recent:
            lines.append(
                f"- {c['day']}: {c['status']}, mood {c['mood']}/5, craving {c['craving']}/10"
                + (f", note: {c['note']}" if c["note"] else "")
            )
    return "\n".join(lines)


def generate_plan(profile: db.Profile) -> dict[str, Any]:
    """Generate a personalized staged recovery plan as structured JSON."""
    prompt = f"""You are an addiction-recovery coach. Create a personalized 4-week plan to help
this person break their habit. Be specific to THEIR habit, triggers, and motivation — no
generic filler.

{_profile_context(profile)}

Return ONLY a JSON object with this exact shape:
{{
  "summary": "2-3 sentence personalized overview of the strategy",
  "weeks": [
    {{"week": 1, "theme": "short theme", "target": "concrete measurable target",
      "tactics": ["specific tactic 1", "specific tactic 2", "specific tactic 3"]}}
  ],
  "coping_toolkit": ["5 specific coping techniques matched to their triggers"],
  "trigger_defenses": [
    {{"trigger": "one of their triggers", "defense": "specific if-then plan"}}
  ],
  "mantra": "one short personal mantra referencing their motivation"
}}
Include exactly 4 weeks. Every tactic must be concrete and actionable."""
    plan = chat_json([{"role": "user", "content": prompt}])
    if "weeks" not in plan or not isinstance(plan["weeks"], list) or not plan["weeks"]:
        raise ValueError("Plan is missing weekly stages")
    return plan


def checkin_nudge(profile: db.Profile, *, status: str, mood: int, craving: int, note: str) -> str:
    """React to today's check-in with a context-aware coaching message."""
    streak = db.current_streak(profile.id)
    prompt = f"""You are a warm, direct recovery coach. The user just submitted today's check-in.

{_profile_context(profile)}

Today's check-in: status={status} (clean = resisted the habit, slip = gave in),
mood={mood}/5, craving intensity={craving}/10, note="{note or 'none'}"
Streak before today: {streak} day(s).

Write a 3-5 sentence personal response. Rules:
- If clean: celebrate specifically (reference their streak, motivation, or note). No generic praise.
- If slip: zero shame. Normalize it, extract one lesson from their note/triggers, give one
  concrete action for the next 24 hours.
- If craving >= 7: acknowledge how hard today was.
- Speak directly to {profile.name}. No headers, no bullet lists, just the message."""
    return chat([{"role": "user", "content": prompt}], temperature=0.8)


def sos_intervention(profile: db.Profile, *, trigger: str, intensity: int) -> dict[str, Any]:
    """Generate an immediate, personalized craving intervention."""
    prompt = f"""EMERGENCY: the user is having a craving RIGHT NOW and pressed the SOS button.

{_profile_context(profile)}

Current trigger: {trigger}
Craving intensity: {intensity}/10

Return ONLY a JSON object:
{{
  "urge_surf": "a 60-second guided urge-surfing script in second person, calm and specific to their habit",
  "distraction": "one concrete 5-minute distraction task doable right now given the trigger",
  "reframe": "one sentence that reframes this exact craving moment",
  "future_you": "2-3 sentences from their future self who broke the habit, referencing their motivation",
  "breathing": "a simple counted breathing pattern instruction (one line)"
}}
Make it feel personal and urgent-friendly, never clinical."""
    return chat_json([{"role": "user", "content": prompt}], temperature=0.9)


def reframe_thought(profile: db.Profile, thought: str) -> dict[str, Any]:
    """CBT-reframe an urge-thought and name the cognitive distortion."""
    prompt = f"""You are a CBT therapist. The user wrote down a thought that is pushing them
toward their habit.

{_profile_context(profile)}

Their thought: "{thought}"

Return ONLY a JSON object:
{{
  "distortion": "name of the main cognitive distortion at play (e.g. permission-giving,
    all-or-nothing, minimization)",
  "why": "1-2 sentences explaining how this distortion works against them",
  "reframe": "a believable, compassionate replacement thought in first person",
  "question": "one Socratic question they can ask themselves next time this thought appears"
}}"""
    return chat_json([{"role": "user", "content": prompt}], temperature=0.7)


def relapse_risk(profile: db.Profile) -> dict[str, Any]:
    """Analyze recent check-ins and estimate relapse risk with reasoning."""
    checkins = db.get_checkins(profile.id, limit=14)
    if not checkins:
        return {
            "level": "unknown",
            "reason": "No check-ins yet — log a few days first so I can analyze your patterns.",
            "action": "Do your first daily check-in today.",
        }
    prompt = f"""You are a relapse-prevention analyst. Study this user's recent pattern and
assess their current relapse risk.

{_profile_context(profile)}

Return ONLY a JSON object:
{{
  "level": "low" | "watch" | "high",
  "reason": "2-3 sentences citing SPECIFIC evidence from their check-ins (trends in mood,
    craving intensity, slips, notes)",
  "action": "one preventive action tailored to what you found",
  "pattern": "one insight about when/why they struggle most, if detectable"
}}"""
    result = chat_json([{"role": "user", "content": prompt}], temperature=0.4)
    if result.get("level") not in {"low", "watch", "high"}:
        result["level"] = "watch"
    return result


def weekly_insights(profile: db.Profile) -> dict[str, Any]:
    """Generate a weekly progress report from real check-in data."""
    checkins = db.get_checkins(profile.id, limit=14)
    prompt = f"""You are a habit-change data analyst. Write this user's progress insights from
their real check-in data.

{_profile_context(profile)}

Total check-ins available: {len(checkins)}

Return ONLY a JSON object:
{{
  "headline": "one encouraging headline summarizing the period",
  "wins": ["2-3 specific wins backed by the data"],
  "watch_outs": ["1-2 specific risks or patterns to watch"],
  "next_week_focus": "one clear focus recommendation",
  "trend": "improving" | "steady" | "struggling"
}}
Cite actual numbers from the data (streak, moods, craving levels)."""
    return chat_json([{"role": "user", "content": prompt}], temperature=0.5)


def coach_reply(profile: db.Profile, history: list[dict[str, str]], user_message: str) -> str:
    """Multi-turn coach chat grounded in the user's live profile and data."""
    if is_crisis(user_message):
        return CRISIS_MESSAGE
    system = f"""You are Unhooked's recovery coach: warm, practical, non-judgmental, and brief
(2-6 sentences unless asked for detail). You use CBT and motivational-interviewing techniques.
You know this user's real data:

{_profile_context(profile)}

Their recovery plan mantra: {profile.plan.get('mantra', 'not set')}
Ground every answer in their actual habit, triggers, and progress. Never invent data.
If they mention self-harm or suicide, tell them to contact AASRA 91-9820466726 (India) or
https://findahelpline.com immediately."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": user_message})
    return chat(messages, temperature=0.7)
