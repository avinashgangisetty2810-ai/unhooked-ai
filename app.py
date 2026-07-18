"""Unhooked — GenAI-powered coach for breaking bad habits and addictions.

Streamlit entrypoint. Every AI feature makes a live LLM call (Groq primary,
Gemini fallback); progress metrics are deterministic Python over real SQLite data.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from core import db, features
from core.llm import LLMError

st.set_page_config(page_title="Unhooked — AI Recovery Coach", page_icon="🔓", layout="wide")

PAGE_DASHBOARD = "🏠 Dashboard"
PAGE_CHECKIN = "📅 Daily Check-in"
PAGE_SOS = "🆘 Craving SOS"
PAGE_COACH = "💬 AI Coach"
PAGE_REFRAME = "🧠 Thought Reframe"
PAGE_PLAN = "🗺️ My Plan"
_PAGES = (PAGE_DASHBOARD, PAGE_CHECKIN, PAGE_SOS, PAGE_COACH, PAGE_REFRAME, PAGE_PLAN)

_HABIT_PRESETS = (
    "Excessive screen time / doomscrolling",
    "Smoking",
    "Vaping",
    "Alcohol",
    "Junk food / sugar",
    "Gaming",
    "Social media",
    "Other (describe below)",
)

_RISK_STYLE = {
    "low": ("🟢", "Low risk"),
    "watch": ("🟡", "Watch zone"),
    "high": ("🔴", "High risk"),
    "unknown": ("⚪", "Not enough data"),
}


def _ai_error(exc: LLMError) -> None:
    st.error(f"AI is temporarily unavailable — please try again in a moment. ({exc})")


def _goto(page: str) -> None:
    """Queue a navigation to another page and rerun (applied before the nav widget renders)."""
    st.session_state["_nav_target"] = page
    st.rerun()


# ---------------------------------------------------------------- onboarding


def _render_onboarding() -> None:
    st.title("🔓 Unhooked")
    st.subheader("Your AI coach for breaking bad habits — one day at a time.")
    st.markdown(
        "Tell us about the habit you want to break. The AI will build a **personalized "
        "4-week recovery plan** around your triggers and motivation."
    )
    with st.form("onboarding"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your name", max_chars=40)
            habit_choice = st.selectbox("Habit to break", _HABIT_PRESETS)
            habit_other = st.text_input("If other, describe it", max_chars=80)
            goal = st.radio("Your goal", ("Quit completely", "Reduce gradually"), horizontal=True)
        with col2:
            triggers = st.text_area(
                "What triggers it? (stress, boredom, late nights, certain people...)",
                max_chars=300,
                height=80,
            )
            motivation = st.text_area(
                "Why do you want to stop? (be honest — the AI uses this to coach you)",
                max_chars=300,
                height=80,
            )
            col_a, col_b = st.columns(2)
            with col_a:
                daily_cost = st.number_input("Money it costs per day (₹)", min_value=0.0, value=0.0, step=10.0)
            with col_b:
                daily_minutes = st.number_input("Time it eats per day (minutes)", min_value=0, value=60, step=15)
        submitted = st.form_submit_button("Build my recovery plan →", use_container_width=True)

    if submitted:
        habit = habit_other.strip() if habit_choice.startswith("Other") and habit_other.strip() else habit_choice
        if not name.strip():
            st.warning("Please enter your name.")
            return
        profile_id = db.create_profile(
            name=name.strip(),
            habit=habit,
            goal=goal,
            triggers=triggers.strip(),
            motivation=motivation.strip(),
            daily_cost=float(daily_cost),
            daily_minutes=int(daily_minutes),
        )
        profile = db.get_profile(profile_id)
        assert profile is not None
        with st.spinner("🧠 Your AI coach is designing your personalized plan..."):
            try:
                plan = features.generate_plan(profile)
                db.save_plan(profile_id, plan)
            except (LLMError, ValueError) as exc:
                st.session_state["plan_error"] = str(exc)
        st.session_state["profile_id"] = profile_id
        st.rerun()


# ------------------------------------------------------------------ sidebar


def _render_sidebar(profiles: list[db.Profile]) -> db.Profile | None:
    with st.sidebar:
        st.markdown("## 🔓 Unhooked")
        st.caption("GenAI recovery coach")
        if not profiles:
            return None
        labels = {p.id: f"{p.name} — {p.habit[:28]}" for p in profiles}
        default_id = st.session_state.get("profile_id", profiles[0].id)
        ids = list(labels.keys())
        index = ids.index(default_id) if default_id in ids else 0
        chosen = st.selectbox("Profile", ids, index=index, format_func=lambda i: labels[i])
        st.session_state["profile_id"] = chosen
        if st.button("➕ New profile", use_container_width=True):
            st.session_state["profile_id"] = None
            st.rerun()
        profile = db.get_profile(chosen)
        if profile:
            streak = db.current_streak(profile.id)
            st.metric("Clean streak", f"{streak} day{'s' if streak != 1 else ''} 🔥")
        st.divider()
        st.caption("⚡ Powered by Llama 3.3 on Groq · Gemini fallback")
        return profile


# ---------------------------------------------------------------- dashboard


def _cached_result(kind: str, profile_id: int) -> dict[str, Any] | None:
    """Return the freshest AI result: session state first, then the last stored event."""
    result: dict[str, Any] | None = st.session_state.get(kind)
    if result is None:
        past = db.get_events(profile_id, kind, limit=1)
        result = past[0] if past else None
    return result


def _run_ai_action(kind: str, profile: db.Profile, action: Any, spinner: str) -> None:
    """Run an AI feature, persist the result as an event, and cache it in session state."""
    with st.spinner(spinner):
        try:
            result = action(profile)
            db.log_event(profile.id, kind, result)
            st.session_state[kind] = result
        except LLMError as exc:
            _ai_error(exc)


def _render_metrics(profile: db.Profile, streak: int, clean_days: int) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clean streak", f"{streak} 🔥", help="Consecutive clean days ending today")
    col2.metric("Clean days (30d)", clean_days, help="Clean check-ins in the last 30 days")
    col3.metric("Money saved", f"₹{clean_days * profile.daily_cost:,.0f}", help="Clean days × daily habit cost")
    col4.metric("Time reclaimed", f"{clean_days * profile.daily_minutes / 60:.1f} h", help="Clean days × daily time cost")


def _render_trend_chart(checkins: list[dict[str, Any]]) -> None:
    st.subheader("📈 Craving & mood trend")
    if checkins:
        chart_data = {
            "craving (0-10)": [c["craving"] for c in reversed(checkins)],
            "mood (1-5)": [c["mood"] for c in reversed(checkins)],
        }
        st.line_chart(chart_data, height=260)
    else:
        st.info("Log your first daily check-in to start tracking trends.")
        if st.button("📅 Do my first check-in →"):
            _goto(PAGE_CHECKIN)


def _render_risk_radar(profile: db.Profile) -> None:
    st.subheader("🎯 Relapse Risk Radar")
    if st.button("Analyze my current risk", use_container_width=True):
        _run_ai_action("risk", profile, features.relapse_risk, "AI analyzing your recent patterns...")
    risk = _cached_result("risk", profile.id)
    if not risk:
        return
    icon, label = _RISK_STYLE.get(risk.get("level", "unknown"), _RISK_STYLE["unknown"])
    st.markdown(f"### {icon} {label}")
    st.write(risk.get("reason", ""))
    if risk.get("action"):
        st.success(f"**Do this:** {risk['action']}")
    if risk.get("pattern"):
        st.caption(f"Pattern spotted: {risk['pattern']}")
    if risk.get("level") in ("watch", "high") and st.button("🆘 Get an SOS intervention now →"):
        _goto(PAGE_SOS)


def _render_insights(profile: db.Profile) -> None:
    st.subheader("🧭 Weekly AI insights")
    if st.button("Generate insights from my data"):
        _run_ai_action("insights", profile, features.weekly_insights, "Crunching your real check-in data...")
    insights = _cached_result("insights", profile.id)
    if not insights:
        return
    st.markdown(f"#### {insights.get('headline', '')}")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Wins**")
        for win in insights.get("wins", []):
            st.markdown(f"- ✅ {win}")
    with col_b:
        st.markdown("**Watch out**")
        for item in insights.get("watch_outs", []):
            st.markdown(f"- ⚠️ {item}")
    if insights.get("next_week_focus"):
        st.info(f"**Next week's focus:** {insights['next_week_focus']}")


def _render_dashboard(profile: db.Profile) -> None:
    st.header(f"Welcome back, {profile.name} 👋")
    checkins = db.get_checkins(profile.id, limit=30)
    streak = db.current_streak(profile.id)
    clean_days = sum(1 for c in checkins if c["status"] == "clean")
    _render_metrics(profile, streak, clean_days)
    left, right = st.columns([3, 2])
    with left:
        _render_trend_chart(checkins)
    with right:
        _render_risk_radar(profile)
    st.divider()
    _render_insights(profile)


# ----------------------------------------------------------------- check-in


def _render_checkin(profile: db.Profile) -> None:
    st.header("📅 Daily check-in")
    st.caption("30 seconds a day. The AI adapts its coaching to what you log.")
    today_logged = any(
        c for c in db.get_checkins(profile.id, limit=1) if c["day"] == date.today().isoformat()
    )
    if today_logged:
        st.success("You already checked in today — resubmitting will update it.")
    with st.form("checkin"):
        status = st.radio(
            "How did today go?",
            ("clean", "slip"),
            format_func=lambda s: "💪 Stayed clean" if s == "clean" else "😔 Slipped",
            horizontal=True,
        )
        mood = st.slider("Mood", 1, 5, 3, help="1 = rough day, 5 = great day")
        craving = st.slider("Craving intensity", 0, 10, 3, help="0 = no craving at all, 10 = overwhelming urge")
        note = st.text_area("Anything worth noting? (optional)", max_chars=300, height=70)
        submitted = st.form_submit_button("Check in →", use_container_width=True)
    if submitted:
        _save_checkin(profile, status=status, mood=mood, craving=craving, note=note.strip())
    if st.session_state.get("checkin_followup"):
        st.markdown("**Rough day? These can help right now:**")
        col_sos, col_reframe = st.columns(2)
        if col_sos.button("🆘 Craving SOS →", use_container_width=True):
            st.session_state.pop("checkin_followup", None)
            _goto(PAGE_SOS)
        if col_reframe.button("🧠 Reframe the thought →", use_container_width=True):
            st.session_state.pop("checkin_followup", None)
            _goto(PAGE_REFRAME)
    _render_checkin_history(profile)


def _save_checkin(profile: db.Profile, *, status: str, mood: int, craving: int, note: str) -> None:
    """Persist today's check-in with an AI nudge and flag hard days for follow-up."""
    with st.spinner("Your coach is reading your check-in..."):
        try:
            nudge = features.checkin_nudge(profile, status=status, mood=mood, craving=craving, note=note)
        except LLMError as exc:
            _ai_error(exc)
            nudge = ""
    db.upsert_checkin(
        profile_id=profile.id,
        status=status,
        mood=mood,
        craving=craving,
        note=note,
        ai_response=nudge,
    )
    if nudge:
        st.chat_message("assistant").write(nudge)
    st.toast("Check-in saved ✅")
    st.session_state["checkin_followup"] = status == "slip" or craving >= 7


def _render_checkin_history(profile: db.Profile) -> None:
    history = db.get_checkins(profile.id, limit=7)
    if not history:
        return
    st.divider()
    st.subheader("Recent check-ins")
    for c in history:
        icon = "💪" if c["status"] == "clean" else "😔"
        with st.expander(f"{icon} {c['day']} — mood {c['mood']}/5, craving {c['craving']}/10"):
            if c["note"]:
                st.markdown(f"*Your note:* {c['note']}")
            if c["ai_response"]:
                st.markdown(f"**Coach said:** {c['ai_response']}")


# ---------------------------------------------------------------------- SOS


def _render_sos(profile: db.Profile) -> None:
    st.header("🆘 Craving SOS")
    st.caption("Craving hitting right now? Get an instant, personalized intervention.")
    with st.form("sos"):
        trigger = st.text_input(
            "What's triggering you right now?",
            placeholder="e.g. stressed after a work call, bored at home alone...",
            max_chars=200,
        )
        intensity = st.slider("How strong is the urge?", 1, 10, 7)
        pressed = st.form_submit_button("🆘 HELP ME NOW", use_container_width=True, type="primary")
    if pressed:
        with st.spinner("Generating your intervention..."):
            try:
                sos_result = features.sos_intervention(
                    profile, trigger=trigger.strip() or "unknown", intensity=intensity
                )
            except LLMError as exc:
                _ai_error(exc)
                return
        db.log_event(profile.id, "sos", {"trigger": trigger, "intensity": intensity, **sos_result})
        st.session_state["last_sos"] = sos_result
    sos: dict[str, Any] | None = st.session_state.get("last_sos")
    if sos:
        st.markdown("### 🌬️ First, breathe")
        st.info(sos.get("breathing", ""))
        st.markdown("### 🏄 Ride the urge (60 seconds)")
        st.write(sos.get("urge_surf", ""))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Do this instead")
            st.success(sos.get("distraction", ""))
        with col2:
            st.markdown("### 🔄 Reframe it")
            st.warning(sos.get("reframe", ""))
        st.markdown("### 💌 A message from future you")
        st.write(f"> {sos.get('future_you', '')}")
        st.caption("Cravings peak and pass in ~15–20 minutes. You've survived every one so far.")
        if st.button("💬 Talk it through with your coach →"):
            _goto(PAGE_COACH)


# ------------------------------------------------------------------ reframe


def _render_reframe(profile: db.Profile) -> None:
    st.header("🧠 Thought Reframe")
    st.caption("Write down the thought pushing you toward the habit. CBT-style AI takes it apart.")
    with st.form("reframe"):
        thought = st.text_area(
            "The thought in your head",
            placeholder='e.g. "Just one more episode won\'t hurt" or "I\'ve had a bad day, I deserve it"',
            max_chars=300,
            height=80,
        )
        submitted = st.form_submit_button("Reframe it →", use_container_width=True)
    if submitted and thought.strip():
        if features.is_crisis(thought):
            st.warning(features.CRISIS_MESSAGE)
            return
        with st.spinner("Analyzing the thought..."):
            try:
                result = features.reframe_thought(profile, thought.strip())
            except LLMError as exc:
                _ai_error(exc)
                return
        db.log_event(profile.id, "reframe", {"thought": thought.strip(), **result})
        st.markdown(f"**🔍 Distortion spotted:** `{result.get('distortion', '')}`")
        st.write(result.get("why", ""))
        st.success(f"**💬 Try this instead:** {result.get('reframe', '')}")
        st.info(f"**❓ Ask yourself:** {result.get('question', '')}")
    past = db.get_events(profile.id, "reframe", limit=5)
    if past:
        st.divider()
        st.subheader("Past reframes")
        for event in past:
            with st.expander(f'"{event.get("thought", "")[:60]}..."'):
                st.markdown(f"**Distortion:** {event.get('distortion', '')}")
                st.markdown(f"**Reframe:** {event.get('reframe', '')}")


# -------------------------------------------------------------------- coach


def _render_coach(profile: db.Profile) -> None:
    st.header("💬 AI Coach")
    st.caption("Available 24/7. Knows your habit, triggers, streak, and recent check-ins.")
    history = db.get_chat_messages(profile.id)
    for message in history:
        st.chat_message(message["role"]).write(message["content"])
    user_message = st.chat_input("Talk to your coach...")
    if user_message:
        st.chat_message("user").write(user_message)
        db.add_chat_message(profile.id, "user", user_message)
        with st.spinner("Coach is typing..."):
            try:
                reply = features.coach_reply(profile, history, user_message)
            except LLMError as exc:
                _ai_error(exc)
                return
        db.add_chat_message(profile.id, "assistant", reply)
        st.chat_message("assistant").write(reply)


# --------------------------------------------------------------------- plan


def _render_plan_body(plan: dict[str, Any]) -> None:
    st.markdown(f"### {plan.get('summary', '')}")
    if plan.get("mantra"):
        st.success(f"**Your mantra:** *{plan['mantra']}*")
    for week in plan.get("weeks", []):
        with st.expander(f"Week {week.get('week')} — {week.get('theme', '')}", expanded=week.get("week") == 1):
            st.markdown(f"**Target:** {week.get('target', '')}")
            for tactic in week.get("tactics", []):
                st.markdown(f"- {tactic}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧰 Coping toolkit")
        for tool in plan.get("coping_toolkit", []):
            st.markdown(f"- {tool}")
    with col2:
        st.subheader("🛡️ Trigger defenses")
        for item in plan.get("trigger_defenses", []):
            st.markdown(f"- **{item.get('trigger', '')}:** {item.get('defense', '')}")


def _render_plan(profile: db.Profile) -> None:
    st.header("🗺️ My Recovery Plan")
    if not profile.plan:
        st.info("No plan yet — generate one below.")
    else:
        _render_plan_body(profile.plan)
    if st.button("🔄 Regenerate plan with AI"):
        with st.spinner("Rebuilding your plan..."):
            try:
                new_plan = features.generate_plan(profile)
                db.save_plan(profile.id, new_plan)
                st.rerun()
            except (LLMError, ValueError) as exc:
                st.error(f"Could not regenerate the plan: {exc}")


# --------------------------------------------------------------------- main


def main() -> None:
    profiles = db.list_profiles()
    profile = _render_sidebar(profiles)
    if profile is None or st.session_state.get("profile_id") is None:
        _render_onboarding()
        return
    if "_nav_target" in st.session_state:
        st.session_state["nav"] = st.session_state.pop("_nav_target")
    page = st.sidebar.radio(
        "Navigate",
        _PAGES,
        key="nav",
        label_visibility="collapsed",
    )
    if st.session_state.pop("plan_error", None) and not profile.plan:
        st.warning(
            "⚠️ Your profile was created, but the AI plan could not be generated. "
            "Head to **🗺️ My Plan** and hit *Regenerate plan with AI*."
        )
    if page == PAGE_DASHBOARD:
        _render_dashboard(profile)
    elif page == PAGE_CHECKIN:
        _render_checkin(profile)
    elif page == PAGE_SOS:
        _render_sos(profile)
    elif page == PAGE_COACH:
        _render_coach(profile)
    elif page == PAGE_REFRAME:
        _render_reframe(profile)
    else:
        _render_plan(profile)


if __name__ == "__main__":
    main()
