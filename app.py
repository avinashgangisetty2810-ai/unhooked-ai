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


def _render_dashboard(profile: db.Profile) -> None:
    st.header(f"Welcome back, {profile.name} 👋")
    checkins = db.get_checkins(profile.id, limit=30)
    streak = db.current_streak(profile.id)
    clean_days = sum(1 for c in checkins if c["status"] == "clean")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clean streak", f"{streak} 🔥")
    col2.metric("Clean days (30d)", clean_days)
    col3.metric("Money saved", f"₹{clean_days * profile.daily_cost:,.0f}")
    col4.metric("Time reclaimed", f"{clean_days * profile.daily_minutes / 60:.1f} h")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("📈 Craving & mood trend")
        if checkins:
            chart_data = {
                "craving (0-10)": [c["craving"] for c in reversed(checkins)],
                "mood (1-5)": [c["mood"] for c in reversed(checkins)],
            }
            st.line_chart(chart_data, height=260)
        else:
            st.info("Log your first daily check-in to start tracking trends.")

    with right:
        st.subheader("🎯 Relapse Risk Radar")
        if st.button("Analyze my current risk", use_container_width=True):
            with st.spinner("AI analyzing your recent patterns..."):
                try:
                    risk = features.relapse_risk(profile)
                    db.log_event(profile.id, "risk", risk)
                    st.session_state["risk"] = risk
                except LLMError as exc:
                    _ai_error(exc)
        risk: dict[str, Any] | None = st.session_state.get("risk")
        if risk is None:
            past = db.get_events(profile.id, "risk", limit=1)
            risk = past[0] if past else None
        if risk:
            icon, label = _RISK_STYLE.get(risk.get("level", "unknown"), _RISK_STYLE["unknown"])
            st.markdown(f"### {icon} {label}")
            st.write(risk.get("reason", ""))
            if risk.get("action"):
                st.success(f"**Do this:** {risk['action']}")
            if risk.get("pattern"):
                st.caption(f"Pattern spotted: {risk['pattern']}")

    st.divider()
    st.subheader("🧭 Weekly AI insights")
    if st.button("Generate insights from my data"):
        with st.spinner("Crunching your real check-in data..."):
            try:
                insights = features.weekly_insights(profile)
                db.log_event(profile.id, "insights", insights)
                st.session_state["insights"] = insights
            except LLMError as exc:
                _ai_error(exc)
    insights: dict[str, Any] | None = st.session_state.get("insights")
    if insights is None:
        past = db.get_events(profile.id, "insights", limit=1)
        insights = past[0] if past else None
    if insights:
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
        craving = st.slider("Craving intensity", 0, 10, 3)
        note = st.text_area("Anything worth noting? (optional)", max_chars=300, height=70)
        submitted = st.form_submit_button("Check in →", use_container_width=True)
    if submitted:
        with st.spinner("Your coach is reading your check-in..."):
            try:
                nudge = features.checkin_nudge(
                    profile, status=status, mood=mood, craving=craving, note=note.strip()
                )
            except LLMError as exc:
                _ai_error(exc)
                nudge = ""
        db.upsert_checkin(
            profile_id=profile.id,
            status=status,
            mood=mood,
            craving=craving,
            note=note.strip(),
            ai_response=nudge,
        )
        if nudge:
            st.chat_message("assistant").write(nudge)
        st.toast("Check-in saved ✅")

    history = db.get_checkins(profile.id, limit=7)
    if history:
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
                sos = features.sos_intervention(
                    profile, trigger=trigger.strip() or "unknown", intensity=intensity
                )
            except LLMError as exc:
                _ai_error(exc)
                return
        db.log_event(profile.id, "sos", {"trigger": trigger, "intensity": intensity, **sos})
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


def _render_plan(profile: db.Profile) -> None:
    st.header("🗺️ My Recovery Plan")
    plan = profile.plan
    if not plan:
        st.info("No plan yet — generate one below.")
    else:
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
    page = st.sidebar.radio(
        "Navigate",
        ("🏠 Dashboard", "📅 Daily Check-in", "🆘 Craving SOS", "💬 AI Coach", "🧠 Thought Reframe", "🗺️ My Plan"),
        label_visibility="collapsed",
    )
    if st.session_state.pop("plan_error", None) and not profile.plan:
        st.warning(
            "⚠️ Your profile was created, but the AI plan could not be generated. "
            "Head to **🗺️ My Plan** and hit *Regenerate plan with AI*."
        )
    if page == "🏠 Dashboard":
        _render_dashboard(profile)
    elif page == "📅 Daily Check-in":
        _render_checkin(profile)
    elif page == "🆘 Craving SOS":
        _render_sos(profile)
    elif page == "💬 AI Coach":
        _render_coach(profile)
    elif page == "🧠 Thought Reframe":
        _render_reframe(profile)
    else:
        _render_plan(profile)


if __name__ == "__main__":
    main()
