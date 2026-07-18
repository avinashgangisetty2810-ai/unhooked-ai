# 🔓 Unhooked — AI Recovery Coach

**A GenAI-powered app that helps people break bad habits and addictions** — excessive screen time, smoking, vaping, junk food, gaming, and more — through personalized recovery plans, adaptive daily coaching, real-time craving interventions, and data-driven relapse-risk analysis.

Built for the **Hack2Skills GenAI Hackathon** — *Breaking Bad Habits & Addiction* challenge.

> 🌐 **Live demo:** https://unhooked.streamlit.app

---

## ✨ What it does

| Feature | GenAI in action |
|---|---|
| 🗺️ **Personalized 4-week recovery plan** | LLM designs a week-by-week plan around *your* habit, triggers, and motivation — with if-then trigger defenses, a coping toolkit, and a personal mantra |
| 📅 **Daily check-ins + adaptive nudges** | Log clean/slip days, mood, and cravings. The AI reads your actual entry and responds — celebrating streaks, normalizing slips without judgment |
| 🆘 **Craving SOS** | Hit the panic button mid-craving → instant 5-part intervention: box breathing, 60-second urge surfing, a concrete distraction, a reframe, and a letter from "future you" |
| 💬 **24/7 AI Coach** | Multi-turn chat grounded in your *real* data — streak, triggers, recent check-ins, plan mantra. CBT + motivational-interviewing style |
| 🧠 **Thought Reframe** | CBT-style analysis of the thought pushing you toward the habit — names the cognitive distortion, explains it, and gives a believable replacement thought |
| 🎯 **Relapse Risk Radar** | AI analyzes your recent check-in patterns and flags low/watch/high risk, citing specific evidence and a preventive action |
| 🧭 **Weekly AI insights** | Data-grounded progress report: wins, watch-outs, and next week's focus — with real numbers from your check-ins |
| 📈 **Progress tracking** | Streak counter, clean days, money saved, time reclaimed, craving & mood trend chart |
| 🚨 **Crisis guardrail** | Self-harm keyword detection short-circuits the LLM and surfaces real helplines (AASRA, iCall, findahelpline.com) |

**No mock data. No canned responses.** Every coaching message, plan, intervention, and insight is generated live by the LLM from the user's actual data. If the AI is unreachable, the app says so honestly instead of faking output.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│  app.py — Streamlit UI (6 pages + onboarding)│
├─────────────────────────────────────────────┤
│  core/features.py — AI features              │
│  plan · nudge · SOS · coach · reframe ·      │
│  risk radar · insights · crisis guardrail    │
├──────────────────────┬──────────────────────┤
│  core/llm.py         │  core/db.py          │
│  Provider chain:     │  SQLite persistence: │
│  Groq (Llama 3.3 70B)│  profiles, check-ins,│
│  → Gemini fallback   │  chats, events       │
│  retries + backoff   │  (parameterized SQL) │
└──────────────────────┴──────────────────────┘
```

- **LLM**: [Groq](https://groq.com) running `llama-3.3-70b-versatile` (JSON mode for structured outputs), with optional Google Gemini fallback. Automatic retry with backoff on rate limits; providers are tried in order and the app only reports failure if *all* fail.
- **Persistence**: SQLite (stdlib) — profiles, daily check-ins (upsert per day), chat history, and AI event logs survive restarts.
- **Privacy**: All data stays in a local SQLite file. API keys live in `st.secrets` / environment variables — never in code.

## 🚀 Run locally

```bash
git clone https://github.com/avinashgangisetty2810-ai/unhooked-ai.git
cd unhooked-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# add your key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml → GROQ_API_KEY = "gsk_..."

streamlit run app.py
```

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys).

### Configuration (secrets or env vars)

| Key | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Primary provider (Llama 3.3 70B on Groq) |
| `GEMINI_API_KEY` | optional | Fallback provider (Gemini 2.5 Flash) |
| `GROQ_MODEL` | optional | Override the Groq model |
| `GEMINI_MODEL` | optional | Override the Gemini model |

## ☁️ Deploy (Streamlit Community Cloud)

1. Fork/push this repo → [share.streamlit.io](https://share.streamlit.io) → *New app* → pick `app.py`.
2. In **App settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
3. Deploy. That's it — no other infrastructure needed.

## 🧪 Try this flow (2 minutes)

1. **Onboard** — name yourself, pick "Excessive screen time / doomscrolling", enter real triggers and motivation → AI builds your 4-week plan.
2. **Daily Check-in** — log a clean day with a note → watch the AI nudge reference what you wrote.
3. **Craving SOS** — type "stressed after a work call" at urge 7 → get the full intervention.
4. **AI Coach** — ask *"How long is my streak and what should I focus on tonight?"* → it answers from your actual data.
5. **Thought Reframe** — enter *"I had a stressful day, I deserve to scroll for an hour"* → it names the distortion (permission-giving) and reframes it.
6. **Dashboard** — run the Risk Radar and Weekly Insights on your real check-in history.

## 🛡️ Responsible AI

- Crisis-language detection bypasses the LLM entirely and shows verified helpline numbers.
- Coaching is explicitly non-judgmental — slips are treated as data, not failure.
- The app never invents progress data; every AI output cites the user's real logged history.
- Not a substitute for professional treatment — the app says so for serious addictions.

## ♿ Accessibility

- **Every input has a help tooltip** explaining what it does and how the AI uses it.
- **Charts have text alternatives** — the trend chart is accompanied by a plain-text summary of the latest values, so the data is available without vision.
- **Emoji are never the only signal** — every status icon (🟢/🟡/🔴 risk, 💪/😔 check-ins) is paired with a text label.
- **Full keyboard operability** — all interactions use native Streamlit widgets (forms, radios, sliders, buttons), which are keyboard-navigable and expose ARIA roles/labels out of the box.
- **High-contrast dark theme** — the palette (#e6edf7 text on #0b1120) exceeds WCAG AA contrast ratios.
- **Clear empty states** — first-time users are guided with explicit next-step buttons rather than blank screens.

## 📁 Project structure

```
unhooked-ai/
├── app.py                     # Streamlit UI — onboarding + 6 pages
├── core/
│   ├── llm.py                 # Provider chain (Groq → Gemini), retries, JSON mode
│   ├── db.py                  # SQLite layer — parameterized queries throughout
│   └── features.py            # All AI features + crisis guardrail
├── .streamlit/
│   ├── config.toml            # Dark theme
│   └── secrets.toml.example   # Key template (real secrets gitignored)
└── requirements.txt           # streamlit, requests, pyarrow — that's all
```
