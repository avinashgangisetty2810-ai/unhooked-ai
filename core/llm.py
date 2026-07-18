"""Provider-agnostic LLM client with a resilient fallback chain.

Primary provider: Groq (Llama 3.3 70B) — fast, generous free tier.
Fallback provider: Google Gemini — independent quota, used only when Groq fails.

All calls are real API calls; there is no canned-response mode. If every
provider fails, :class:`LLMError` is raised and the UI shows an honest
"AI unavailable" state.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any, Final

import requests

_GROQ_URL: Final[str] = "https://api.groq.com/openai/v1/chat/completions"
_GEMINI_URL: Final[str] = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_DEFAULT_GROQ_MODEL: Final[str] = "llama-3.3-70b-versatile"
_DEFAULT_GEMINI_MODEL: Final[str] = "gemini-2.5-flash"

_TIMEOUT_SECONDS: Final[int] = 60
_MAX_RETRIES: Final[int] = 2
_RETRY_BACKOFF_SECONDS: Final[float] = 1.5


class LLMError(Exception):
    """Raised when every configured provider fails to answer."""


def _read_secret(name: str) -> str:
    """Read a secret from the environment first, then Streamlit secrets."""
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        import streamlit as st

        if st.secrets.load_if_toml_exists():
            return str(st.secrets.get(name, ""))
    except Exception:  # pragma: no cover - streamlit not available in tests
        pass
    return ""


def _call_groq(messages: list[dict[str, str]], *, json_mode: bool, temperature: float) -> str:
    api_key = _read_secret("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not configured")
    model = _read_secret("GROQ_MODEL") or _DEFAULT_GROQ_MODEL
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    response = requests.post(
        _GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    return str(body["choices"][0]["message"]["content"])


def _call_gemini(messages: list[dict[str, str]], *, json_mode: bool, temperature: float) -> str:
    api_key = _read_secret("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not configured")
    model = _read_secret("GEMINI_MODEL") or _DEFAULT_GEMINI_MODEL

    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
        if m["role"] != "system"
    ]
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048},
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    response = requests.post(
        _GEMINI_URL.format(model=model),
        headers={"x-goog-api-key": api_key},
        json=payload,
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    body = response.json()
    return str(body["candidates"][0]["content"]["parts"][0]["text"])


def chat(
    messages: list[dict[str, str]],
    *,
    json_mode: bool = False,
    temperature: float = 0.7,
) -> str:
    """Send a chat request through the provider chain and return the text reply.

    Args:
        messages: OpenAI-style message dicts (``role``/``content``).
        json_mode: Force the model to emit a single JSON object.
        temperature: Sampling temperature.

    Returns:
        The assistant reply text (JSON string when ``json_mode`` is set).

    Raises:
        LLMError: If every provider in the chain fails.
    """
    errors: list[str] = []
    for provider_name, provider in (("groq", _call_groq), ("gemini", _call_gemini)):
        for attempt in range(_MAX_RETRIES):
            try:
                return provider(messages, json_mode=json_mode, temperature=temperature)
            except LLMError as exc:  # provider not configured — skip retries
                errors.append(f"{provider_name}: {exc}")
                break
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else 0
                errors.append(f"{provider_name}: HTTP {code}")
                if code == 429 and attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                break
            except (requests.RequestException, KeyError, IndexError) as exc:
                errors.append(f"{provider_name}: {type(exc).__name__}")
                break
    raise LLMError("All AI providers failed (" + "; ".join(errors) + ")")


def chat_stream(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
) -> Iterator[str]:
    """Yield the assistant reply incrementally for live typing in the UI.

    Streams token chunks from Groq (SSE). On any streaming failure before the
    first chunk, falls back to the full provider chain and yields the complete
    reply as a single chunk — so callers always get text or an :class:`LLMError`.

    Args:
        messages: OpenAI-style message dicts (``role``/``content``).
        temperature: Sampling temperature.

    Yields:
        Reply text fragments in generation order.

    Raises:
        LLMError: If streaming and every fallback provider fail.
    """
    api_key = _read_secret("GROQ_API_KEY")
    if api_key:
        model = _read_secret("GROQ_MODEL") or _DEFAULT_GROQ_MODEL
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
            "stream": True,
        }
        yielded = False
        try:
            with requests.post(
                _GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=_TIMEOUT_SECONDS,
                stream=True,
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data.strip() == "[DONE]":
                        break  # fall back below if the stream produced no text
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yielded = True
                        yield delta
                if yielded:
                    return
        except (requests.RequestException, KeyError, IndexError, ValueError):
            if yielded:  # partial reply already shown — don't duplicate via fallback
                return
    yield chat(messages, temperature=temperature)


def chat_json(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.6,
) -> dict[str, Any]:
    """Like :func:`chat` but parses and returns the reply as a JSON object.

    Raises:
        LLMError: If the reply is not valid JSON or all providers fail.
    """
    raw = chat(messages, json_mode=True, temperature=temperature)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
        data = json.loads(cleaned)
    except (ValueError, TypeError) as exc:
        raise LLMError(f"AI returned malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("AI returned JSON that is not an object")
    return data
