"""Unit tests for the LLM provider chain (core.llm) — all network calls mocked."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from core import llm


def _response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.json.return_value = payload
    if status >= 400:
        error = requests.HTTPError(response=mock)
        mock.raise_for_status.side_effect = error
    return mock


def _groq_ok(text: str) -> MagicMock:
    return _response({"choices": [{"message": {"content": text}}]})


def _gemini_ok(text: str) -> MagicMock:
    return _response({"candidates": [{"content": {"parts": [{"text": text}]}}]})


@pytest.fixture(autouse=True)
def _fake_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")


class TestReadSecret:
    def test_env_var_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_SECRET", "from-env")
        assert llm._read_secret("MY_SECRET") == "from-env"

    def test_missing_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NOPE_SECRET", raising=False)
        with patch("streamlit.secrets") as secrets:
            secrets.load_if_toml_exists.return_value = False
            assert llm._read_secret("NOPE_SECRET") == ""


class TestChat:
    def test_groq_success(self) -> None:
        with patch.object(llm.requests, "post", return_value=_groq_ok("hello")) as post:
            assert llm.chat([{"role": "user", "content": "hi"}]) == "hello"
        assert post.call_count == 1
        assert post.call_args.args[0] == llm._GROQ_URL

    def test_falls_back_to_gemini_on_groq_http_error(self) -> None:
        responses = [_response({}, status=500), _gemini_ok("from gemini")]
        with patch.object(llm.requests, "post", side_effect=responses):
            assert llm.chat([{"role": "user", "content": "hi"}]) == "from gemini"

    def test_retries_groq_on_429_then_succeeds(self) -> None:
        responses = [_response({}, status=429), _groq_ok("second try")]
        with (
            patch.object(llm.requests, "post", side_effect=responses),
            patch.object(llm.time, "sleep") as sleep,
        ):
            assert llm.chat([{"role": "user", "content": "hi"}]) == "second try"
        sleep.assert_called_once()

    def test_all_providers_fail_raises_llmerror(self) -> None:
        with (
            patch.object(llm.requests, "post", side_effect=requests.ConnectionError),
            pytest.raises(llm.LLMError, match="All AI providers failed"),
        ):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_missing_keys_skip_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "")
        with patch("streamlit.secrets") as secrets:
            secrets.load_if_toml_exists.return_value = False
            with patch.object(llm.requests, "post", return_value=_gemini_ok("gemini only")) as post:
                assert llm.chat([{"role": "user", "content": "hi"}]) == "gemini only"
        assert post.call_count == 1

    def test_json_mode_sets_response_format(self) -> None:
        with patch.object(llm.requests, "post", return_value=_groq_ok("{}")) as post:
            llm.chat([{"role": "user", "content": "hi"}], json_mode=True)
        payload = post.call_args.kwargs["json"]
        assert payload["response_format"] == {"type": "json_object"}

    def test_gemini_system_message_mapped(self) -> None:
        responses = [_response({}, status=500), _gemini_ok("ok")]
        with patch.object(llm.requests, "post", side_effect=responses) as post:
            llm.chat(
                [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "again"},
                ]
            )
        payload = post.call_args.kwargs["json"]
        assert payload["systemInstruction"]["parts"][0]["text"] == "be brief"
        assert [c["role"] for c in payload["contents"]] == ["user", "model", "user"]


class TestChatJson:
    def test_parses_plain_json(self) -> None:
        with patch.object(llm.requests, "post", return_value=_groq_ok('{"a": 1}')):
            assert llm.chat_json([{"role": "user", "content": "hi"}]) == {"a": 1}

    def test_parses_fenced_json(self) -> None:
        fenced = "```json\n" + json.dumps({"b": 2}) + "\n```"
        with patch.object(llm.requests, "post", return_value=_groq_ok(fenced)):
            assert llm.chat_json([{"role": "user", "content": "hi"}]) == {"b": 2}

    def test_malformed_json_raises(self) -> None:
        with patch.object(llm.requests, "post", return_value=_groq_ok("not json at all")):
            with pytest.raises(llm.LLMError, match="malformed JSON"):
                llm.chat_json([{"role": "user", "content": "hi"}])

    def test_non_object_json_raises(self) -> None:
        with patch.object(llm.requests, "post", return_value=_groq_ok("[1, 2]")):
            with pytest.raises(llm.LLMError, match="not an object"):
                llm.chat_json([{"role": "user", "content": "hi"}])
