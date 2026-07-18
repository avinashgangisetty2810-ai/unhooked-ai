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

    def test_streamlit_secrets_used_when_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOML_SECRET", raising=False)
        with patch("streamlit.secrets") as secrets:
            secrets.load_if_toml_exists.return_value = True
            secrets.get.return_value = "from-toml"
            assert llm._read_secret("TOML_SECRET") == "from-toml"
        secrets.get.assert_called_once_with("TOML_SECRET", "")


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

    def test_gemini_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "")
        with patch("streamlit.secrets") as secrets:
            secrets.load_if_toml_exists.return_value = False
            with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
                llm._call_gemini([{"role": "user", "content": "hi"}], json_mode=False, temperature=0.5)

    def test_gemini_json_mode_sets_mime_type(self) -> None:
        with patch.object(llm.requests, "post", return_value=_gemini_ok("{}")) as post:
            llm._call_gemini([{"role": "user", "content": "hi"}], json_mode=True, temperature=0.5)
        payload = post.call_args.kwargs["json"]
        assert payload["generationConfig"]["responseMimeType"] == "application/json"


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


def _sse_line(content: str) -> bytes:
    return b"data: " + json.dumps({"choices": [{"delta": {"content": content}}]}).encode()


def _sse_response(lines: Any, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.iter_lines.return_value = lines
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestChatStream:
    def test_streams_chunks_until_done(self) -> None:
        lines = [_sse_line("Hel"), _sse_line("lo"), b"data: [DONE]", _sse_line("never")]
        with patch.object(llm.requests, "post", return_value=_sse_response(lines)) as post:
            chunks = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
        assert chunks == ["Hel", "lo"]
        assert post.call_args.kwargs["json"]["stream"] is True
        assert post.call_args.kwargs["stream"] is True

    def test_skips_blank_non_data_and_empty_delta_lines(self) -> None:
        lines = [b"", b": keep-alive", _sse_line(""), "data: [DONE]", _sse_line("never")]
        with patch.object(llm.requests, "post", return_value=_sse_response(lines)):
            with patch.object(llm, "chat", return_value="fallback") as chat:
                chunks = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
        # nothing streamed before [DONE] -> falls through to the blocking chain
        chat.assert_called_once()
        assert chunks == ["fallback"]

    def test_stream_ends_without_done_after_chunks(self) -> None:
        lines = [_sse_line("partial")]
        with patch.object(llm.requests, "post", return_value=_sse_response(lines)):
            with patch.object(llm, "chat") as chat:
                chunks = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
        chat.assert_not_called()
        assert chunks == ["partial"]

    def test_missing_key_falls_back_to_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "")
        with patch("streamlit.secrets") as secrets:
            secrets.load_if_toml_exists.return_value = False
            with patch.object(llm, "chat", return_value="full reply") as chat:
                chunks = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
        chat.assert_called_once()
        assert chunks == ["full reply"]

    def test_http_error_falls_back_to_chat(self) -> None:
        with patch.object(llm.requests, "post", return_value=_sse_response([], status=500)):
            with patch.object(llm, "chat", return_value="fallback") as chat:
                chunks = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
        chat.assert_called_once()
        assert chunks == ["fallback"]

    def test_error_after_first_chunk_stops_without_fallback(self) -> None:
        def _lines() -> Any:
            yield _sse_line("partial")
            raise requests.ConnectionError("dropped")

        with patch.object(llm.requests, "post", return_value=_sse_response(_lines())):
            with patch.object(llm, "chat") as chat:
                chunks = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
        chat.assert_not_called()  # never duplicate a partially shown reply
        assert chunks == ["partial"]

    def test_error_before_first_chunk_falls_back(self) -> None:
        def _lines() -> Any:
            raise requests.ConnectionError("dropped")
            yield  # pragma: no cover

        with patch.object(llm.requests, "post", return_value=_sse_response(_lines())):
            with patch.object(llm, "chat", return_value="fallback") as chat:
                chunks = list(llm.chat_stream([{"role": "user", "content": "hi"}]))
        chat.assert_called_once()
        assert chunks == ["fallback"]

    def test_json_mode_sets_response_format_and_propagates_to_fallback(self) -> None:
        with patch.object(llm.requests, "post", return_value=_sse_response([], status=500)) as post:
            with patch.object(llm, "chat", return_value="{}") as chat:
                list(llm.chat_stream([{"role": "user", "content": "hi"}], json_mode=True))
        assert post.call_args.kwargs["json"]["response_format"] == {"type": "json_object"}
        assert chat.call_args.kwargs["json_mode"] is True


class TestChatJsonStreaming:
    def test_on_delta_receives_accumulated_text_and_result_parses(self) -> None:
        lines = [_sse_line('{"a"'), _sse_line(": 1}"), b"data: [DONE]"]
        seen: list[str] = []
        with patch.object(llm.requests, "post", return_value=_sse_response(lines)):
            result = llm.chat_json([{"role": "user", "content": "hi"}], on_delta=seen.append)
        assert result == {"a": 1}
        assert seen == ['{"a"', '{"a": 1}']  # accumulated, in order

    def test_on_delta_malformed_stream_raises(self) -> None:
        lines = [_sse_line("not json"), b"data: [DONE]"]
        with patch.object(llm.requests, "post", return_value=_sse_response(lines)):
            with pytest.raises(llm.LLMError, match="malformed JSON"):
                llm.chat_json([{"role": "user", "content": "hi"}], on_delta=lambda _: None)
