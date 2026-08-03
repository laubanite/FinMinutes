import httpx
from unittest.mock import MagicMock, patch

import pytest
from openai import APIError, APITimeoutError, AuthenticationError, RateLimitError

from finminutes.core.exceptions import (
    ConfigError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from finminutes.core.llm_client import (
    DeepSeekClient,
    LLMFactory,
    OllamaClient,
    OpenAIClient,
    OpenRouterClient,
)


def _make_mock_response(content: str = "mock reply"):
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _mock_http_response(status_code: int):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.request = MagicMock(spec=httpx.Request)
    resp.headers = {}
    return resp


@pytest.fixture(autouse=True)
def _patch_openai():
    with patch("finminutes.core.llm_client.OpenAI") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


class TestOpenAIBasedClient:
    def test_generate_returns_content(self, _patch_openai):
        _patch_openai.chat.completions.create.return_value = _make_mock_response("Hello!")
        client = OpenAIClient(api_key="sk-test", base_url="https://test.com/v1", model="gpt-4")
        result = client.generate("hi")
        assert result == "Hello!"

    def test_generate_empty_content(self, _patch_openai):
        _patch_openai.chat.completions.create.return_value = _make_mock_response(None)
        client = OpenAIClient(api_key="sk-test", base_url="https://test.com/v1", model="gpt-4")
        assert client.generate("hi") == ""

    def test_generate_passes_kwargs(self, _patch_openai):
        _patch_openai.chat.completions.create.return_value = _make_mock_response()
        client = OpenAIClient(api_key="sk-test", base_url="https://test.com/v1", model="gpt-4")
        client.generate("hi", temperature=0.5, max_tokens=100)
        _patch_openai.chat.completions.create.assert_called_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
        )


class TestConnection:
    def test_connection_success(self, _patch_openai):
        _patch_openai.chat.completions.create.return_value = _make_mock_response("Hello!")
        client = OpenAIClient(api_key="sk-test", base_url="https://test.com/v1", model="gpt-4")
        ok, msg, latency = client.test_connection()
        assert ok is True
        assert msg == "OK"
        assert isinstance(latency, (int, float))

    def test_connection_failure(self, _patch_openai):
        req = MagicMock(spec=httpx.Request)
        req.headers = {}
        _patch_openai.chat.completions.create.side_effect = APIError("boom", request=req, body=None)
        client = OpenAIClient(api_key="sk-test", base_url="https://test.com/v1", model="gpt-4")
        ok, msg, latency = client.test_connection()
        assert ok is False
        assert latency is None


class TestProviderSpecificClients:
    def test_openrouter_default_headers(self):
        with patch("finminutes.core.llm_client.OpenAI") as mock:
            OpenRouterClient(api_key="sk-or", base_url="https://openrouter.ai/api/v1", model="gemini-flash")
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs["default_headers"]["HTTP-Referer"] == "https://finminutes.ai"
            assert call_kwargs["default_headers"]["X-Title"] == "FinMinutes"

    def test_deepseek_base_url(self):
        with patch("finminutes.core.llm_client.OpenAI") as mock:
            DeepSeekClient(api_key="sk-ds")
            url = mock.call_args.kwargs["base_url"]
            assert "deepseek" in url

    def test_ollama_no_auth(self):
        with patch("finminutes.core.llm_client.OpenAI") as mock:
            OllamaClient()
            assert mock.call_args.kwargs["api_key"] == ""
            assert "localhost" in mock.call_args.kwargs["base_url"]


class TestRetry:
    def test_retry_on_rate_limit_then_succeed(self, _patch_openai):
        resp_429 = _mock_http_response(429)
        _patch_openai.chat.completions.create.side_effect = [
            RateLimitError("too fast", response=resp_429, body=None),
            RateLimitError("too fast", response=resp_429, body=None),
            _make_mock_response("finally"),
        ]
        client = OpenAIClient(api_key="sk-test", base_url="https://test.com/v1", model="gpt-4", max_retries=3)
        result = client.generate("hi")
        assert result == "finally"
        assert _patch_openai.chat.completions.create.call_count == 3

    def test_retry_exhausted_raises_rate_limit(self, _patch_openai):
        resp_429 = _mock_http_response(429)
        _patch_openai.chat.completions.create.side_effect = RateLimitError("too fast", response=resp_429, body=None)
        client = OpenAIClient(api_key="sk-test", base_url="https://test.com/v1", model="gpt-4", max_retries=2)
        with pytest.raises(LLMRateLimitError):
            client.generate("hi")
        assert _patch_openai.chat.completions.create.call_count == 3

    def test_authentication_error_raises(self, _patch_openai):
        resp_401 = _mock_http_response(401)
        _patch_openai.chat.completions.create.side_effect = AuthenticationError("bad key", response=resp_401, body=None)
        client = OpenAIClient(api_key="bad", base_url="https://test.com/v1", model="gpt-4")
        with pytest.raises(LLMAuthenticationError):
            client.generate("hi")

    def test_timeout_retry_then_raises(self, _patch_openai):
        req = MagicMock(spec=httpx.Request)
        req.headers = {}
        _patch_openai.chat.completions.create.side_effect = APITimeoutError(req)
        client = OpenAIClient(api_key="sk-test", base_url="https://test.com/v1", model="gpt-4", max_retries=1)
        with pytest.raises(LLMTimeoutError):
            client.generate("hi")
        assert _patch_openai.chat.completions.create.call_count == 2

    def test_api_error_raises_connection_error(self, _patch_openai):
        req = MagicMock(spec=httpx.Request)
        req.headers = {}
        _patch_openai.chat.completions.create.side_effect = APIError("server error", request=req, body=None)
        client = OpenAIClient(api_key="sk-test", base_url="https://test.com/v1", model="gpt-4")
        with pytest.raises(LLMConnectionError):
            client.generate("hi")


class TestLLMFactory:
    def test_create_openrouter(self):
        client = LLMFactory.create({"provider": "openrouter", "api_key": "sk-or", "base_url": "https://or.ai/v1", "model": "m"})
        assert isinstance(client, OpenRouterClient)

    def test_create_deepseek(self):
        client = LLMFactory.create({"provider": "deepseek", "api_key": "sk-ds", "base_url": "https://ds.ai/v1", "model": "m"})
        assert isinstance(client, DeepSeekClient)

    def test_create_openai(self):
        client = LLMFactory.create({"provider": "openai", "api_key": "sk-oa", "base_url": "https://oa.ai/v1", "model": "gpt-4"})
        assert isinstance(client, OpenAIClient)

    def test_create_ollama(self):
        client = LLMFactory.create({"provider": "ollama", "base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b"})
        assert isinstance(client, OllamaClient)

    def test_create_unknown_provider(self):
        with pytest.raises(ConfigError):
            LLMFactory.create({"provider": "unknown"})


class TestConfigManagerIntegration:
    def test_get_llm_client_from_config(self):
        from finminutes.core.config_manager import ConfigManager
        import os
        import tempfile
        import yaml

        config_data = {
            "active_llm": "openrouter_free",
            "llm_providers": {
                "openrouter_free": {
                    "provider": "openrouter",
                    "api_key": "${OPENROUTER_API_KEY}",
                    "base_url": "https://openrouter.ai/api/v1",
                    "model": "google/gemini-2.0-flash-lite-preview-02-05",
                },
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f, allow_unicode=True)
            path = f.name

        cm = ConfigManager(path)
        client = cm.get_llm_client()
        assert isinstance(client, OpenRouterClient)
        os.unlink(path)
