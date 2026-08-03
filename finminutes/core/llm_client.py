import random
import time
from abc import ABC, abstractmethod

from openai import APIError, APITimeoutError, AuthenticationError, RateLimitError, OpenAI

from finminutes.core.exceptions import (
    ConfigError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        ...

    @abstractmethod
    def test_connection(self) -> tuple:
        ...


class _OpenAIBasedClient(LLMClient):
    def __init__(self, api_key: str, base_url: str, model: str, max_retries: int = 3, timeout: int = 30, default_headers: dict | None = None):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        client_kwargs = {"api_key": api_key or "", "base_url": base_url, "timeout": timeout}
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self._client = OpenAI(**client_kwargs)

    def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        return self._call_with_retry(prompt, system=system, **kwargs)

    def test_connection(self):
        start = time.time()
        try:
            self._call_with_retry("Hello")
            latency_ms = (time.time() - start) * 1000
            return (True, "OK", round(latency_ms, 1))
        except Exception as e:
            return (False, str(e), None)

    def _call_with_retry(self, prompt: str, system: str = "", **kwargs) -> str:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
                content = response.choices[0].message.content
                return content or ""
            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = min(1.0 * (2**attempt) + random.uniform(0, 0.5), 60.0)
                    time.sleep(delay)
                    continue
                raise LLMRateLimitError(
                    f"Rate limit exceeded after {self.max_retries + 1} attempts. "
                    "Consider switching to a different provider or reducing request frequency."
                ) from e
            except AuthenticationError as e:
                raise LLMAuthenticationError(
                    f"Authentication failed for provider. Check your API key."
                ) from e
            except APITimeoutError as e:
                last_error = e
                if attempt < self.max_retries:
                    continue
                raise LLMTimeoutError(
                    f"Request timed out after {self.max_retries + 1} attempts."
                ) from e
            except APIError as e:
                code = getattr(e, "status_code", "?")
                msg = getattr(e, "message", str(e))
                raise LLMConnectionError(
                    f"API error (HTTP {code}): {msg}"
                ) from e
        raise last_error


class OpenRouterClient(_OpenAIBasedClient):
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", model: str = "google/gemini-2.0-flash-lite-preview-02-05", **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            default_headers={
                "HTTP-Referer": "https://finminutes.ai",
                "X-Title": "FinMinutes",
            },
            **kwargs,
        )


class DeepSeekClient(_OpenAIBasedClient):
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-chat", **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)


class OpenAIClient(_OpenAIBasedClient):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o", **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)


class OllamaClient(_OpenAIBasedClient):
    def __init__(self, api_key: str = "", base_url: str = "http://localhost:11434/v1", model: str = "qwen2.5:7b", **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)


class LLMFactory:
    @staticmethod
    def create(config: dict) -> LLMClient:
        provider = config.get("provider", "")
        kwargs = {
            "api_key": config.get("api_key", ""),
            "base_url": config.get("base_url", ""),
            "model": config.get("model", ""),
        }

        if provider == "openrouter":
            return OpenRouterClient(**kwargs)
        elif provider == "deepseek":
            return DeepSeekClient(**kwargs)
        elif provider == "openai":
            return OpenAIClient(**kwargs)
        elif provider == "ollama":
            return OllamaClient(**kwargs)
        else:
            raise ConfigError(f"Unknown LLM provider: '{provider}'")
