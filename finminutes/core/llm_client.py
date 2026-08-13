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
    def __init__(self, api_key: str, base_url: str, model: str, max_retries: int = 3, timeout: int = 120, default_headers: dict | None = None,
                 min_interval_ms: float = 0, max_tokens: int = 0, retry_callback=None):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.min_interval_ms = min_interval_ms or 0
        self.max_tokens = max_tokens or 0
        self.retry_callback = retry_callback
        self._last_call_at = 0.0
        client_kwargs = {"api_key": api_key or "", "base_url": base_url, "timeout": timeout}
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self._client = OpenAI(**client_kwargs)

    def generate(self, prompt: str, system: str = "", retry_callback=None, **kwargs) -> str:
        return self._call_with_retry(prompt, system=system, retry_callback=retry_callback, **kwargs)

    def test_connection(self):
        start = time.time()
        try:
            self._call_with_retry("Hello")
            latency_ms = (time.time() - start) * 1000
            return (True, "OK", round(latency_ms, 1))
        except Exception as e:
            return (False, str(e), None)

    def _pace(self):
        """调用间最小间隔节流：避免突发请求撞限速（如 OpenRouter 免费档 20 req/min）。"""
        if self.min_interval_ms <= 0:
            return
        interval = self.min_interval_ms / 1000.0
        now = time.monotonic()
        elapsed = now - self._last_call_at
        if self._last_call_at > 0 and elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call_at = time.monotonic()

    def _call_with_retry(self, prompt: str, system: str = "", retry_callback=None, **kwargs) -> str:
        if retry_callback is None:
            retry_callback = self.retry_callback
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                self._pace()
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                call_kwargs = dict(kwargs)
                # 显式 max_tokens：否则 API 用模型默认值（如 DeepSeek 4096 token），
                # 会卡死单次输出，大上下文模型的性能发挥不出来。
                if self.max_tokens > 0 and "max_tokens" not in call_kwargs:
                    call_kwargs["max_tokens"] = self.max_tokens
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **call_kwargs,
                )
                content = response.choices[0].message.content
                return content or ""
            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self._rate_limit_delay(e, attempt)
                    if retry_callback:
                        try:
                            retry_callback(attempt + 1, delay)
                        except Exception:
                            pass
                    time.sleep(delay)
                    continue
                raise LLMRateLimitError(
                    f"Rate limit exceeded after {self.max_retries + 1} attempts. "
                    "Consider switching to a different provider, reducing request frequency, "
                    "or configuring a fallback provider."
                ) from e
            except AuthenticationError as e:
                # 致命错误：不重试，直接给出可操作的引导
                raise LLMAuthenticationError(
                    f"认证失败：请检查 {self.model} 的 API Key 是否正确。"
                    f"可用 `finminutes config set-key <provider>` 配置，或运行 `finminutes init` 重新引导。"
                ) from e
            except APITimeoutError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = min(2.0 * (2**attempt), 30.0)
                    if retry_callback:
                        try:
                            retry_callback(attempt + 1, delay)
                        except Exception:
                            pass
                    time.sleep(delay)
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

    @staticmethod
    def _rate_limit_delay(e, attempt: int) -> float:
        """限流等待：优先尊重服务端 Retry-After 头，缺失时指数退避。"""
        resp = getattr(e, "response", None)
        if resp is not None:
            headers = getattr(resp, "headers", {})
            try:
                retry_after = headers.get("retry-after")
            except Exception:
                retry_after = None
            if retry_after:
                try:
                    return max(float(retry_after), 0.1) + 0.5
                except (ValueError, TypeError):
                    pass
        return min(1.0 * (2**attempt) + random.uniform(0, 0.5), 60.0)


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


class SiliconFlowClient(_OpenAIBasedClient):
    """硅基流动（SiliconFlow）OpenAI 兼容 API。免费模型：Qwen/Qwen2.5-7B-Instruct、
    THUDM/GLM-4-9B-0414、deepseek-ai/DeepSeek-V3 等（以平台实际免费清单为准）。"""

    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1", model: str = "deepseek-ai/DeepSeek-V3", **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)


class FallbackLLMClient(LLMClient):
    """主 provider 失败时自动降级到 fallback provider。

    用于 OpenRouter 免费档限流等场景：平时免费，偶发失败自动转付费/本地模型。
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient):
        self._primary = primary
        self._fallback = fallback

    def generate(self, prompt: str, system: str = "", retry_callback=None, **kwargs) -> str:
        try:
            return self._primary.generate(prompt, system=system, retry_callback=retry_callback, **kwargs)
        except Exception as primary_err:
            try:
                return self._fallback.generate(prompt, system=system, retry_callback=retry_callback, **kwargs)
            except Exception as fallback_err:
                raise LLMConnectionError(
                    f"主 provider 与降级 provider 均失败。主: {primary_err}；降级: {fallback_err}"
                ) from fallback_err

    def test_connection(self):
        ok, msg, latency = self._primary.test_connection()
        if ok:
            return (True, "OK", latency)
        return self._fallback.test_connection()


class LLMFactory:
    @staticmethod
    def create(config: dict) -> LLMClient:
        provider = config.get("provider", "")
        kwargs = {
            "api_key": config.get("api_key", ""),
            "base_url": config.get("base_url", ""),
            "model": config.get("model", ""),
            "min_interval_ms": config.get("min_request_interval_ms", 0) or 0,
            "max_tokens": config.get("max_tokens", 0) or 0,
        }

        if provider == "openrouter":
            client = OpenRouterClient(**kwargs)
        elif provider == "deepseek":
            client = DeepSeekClient(**kwargs)
        elif provider == "openai":
            client = OpenAIClient(**kwargs)
        elif provider == "ollama":
            client = OllamaClient(**kwargs)
        elif provider == "siliconflow":
            client = SiliconFlowClient(**kwargs)
        else:
            raise ConfigError(f"Unknown LLM provider: '{provider}'")

        # fallback 链：主 provider 限流/失败时自动降级。
        # 仅当 fallback 可用（本地 ollama，或已配置 api_key）时才启用，避免静默调用空 key 的 provider。
        fallback_name = config.get("fallback", "")
        if fallback_name:
            fb_cfg = config.get("_all_providers", {}).get(fallback_name)
            if fb_cfg and (fb_cfg.get("provider") == "ollama" or fb_cfg.get("api_key")):
                try:
                    return FallbackLLMClient(client, LLMFactory.create(fb_cfg))
                except Exception:
                    pass
        return client
