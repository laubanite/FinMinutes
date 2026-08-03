import os
import re

import yaml

from finminutes.core.exceptions import ConfigError
from finminutes.core.llm_client import LLMClient, LLMFactory

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


class ConfigManager:
    def __init__(self, path: str | None = None):
        self._path = path or self._default_config_path()
        self._config: dict = {}
        self.load()

    # ---- public API ----

    def load(self):
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            pkg_default = self._package_default_path()
            if os.path.exists(pkg_default):
                with open(pkg_default, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
            else:
                raise ConfigError(
                    f"Config file not found at {self._path} "
                    f"and no default package config available."
                )
        self._config = self._resolve_env_vars(raw)

    def save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._config, f, allow_unicode=True, default_flow_style=False)

    def get_active_llm(self) -> str:
        return self._config.get("active_llm", "openrouter_free")

    def set_active_llm(self, preset: str):
        if preset not in self._config.get("llm_providers", {}):
            raise ConfigError(f"Unknown LLM preset: {preset}")
        self._config["active_llm"] = preset

    def set_active_asr(self, preset: str):
        if preset not in self._config.get("asr_providers", {}):
            raise ConfigError(f"Unknown ASR preset: {preset}")
        self._config["active_asr"] = preset

    def get_llm_config(self) -> dict:
        preset = self.get_active_llm()
        providers = self._config.get("llm_providers", {})
        if preset not in providers:
            raise ConfigError(f"LLM preset '{preset}' not found in llm_providers")
        return providers[preset]

    def get_llm_client(self) -> LLMClient:
        return LLMFactory.create(self.get_llm_config())

    def get_active_asr(self) -> str:
        return self._config.get("active_asr", "groq")

    def get_asr_config(self, name: str = "") -> dict:
        name = name or self.get_active_asr()
        providers = self._config.get("asr_providers", {})
        if name not in providers:
            raise ConfigError(f"ASR provider '{name}' not found in asr_providers")
        return providers[name]

    def get_asr_client(self, name: str = ""):
        from finminutes.core.asr_client import GroqASRClient, SiliconFlowASRClient
        cfg = self.get_asr_config(name)
        provider = cfg.get("provider", "groq")
        if provider == "groq":
            return GroqASRClient(cfg)
        if provider == "siliconflow":
            return SiliconFlowASRClient(cfg)
        raise ConfigError(f"Unsupported ASR provider: {provider}")

    @property
    def config(self) -> dict:
        return self._config

    # ---- internal helpers ----

    def _resolve_env_vars(self, obj):
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        if isinstance(obj, str):
            return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), obj)
        return obj

    @staticmethod
    def _default_config_path() -> str:
        return os.path.expanduser("~/.finminutes/config.yaml")

    @staticmethod
    def _package_default_path() -> str:
        return os.path.join(os.path.dirname(__file__), "..", "config.yaml")
