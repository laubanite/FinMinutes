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
        # 以仓库内置默认配置为底，用户配置覆盖其上（深合并）。
        # 这样新增的配置项（如 min_request_interval_ms / fallback / rewrite_chunk_size）
        # 对已存在的用户配置文件同样生效。
        pkg_default = self._package_default()
        if not pkg_default:
            if not os.path.exists(self._path):
                raise ConfigError(
                    f"Config file not found at {self._path} "
                    f"and no default package config available."
                )
        user_raw = {}
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                user_raw = yaml.safe_load(f) or {}
        self._config = self._resolve_env_vars(self._deep_merge(pkg_default, user_raw))

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
        cfg = dict(self.get_llm_config())
        # 注入全量 providers，供 LLMFactory 解析 fallback 链
        cfg["_all_providers"] = self._config.get("llm_providers", {})
        return LLMFactory.create(cfg)

    def get_coverage_threshold(self) -> float:
        """内容完整度阈值（0-1），非法值回落到 0.90。"""
        raw = self._config.get("coverage_threshold", 0.90)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            try:
                v = float(raw)
            except (TypeError, ValueError):
                return 0.90
            if 0.0 <= v <= 1.0:
                return v
        return 0.90

    def get_rewrite_chunk_size(self) -> int:
        return self._provider_capability("rewrite_chunk_size", 4000)

    def get_qa_chunk_size(self) -> int:
        return self._provider_capability("qa_chunk_size", 8000)

    def _provider_capability(self, key: str, default: int) -> int:
        """按 active provider 的配置取值：provider 配置优先，顶层全局值作默认。

        使不同上下文窗口的模型各取合适的分片大小（如 deepseek 1M 上下文可大分片，
        OpenRouter 免费档小输出取小分片）。0/空/非数字一律回落到默认值。
        """
        global_raw = self._config.get(key)
        prov_raw = None
        try:
            prov_raw = self.get_llm_config().get(key)
        except ConfigError:
            pass
        raw = prov_raw if prov_raw is not None else global_raw
        if raw is None:
            return default
        if isinstance(raw, bool):
            return default
        if isinstance(raw, int) and raw > 0:
            return raw
        if isinstance(raw, str) and raw.isdigit():
            return int(raw)
        return default

    def set_by_path(self, path: str, value) -> tuple[bool, str]:
        """按点路径写入任意配置项。

        - 中间键必须已存在（避免误建）；新增提供商请用 `config add`。
        - 返回 (是否成功, 错误信息或空串)。
        """
        keys = [k for k in path.split(".") if k]
        if not keys:
            return False, "路径不能为空"
        node = self._config
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                return False, (
                    f"路径不存在: {path}（可用 `config show --all` 查看完整结构，"
                    f"或 `config add` 新增提供商）"
                )
            node = node[k]
        if not isinstance(node, dict):
            return False, f"路径不是字典: {path}"
        node[keys[-1]] = value
        return True, ""

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

    def _package_default(self) -> dict:
        pkg_default = self._package_default_path()
        if os.path.exists(pkg_default):
            with open(pkg_default, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """将 override 深合并到 base 之上；list 直接替换。"""
        result = dict(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = ConfigManager._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

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
