import os
import tempfile

import pytest
import yaml

from finminutes.core.config_manager import ConfigManager
from finminutes.core.exceptions import ConfigError

_DEFAULT_CONFIG = {
    "active_llm": "openrouter_free",
    "llm_providers": {
        "openrouter_free": {
            "provider": "openrouter",
            "api_key": "${OPENROUTER_API_KEY}",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "google/gemini-2.0-flash-lite-preview-02-05",
        },
        "deepseek": {
            "provider": "deepseek",
            "api_key": "${DEEPSEEK_API_KEY}",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
        },
    },
    "logging": {
        "level": "INFO",
        "redact_transcript": True,
        "redact_api_key": True,
    },
}


@pytest.fixture
def config_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(_DEFAULT_CONFIG, f, allow_unicode=True)
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_load_default_config(config_file):
    cm = ConfigManager(config_file)
    assert cm.get_active_llm() == "openrouter_free"


def test_get_llm_config(config_file):
    cm = ConfigManager(config_file)
    cfg = cm.get_llm_config()
    assert cfg["provider"] == "openrouter"
    assert cfg["model"] == "google/gemini-2.0-flash-lite-preview-02-05"


def test_set_active_llm(config_file):
    cm = ConfigManager(config_file)
    cm.set_active_llm("deepseek")
    assert cm.get_active_llm() == "deepseek"
    cfg = cm.get_llm_config()
    assert cfg["provider"] == "deepseek"


def test_set_active_llm_unknown(config_file):
    cm = ConfigManager(config_file)
    with pytest.raises(ConfigError, match="Unknown LLM preset"):
        cm.set_active_llm("nonexistent")


def test_save_and_reload(config_file):
    cm = ConfigManager(config_file)
    cm.set_active_llm("deepseek")
    cm.save()

    cm2 = ConfigManager(config_file)
    assert cm2.get_active_llm() == "deepseek"


def test_env_var_substitution(config_file):
    os.environ["TEST_OPENROUTER_KEY"] = "sk-test-key-12345"
    _DEFAULT_CONFIG["llm_providers"]["openrouter_free"]["api_key"] = "${TEST_OPENROUTER_KEY}"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(_DEFAULT_CONFIG, f, allow_unicode=True)

    cm = ConfigManager(config_file)
    assert cm.get_llm_config()["api_key"] == "sk-test-key-12345"
    del os.environ["TEST_OPENROUTER_KEY"]


def test_env_var_missing_becomes_empty(config_file):
    cm = ConfigManager(config_file)
    assert cm.get_llm_config()["api_key"] == ""


def test_nonexistent_path_falls_back_to_package_default():
    fake_path = os.path.join(tempfile.gettempdir(), "_nonexistent_finminutes_config.yaml")
    if os.path.exists(fake_path):
        os.unlink(fake_path)
    cm = ConfigManager(fake_path)
    assert cm.get_active_llm() == "openrouter_free"


def test_config_property(config_file):
    cm = ConfigManager(config_file)
    assert isinstance(cm.config, dict)
    assert cm.config["active_llm"] == "openrouter_free"
