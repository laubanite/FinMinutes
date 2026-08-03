import os
import tempfile
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from finminutes.cli import main, _mask_key, _get_preset_defaults


class TestVersion:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestConfigShow:
    def test_show_runs(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "show"])
        assert result.exit_code == 0
        assert "当前激活配置" in result.output
        assert "[LLM]" in result.output
        assert "[ASR]" in result.output

    def test_mask_key_groq(self):
        assert _mask_key("gsk_xxxxxxxxxxxxxxxxxxxxxxxx") == "gsk_***xxx"

    def test_mask_key_openrouter(self):
        assert _mask_key("sk-or-abcdefghxyz") == "sk-or-***xyz"

    def test_mask_key_openai(self):
        assert _mask_key("sk-abcdefghxyz") == "sk-***xyz"

    def test_mask_key_short(self):
        assert _mask_key("abc") == "***"

    def test_mask_key_other(self):
        assert _mask_key("some-random-token-123") == "***"


class TestConfigListProviders:
    def test_list_providers_runs(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "list-providers"])
        assert result.exit_code == 0
        assert "openrouter_free" in result.output or "openai" in result.output or "deepseek" in result.output or "ollama" in result.output


class TestConfigSet:
    def test_set_active_llm(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "set", "active_llm", "openai"])
        assert result.exit_code == 0
        # Verify by checking show
        result2 = runner.invoke(main, ["config", "show"])
        assert "[LLM] openai" in result2.output
        # Reset
        runner.invoke(main, ["config", "set", "active_llm", "openrouter_free"])

    def test_set_active_llm_unknown(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "set", "active_llm", "nonexistent"])
        assert result.exit_code != 0

    def test_set_active_asr(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "set", "active_asr", "siliconflow"])
        assert result.exit_code == 0
        result2 = runner.invoke(main, ["config", "show"])
        assert "[ASR] siliconflow" in result2.output
        runner.invoke(main, ["config", "set", "active_asr", "groq"])

    def test_set_active_asr_unknown(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "set", "active_asr", "nonexistent"])
        assert result.exit_code != 0

    def test_set_unsupported_key(self):
        runner = CliRunner()
        result = runner.invoke(main, ["config", "set", "foo.bar", "baz"])
        assert result.exit_code != 0
        assert "仅支持设置 active_llm 和 active_asr" in result.output


class TestProcess:
    def test_process_no_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["process", "--transcript", "nonexistent.txt"])
        assert result.exit_code != 0
        assert "不存在" in result.output or "not found" in result.output

    def test_process_fast_mode(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("王总：今天讨论Q3业绩。\n李总：营收良好。")
            tmp = f.name
        review_path = os.path.splitext(tmp)[0] + "_校验稿.md"
        try:
            result = runner.invoke(main, ["process", "--transcript", tmp, "--mode", "fast"])
            assert result.exit_code == 0
            assert "校验稿已输出至" in result.output
            assert os.path.exists(review_path)
            with open(review_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "# 校验稿" in content
        finally:
            os.unlink(tmp)
            if os.path.exists(review_path):
                os.unlink(review_path)

    def test_process_fast_mode_custom_dir(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("王总：今天讨论Q3业绩。\n李总：营收良好。")
            tmp = f.name
        out_dir = tempfile.mkdtemp()
        review_path = os.path.join(out_dir, os.path.splitext(os.path.basename(tmp))[0] + "_校验稿.md")
        try:
            result = runner.invoke(main, [
                "process", "--transcript", tmp, "--mode", "fast",
                "--output", out_dir + os.sep,
            ])
            assert result.exit_code == 0
            assert "校验稿已输出至" in result.output
            assert os.path.exists(review_path)
        finally:
            os.unlink(tmp)
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_process_output_file(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("测试内容")
            tmp_in = f.name
        tmp_prefix = tempfile.mktemp(suffix="_test")
        review_path = f"{tmp_prefix}_校验稿.md"
        try:
            result = runner.invoke(main, [
                "process", "--transcript", tmp_in, "--mode", "fast", "--output", tmp_prefix,
            ])
            assert result.exit_code == 0
            assert "校验稿已输出至" in result.output
            assert os.path.exists(review_path)
            with open(review_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "# 校验稿" in content
        finally:
            os.unlink(tmp_in)
            if os.path.exists(review_path):
                os.unlink(review_path)

    def test_process_invalid_mode(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("test")
            tmp = f.name
        try:
            result = runner.invoke(main, ["process", "--transcript", tmp, "--mode", "turbo"])
            assert result.exit_code != 0
        finally:
            os.unlink(tmp)


class TestGetPresetDefaults:
    def test_openrouter_free(self):
        d = _get_preset_defaults("openrouter_free")
        assert d["provider"] == "openrouter"

    def test_deepseek(self):
        d = _get_preset_defaults("deepseek")
        assert d["provider"] == "deepseek"

    def test_openai(self):
        d = _get_preset_defaults("openai")
        assert d["provider"] == "openai"

    def test_ollama(self):
        d = _get_preset_defaults("ollama")
        assert d["provider"] == "ollama"
        assert d["api_key"] == ""

    def test_unknown(self):
        d = _get_preset_defaults("nonexistent")
        assert d == {}


class TestInit:
    def test_init_force_without_existing(self):
        runner = CliRunner()
        # Run init with force, but it needs interactive input which CliRunner can supply
        result = runner.invoke(main, ["init", "--force"], input="1\n\nn\n")
        # It will try to test connection and fail since no API key
        assert result.exit_code == 0

    def test_init_shows_welcome_or_exists(self):
        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        # With or without existing config, init should not crash
        assert result.exit_code == 0
        assert any(kw in result.output for kw in ("FinMinutes", "config", "配置已就绪"))


class TestHelp:
    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "process" in result.output
        assert "config" in result.output
        assert "transcribe" in result.output
        assert "glossary" in result.output
        assert "render" in result.output
        assert "serve" not in result.output

    def test_main_help_order(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        order = ["config", "init", "glossary", "transcribe", "process", "render"]
        positions = [result.output.index(cmd) for cmd in order]
        assert positions == sorted(positions)

    def test_serve_hidden_but_invokable(self):
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "启动" in result.output
