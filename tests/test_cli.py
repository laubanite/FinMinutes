import os
import tempfile
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from finminutes.cli import main, _mask_key, _normalize_path


class TestVersion:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


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
        assert "openrouter_free" in result.output or "openai" in result.output or "deepseek" in result.output or "siliconflow" in result.output


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
        assert "路径不存在" in result.output


class TestConfigNewCommands:
    """V3 新增 config 命令（点路径 set / set-key ASR / add / list / show --all），隔离配置。"""

    def _invoke(self, monkeypatch, tmp_path, args, input=None):
        monkeypatch.setattr("finminutes.cli._CONFIG_PATH", str(tmp_path / "config.yaml"))
        return CliRunner().invoke(main, args, input=input)

    def _config(self, tmp_path):
        from finminutes.core.config_manager import ConfigManager
        return ConfigManager(str(tmp_path / "config.yaml"))

    def test_set_dotted_path(self, monkeypatch, tmp_path):
        r = self._invoke(monkeypatch, tmp_path, ["config", "set", "asr_providers.groq.chunk_duration_minutes", "15"])
        assert r.exit_code == 0
        assert self._config(tmp_path).config["asr_providers"]["groq"]["chunk_duration_minutes"] == 15

    def test_set_dotted_path_missing(self, monkeypatch, tmp_path):
        r = self._invoke(monkeypatch, tmp_path, ["config", "set", "asr_providers.nope.x", "1"])
        assert r.exit_code != 0
        assert "路径不存在" in r.output

    def test_set_active_backward_compat(self, monkeypatch, tmp_path):
        r = self._invoke(monkeypatch, tmp_path, ["config", "set", "active_llm", "openai"])
        assert r.exit_code == 0
        assert self._config(tmp_path).get_active_llm() == "openai"

    def test_set_key_asr(self, monkeypatch, tmp_path):
        r = self._invoke(monkeypatch, tmp_path, ["config", "set-key", "groq"], input="gsk_test123\n")
        assert r.exit_code == 0
        assert self._config(tmp_path).config["asr_providers"]["groq"]["api_key"] == "gsk_test123"

    def test_set_key_llm(self, monkeypatch, tmp_path):
        r = self._invoke(monkeypatch, tmp_path, ["config", "set-key", "deepseek"], input="sk-ds999\n")
        assert r.exit_code == 0
        assert self._config(tmp_path).config["llm_providers"]["deepseek"]["api_key"] == "sk-ds999"

    def test_set_key_unknown(self, monkeypatch, tmp_path):
        r = self._invoke(monkeypatch, tmp_path, ["config", "set-key", "nope"])
        assert r.exit_code != 0
        assert "未知的提供商" in r.output

    def test_config_list_shows_both(self, monkeypatch, tmp_path):
        r = self._invoke(monkeypatch, tmp_path, ["config", "list"])
        assert r.exit_code == 0
        assert "[LLM 提供商]" in r.output
        assert "[ASR 提供商]" in r.output
        assert "groq" in r.output

    def test_config_list_providers_alias(self, monkeypatch, tmp_path):
        r = self._invoke(monkeypatch, tmp_path, ["config", "list-providers"])
        assert r.exit_code == 0
        assert "[LLM 提供商]" in r.output

    def test_config_show_all(self, monkeypatch, tmp_path):
        r = self._invoke(monkeypatch, tmp_path, ["config", "show", "--all"])
        assert r.exit_code == 0
        assert "全部 LLM 提供商" in r.output
        assert "全部 ASR 提供商" in r.output

    def test_config_add_asr(self, monkeypatch, tmp_path):
        inputs = "siliconflow\nsk-new1\nmy-model\n\n100\n20\n10\n5\ny\n"
        r = self._invoke(monkeypatch, tmp_path, ["config", "add", "asr", "myasr"], input=inputs)
        assert r.exit_code == 0
        cfg = self._config(tmp_path)
        assert cfg.config["asr_providers"]["myasr"]["provider"] == "siliconflow"
        assert cfg.get_active_asr() == "myasr"

    def test_config_add_llm_not_active(self, monkeypatch, tmp_path):
        # 自定义 LLM：API 格式(openai 兼容) → key → base_url(默认) → 模型名 → 不设为激活
        inputs = "openai\nsk-llm1\n\nmy-model\nn\n"
        r = self._invoke(monkeypatch, tmp_path, ["config", "add", "llm", "myllm"], input=inputs)
        assert r.exit_code == 0, r.output
        cfg = self._config(tmp_path)
        assert cfg.config["llm_providers"]["myllm"]["provider"] == "openai"
        assert cfg.get_active_llm() == "openrouter_free"  # 未设为激活

    def test_config_add_llm_anthropic_format(self, monkeypatch, tmp_path):
        """自定义 LLM 选 Anthropic 格式 → provider=anthropic。"""
        inputs = "anthropic\nsk-ant1\n\nmy-model\ny\n"
        r = self._invoke(monkeypatch, tmp_path, ["config", "add", "llm", "myclaude"], input=inputs)
        assert r.exit_code == 0, r.output
        cfg = self._config(tmp_path)
        assert cfg.config["llm_providers"]["myclaude"]["provider"] == "anthropic"
        assert cfg.config["llm_providers"]["myclaude"]["base_url"].startswith("https://api.anthropic.com")
        assert cfg.get_active_llm() == "myclaude"

    def test_config_add_preset_only_key(self, monkeypatch, tmp_path):
        """内置预设名（groq）→ 预设档：只问 API Key，不重复问全参数。"""
        r = self._invoke(monkeypatch, tmp_path, ["config", "add", "asr", "groq"], input="gsk_new_key\ny\n")
        assert r.exit_code == 0, r.output
        cfg = self._config(tmp_path)
        assert cfg.config["asr_providers"]["groq"]["api_key"] == "gsk_new_key"
        assert "模型名" not in r.output  # 未走自定义档

    def test_config_add_duplicate_user_created(self, monkeypatch, tmp_path):
        """用户自建且已存在的名称 → 报错「已存在」。"""
        self._invoke(monkeypatch, tmp_path, ["config", "add", "llm", "myllm"],
                     input="openai\nsk-llm1\n\nmy-model\nn\n")
        r = self._invoke(monkeypatch, tmp_path, ["config", "add", "llm", "myllm"])
        assert r.exit_code != 0
        assert "已存在" in r.output

    def test_config_remove_user_provider(self, monkeypatch, tmp_path):
        """删除用户自建 provider（清残留，如早期固化进系统盘配置的 ollama）。"""
        self._invoke(monkeypatch, tmp_path, ["config", "add", "llm", "myllm"],
                     input="openai\nsk-llm1\n\nmy-model\nn\n")
        r = self._invoke(monkeypatch, tmp_path, ["config", "remove", "llm", "myllm"], input="y\n")
        assert r.exit_code == 0, r.output
        cfg = self._config(tmp_path)
        assert "myllm" not in cfg.config["llm_providers"]

    def test_config_remove_builtin_refused(self, monkeypatch, tmp_path):
        """内置预设（出厂 config 定义）删除后加载会被默认恢复 → 拒绝。"""
        r = self._invoke(monkeypatch, tmp_path, ["config", "remove", "llm", "deepseek"], input="y\n")
        assert r.exit_code != 0
        assert "内置预设" in r.output

    def test_config_remove_active_refused(self, monkeypatch, tmp_path):
        """不能删当前激活的 provider。"""
        self._invoke(monkeypatch, tmp_path, ["config", "add", "asr", "myasr"],
                     input="groq\nsk-a1\nmy-asr-model\n\n5\n10\n3\n2\ny\n")
        r = self._invoke(monkeypatch, tmp_path, ["config", "remove", "asr", "myasr"], input="y\n")
        assert r.exit_code != 0
        assert "当前激活" in r.output


class TestProcess:
    def test_process_no_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["process", "--transcript", "nonexistent.txt"])
        assert result.exit_code != 0
        assert "不存在" in result.output or "not found" in result.output

    def test_process_fast_mode_outputs_cleaned_text(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("王总：今天讨论Q3业绩。\n李总：营收良好。")
            tmp = f.name
        clean_path = os.path.splitext(tmp)[0] + "_清洗稿.txt"
        try:
            result = runner.invoke(main, ["process", "--transcript", tmp, "--mode", "fast"])
            assert result.exit_code == 0
            assert "清洗稿已输出至" in result.output
            assert os.path.exists(clean_path)
            with open(clean_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Q3业绩" in content
            assert "校验稿已输出至" not in result.output  # fast 不再产出空校验稿
        finally:
            os.unlink(tmp)
            if os.path.exists(clean_path):
                os.unlink(clean_path)

    def test_process_fast_mode_custom_dir(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("王总：今天讨论Q3业绩。\n李总：营收良好。")
            tmp = f.name
        out_dir = tempfile.mkdtemp()
        clean_path = os.path.join(out_dir, os.path.splitext(os.path.basename(tmp))[0] + "_清洗稿.txt")
        try:
            result = runner.invoke(main, [
                "process", "--transcript", tmp, "--mode", "fast",
                "--output", out_dir + os.sep,
            ])
            assert result.exit_code == 0
            assert "清洗稿已输出至" in result.output
            assert os.path.exists(clean_path)
        finally:
            os.unlink(tmp)
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_process_fast_output_prefix(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("测试内容")
            tmp_in = f.name
        tmp_prefix = tempfile.mktemp(suffix="_test")
        clean_path = f"{tmp_prefix}_清洗稿.txt"
        try:
            result = runner.invoke(main, [
                "process", "--transcript", tmp_in, "--mode", "fast", "--output", tmp_prefix,
            ])
            assert result.exit_code == 0
            assert "清洗稿已输出至" in result.output
            assert os.path.exists(clean_path)
            with open(clean_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert content.strip()  # 非空清洗稿
        finally:
            os.unlink(tmp_in)
            if os.path.exists(clean_path):
                os.unlink(clean_path)

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


class TestProviderLabel:
    def test_free_with_model_and_rate_limit(self):
        from finminutes.cli import _provider_label
        assert _provider_label({"free": True, "model": "gemini-x", "rate_limit": "20 req/min"}) == "免费 · gemini-x · 20 req/min"

    def test_paid_shows_model_only(self):
        from finminutes.cli import _provider_label
        assert _provider_label({"provider": "deepseek", "model": "deepseek-v4-flash"}) == "deepseek-v4-flash"

    def test_empty_no_label(self):
        from finminutes.cli import _provider_label
        assert _provider_label({"provider": "openai"}) == ""


class TestNormalizePath:
    def test_backslash_converted(self):
        assert _normalize_path("E:\\foo\\bar.txt") == "E:/foo/bar.txt"

    def test_forward_slash_unchanged(self):
        assert _normalize_path("E:/foo/bar.txt") == "E:/foo/bar.txt"

    def test_empty_unchanged(self):
        assert _normalize_path("") == ""


class TestInit:
    """init 是交互式向导：隔离配置路径并 mock 连接测试，避免依赖真实网络/配额。"""

    @staticmethod
    def _mock_connections(monkeypatch, llm_ok=True, llm_msg="ok", asr_ok=True, asr_msg="ok"):
        llm_client = MagicMock()
        llm_client.test_connection.return_value = (llm_ok, llm_msg, 120 if llm_ok else 0)
        fake = MagicMock()
        fake.create.return_value = llm_client
        monkeypatch.setattr("finminutes.cli.LLMFactory", fake)

        asr_client = MagicMock()
        asr_client.test_connection.return_value = (asr_ok, asr_msg, 80 if asr_ok else 0)
        monkeypatch.setattr("finminutes.cli._build_asr_client", lambda cfg: asr_client)

    @staticmethod
    def _write_config(path, api_key="sk-fake-key-123"):
        import yaml
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "active_llm": "openrouter_free",
                "active_asr": "groq",
                "llm_providers": {"openrouter_free": {"provider": "openrouter", "api_key": api_key}},
                "asr_providers": {"groq": {"provider": "groq", "api_key": api_key}},
            }, f, allow_unicode=True, default_flow_style=False)

    def test_init_ready_when_complete(self, monkeypatch, tmp_path):
        """LLM + ASR 均已配置 Key 且 LLM 连接正常 → 打印「配置已就绪」直接返回。"""
        cfg_path = str(tmp_path / "config.yaml")
        monkeypatch.setattr("finminutes.cli._CONFIG_PATH", cfg_path)
        self._write_config(cfg_path)
        self._mock_connections(monkeypatch, llm_ok=True, asr_ok=True)
        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0
        assert "配置已就绪" in result.output

    def test_init_force_without_existing_declines(self, monkeypatch, tmp_path):
        """--force 无现有配置：选 LLM 预设、key 留空、连接失败后选择不保存，正常退出。"""
        monkeypatch.setattr("finminutes.cli._CONFIG_PATH", str(tmp_path / "config.yaml"))
        self._mock_connections(monkeypatch, llm_ok=False, llm_msg="no api key", asr_ok=True)
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--force"], input="1\n\nn\n")
        assert result.exit_code == 0

    def test_init_full_flow_writes_both_providers(self, monkeypatch, tmp_path):
        """完整引导写最小配置：只更新选中的 LLM/ASR provider 与 active，保留默认参数由加载时合并。"""
        cfg_path = str(tmp_path / "config.yaml")
        monkeypatch.setattr("finminutes.cli._CONFIG_PATH", cfg_path)
        self._write_config(cfg_path)
        self._mock_connections(monkeypatch, llm_ok=True, asr_ok=True)
        runner = CliRunner()
        # --force 强制走完整引导；已有 key 自动保留，两个选择都用默认（LLM=1, ASR=1）
        result = runner.invoke(main, ["init", "--force"], input="1\n1\n")
        assert result.exit_code == 0, result.output
        assert "配置文件已保存至" in result.output
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["active_llm"] == "openrouter_free"
        assert data["active_asr"] == "groq"
        assert "配置分层提醒" in result.output


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

    def test_export_prompt_in_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "export-prompt" in result.output
        assert "import-result" in result.output


class TestExportPrompt:
    def test_export_prompt_generates_prompt_file(self, tmp_path):
        transcript = tmp_path / "meeting_转录稿.txt"
        transcript.write_text("你好，我们公司营收1.5亿元。", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["export-prompt", "-t", str(transcript)])
        assert result.exit_code == 0, result.output
        prompt_file = tmp_path / "meeting_转录稿_prompt.txt"
        assert prompt_file.exists()
        content = prompt_file.read_text(encoding="utf-8")
        assert "转录全文" in content
        assert "营收1.5亿元" in content
        assert "qa_pairs" in content  # 默认 qa 格式

    def test_export_prompt_both_format(self, tmp_path):
        transcript = tmp_path / "m.txt"
        transcript.write_text("你好，我们公司营收1.5亿元。", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["export-prompt", "-t", str(transcript), "-f", "both"])
        assert result.exit_code == 0, result.output
        content = (tmp_path / "m_prompt.txt").read_text(encoding="utf-8")
        assert "sections" in content
        assert "qa_pairs" in content

    def test_export_prompt_with_glossary_path(self, tmp_path):
        """-g 传文件路径：术语表按路径加载并进入纠错规则。"""
        import yaml
        transcript = tmp_path / "m.txt"
        transcript.write_text("公司营收1.5亿元。", encoding="utf-8")
        gl_path = tmp_path / "custom.yaml"
        gl_path.write_text(
            yaml.safe_dump(
                {"industry": "test", "terms": [{"term": "ABC", "corrections": ["A B C"]}]},
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["export-prompt", "-t", str(transcript), "-g", str(gl_path)])
        assert result.exit_code == 0, result.output
        content = (tmp_path / "m_prompt.txt").read_text(encoding="utf-8")
        assert "应纠正为「ABC」" in content

    def test_export_prompt_with_bundled_tag(self, tmp_path):
        """-g 传内置标签名（semiconductor）：tag 兜底仍可用。"""
        transcript = tmp_path / "m.txt"
        transcript.write_text("HBM是先进封装的关键技术。", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["export-prompt", "-t", str(transcript), "-g", "semiconductor"])
        assert result.exit_code == 0, result.output
        content = (tmp_path / "m_prompt.txt").read_text(encoding="utf-8")
        assert "应纠正为「" in content


class TestImportResult:
    def test_import_result_parses_json(self, tmp_path):
        result_file = tmp_path / "result.txt"
        result_file.write_text(
            '{"qa_pairs": [{"question": "营收多少？", "answer": "1.5亿元", "asker": "分析师"}]}',
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["import-result", "-r", str(result_file)])
        assert result.exit_code == 0, result.output
        review = tmp_path / "result_校验稿.md"
        assert review.exists()
        content = review.read_text(encoding="utf-8")
        assert "营收多少" in content
        assert "1.5亿元" in content

    def test_import_result_parses_sections(self, tmp_path):
        result_file = tmp_path / "speech_result.txt"
        result_file.write_text(
            '{"sections": [{"title": "公司概况", "content": "营收1.5亿元"}]}',
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["import-result", "-r", str(result_file)])
        assert result.exit_code == 0, result.output
        review = tmp_path / "speech_result_校验稿.md"
        assert review.exists()
        content = review.read_text(encoding="utf-8")
        assert "公司概况" in content
        assert "营收1.5亿元" in content

    def test_import_result_falls_back_to_content(self, tmp_path):
        """无法解析为 JSON 时，整段作为内容兜底，仍出校验稿（信息零丢失）。"""
        result_file = tmp_path / "bad.txt"
        result_file.write_text("完全不是 JSON 的原始文本，但包含关键数字 2.3 亿。", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["import-result", "-r", str(result_file)])
        assert result.exit_code == 0, result.output
        review = tmp_path / "bad_校验稿.md"
        assert review.exists()
        content = review.read_text(encoding="utf-8")
        assert "2.3 亿" in content

    def test_import_result_warns_without_source(self, tmp_path):
        """未提供 -s 时提示无法校验完整度，但不写调试 prompt。"""
        result_file = tmp_path / "r.txt"
        result_file.write_text('{"qa_pairs": [{"question": "营收多少？", "answer": "1.5亿元"}]}', encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(main, ["import-result", "-r", str(result_file)])
        assert result.exit_code == 0, result.output
        assert "无法校验内容完整度" in result.output
        assert "[完整度]" not in result.output
        assert not (tmp_path / "r_调试prompt.txt").exists()

    def test_import_result_reports_coverage_and_debug_prompt(self, tmp_path):
        """带 -s 时输出完整度；低于阈值自动写调试 prompt。"""
        transcript = tmp_path / "src.txt"
        transcript.write_text(
            "公司营收1.8亿元，毛利率30%，计划2026年上市，员工约5000人，"
            "计划扩产至8000万颗月产能，本轮融资1.2亿元，累计出货14亿颗，"
            "射频前端、滤波器、卫星通信模组，5G工业专网，海外物联网AI穿戴。",
            encoding="utf-8",
        )
        result_file = tmp_path / "r.txt"
        # 成果稿只覆盖了营收一个数字 → 完整度远低于 90%
        result_file.write_text(
            '{"qa_pairs": [{"question": "营收多少？", "answer": "1.8亿元"}]}',
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["import-result", "-r", str(result_file), "-s", str(transcript)])
        assert result.exit_code == 0, result.output
        assert "[完整度]" in result.output
        review = tmp_path / "r_校验稿.md"
        assert review.exists()
        debug = tmp_path / "r_调试prompt.txt"
        assert debug.exists(), "低于阈值应自动写调试 prompt"
        content = debug.read_text(encoding="utf-8")
        assert "未覆盖转录片段" in content
        assert "1.8亿元" in content  # 当前已提取内容带入 prompt

    def test_import_result_full_coverage_no_debug_prompt(self, tmp_path):
        """转录被完整覆盖时，不写调试 prompt。"""
        transcript = tmp_path / "src.txt"
        transcript.write_text("公司营收1.8亿元。", encoding="utf-8")
        result_file = tmp_path / "r.txt"
        result_file.write_text(
            '{"qa_pairs": [{"question": "营收多少？", "answer": "1.8亿元"}]}',
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["import-result", "-r", str(result_file), "-s", str(transcript)])
        assert result.exit_code == 0, result.output
        assert not (tmp_path / "r_调试prompt.txt").exists()


class TestRenderFromJson:
    def _run(self, runner, args):
        with (
            patch("finminutes.cli._get_config", return_value=MagicMock()),
            patch("finminutes.cli._ensure_llm_config", return_value=True),
        ):
            llm = MagicMock()
            gen = MagicMock()
            gen.generate.return_value = "## 总结\n\n成品内容。"
            with patch("finminutes.cli.FormalGenerator", return_value=gen):
                return runner.invoke(main, args)

    def test_render_from_json(self, tmp_path):
        result_file = tmp_path / "ai_result.txt"
        result_file.write_text(
            '{"sections": [{"title": "公司概况", "content": "营收1.5亿元"}], '
            '"qa_pairs": [{"question": "营收多少？", "answer": "1.5亿元", "asker": "分析师"}]}',
            encoding="utf-8",
        )
        runner = CliRunner()
        result = self._run(runner, ["render", "--json", str(result_file)])
        assert result.exit_code == 0, result.output
        formal = tmp_path / "ai_result_成品稿.md"
        assert formal.exists()
        assert "成品内容" in formal.read_text(encoding="utf-8")

    def test_render_from_json_falls_back_to_content(self, tmp_path):
        """非 JSON 时整段作为内容兜底（与 import-result 一致，信息零丢失），仍出成品稿。"""
        result_file = tmp_path / "bad.txt"
        result_file.write_text("没有可用 JSON 的内容。", encoding="utf-8")
        runner = CliRunner()
        result = self._run(runner, ["render", "-j", str(result_file)])
        assert result.exit_code == 0, result.output
        formal = tmp_path / "bad_成品稿.md"
        assert formal.exists()

    def test_render_json_passes_dicts_to_generator(self, tmp_path):
        """回归：-j 路径此前把 QAPair 对象传给 FormalGenerator（内部 .get）导致 TypeError；
        现在应传 dict，真实 _format_qa_pairs 可正常处理。"""
        result_file = tmp_path / "ai_result.txt"
        result_file.write_text(
            '{"sections": [{"title": "公司概况", "content": "营收1.5亿元"}], '
            '"qa_pairs": [{"question": "营收多少？", "answer": "1.5亿元", "asker": "分析师"}]}',
            encoding="utf-8",
        )
        runner = CliRunner()
        llm = MagicMock()
        llm.generate.return_value = "成品内容。"
        cfg_mock = MagicMock()
        cfg_mock.get_llm_client.return_value = llm
        with (
            patch("finminutes.cli._get_config", return_value=cfg_mock),
            patch("finminutes.cli._ensure_llm_config", return_value=True),
        ):
            result = runner.invoke(main, ["render", "-j", str(result_file)])
        assert result.exit_code == 0, result.output
        formal = tmp_path / "ai_result_成品稿.md"
        assert formal.exists()
        # 确认喂给 LLM 的 prompt 里包含格式化后的 QA（说明 dict 转换成功）
        args = llm.generate.call_args
        assert args is not None
        assert "营收多少" in args.kwargs["prompt"]

    def test_render_requires_exactly_one_input(self, tmp_path):
        """-r 与 -j 必须且只能给一个；都不给或都给都要报错。"""
        result_file = tmp_path / "ai_result.txt"
        result_file.write_text('{"qa_pairs": [{"question": "q", "answer": "a"}]}', encoding="utf-8")
        review_file = tmp_path / "review.md"
        review_file.write_text(
            "---\nqa_pairs:\n- question: q\n  answer: a\n---\n# 校验稿\n", encoding="utf-8"
        )
        runner = CliRunner()
        for args in (
            ["render"],                              # 都不给
            ["render", "-r", str(review_file), "-j", str(result_file)],  # 都给
        ):
            result = self._run(runner, args)
            assert result.exit_code != 0, (args, result.output)
            assert "二选一" in result.output
