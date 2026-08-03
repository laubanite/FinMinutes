import json
from unittest.mock import MagicMock, patch

import pytest

from finminutes.core.pipeline import Pipeline, PipelineResult


SAMPLE_TRANSCRIPT_SMALL = (
    "王总：大家好，今天讨论Q3业绩。\n"
    "王总：嗯，这个营收大概是三亿元。\n"
    "李总：毛利率保持在40%左右，净利润八千万。 "
)


SAMPLE_TRANSCRIPT_MEDIUM = "Sentence one. " * 200  # ~3000 chars


SAMPLE_TRANSCRIPT_LARGE = "Long text. " * 400  # ~4000+ chars (adjust to be 5000+)


SAMPLE_STRUCTURED_JSON = json.dumps(
    {
        "sections": [
            {"title": "业绩概况", "content": "讨论Q3业绩，营收150亿", "citations": ["L1"]},
        ],
        "qa_pairs": [],
        "takeaways": [{"content": "营收增长", "type": "事实"}],
    },
    ensure_ascii=False,
)


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.generate.return_value = "增强后的文本内容"
    return m


@pytest.fixture
def mock_config(mock_llm):
    m = MagicMock()
    m.get_llm_client.return_value = mock_llm
    return m


class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult()
        assert r.mode == ""
        assert r.transcript_raw == ""
        assert r.transcript_cleaned == ""
        assert r.structured_minutes is None
        assert r.fact_check_report is None
        assert r.markdown_output == ""
        assert r.errors == []
        assert r.success is True
        assert r.performance_tier == ""

    def test_summary_fast_mode(self):
        r = PipelineResult()
        r.mode = "fast"
        r.performance_tier = "small"
        r.transcript_raw = "hello"
        r.transcript_cleaned = "hello"
        r.elapsed_total = 0.05
        r.success = True
        s = r.summary
        assert "fast" in s
        assert "small" in s
        assert "成功" in s
        assert "0.05" in s

    def test_summary_with_errors(self):
        r = PipelineResult()
        r.errors = ["something broke"]
        r.success = False
        s = r.summary
        assert "有错误" in s
        assert "1" in s


class TestPerformanceClassification:
    def test_small(self):
        assert Pipeline._classify("a" * 999) == "small"

    def test_medium(self):
        assert Pipeline._classify("a" * 1000) == "medium"

    def test_large(self):
        assert Pipeline._classify("a" * 5000) == "large"

    def test_empty(self):
        assert Pipeline._classify("") == "small"


class TestPipelineFastMode:
    def test_fast_mode_cleans_transcript(self, mock_config):
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="fast")
        assert result.mode == "fast"
        assert result.transcript_cleaned != result.transcript_raw
        assert result.performance_tier == "small"
        assert result.markdown_output != ""
        assert result.structured_minutes is None
        assert result.fact_check_report is None
        assert result.success is True

    def test_fast_mode_no_llm_calls(self, mock_config, mock_llm):
        p = Pipeline(mock_config)
        p.run(SAMPLE_TRANSCRIPT_SMALL, mode="fast")
        mock_llm.generate.assert_not_called()

    def test_fast_mode_empty_transcript(self, mock_config):
        p = Pipeline(mock_config)
        result = p.run("", mode="fast")
        assert result.success is True
        assert result.markdown_output != ""


class TestPipelineStandardMode:
    def test_standard_mode_calls_rewrite(self, mock_config, mock_llm):
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="standard")
        assert result.mode == "standard"
        assert result.transcript_enhanced != ""
        mock_llm.generate.assert_called_once()

    def test_standard_mode_no_structure(self, mock_config, mock_llm):
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="standard")
        assert result.structured_minutes is None
        assert result.fact_check_report is None

    def test_standard_mode_rewrite_error(self, mock_config):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM failed")
        mock_config.get_llm_client.return_value = mock_llm
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="standard")
        assert result.success is False
        assert len(result.errors) >= 1
        assert "改写失败" in result.errors[0]
        assert result.transcript_enhanced != ""  # falls back to cleaned


class TestPipelineFullMode:
    def test_full_mode_calls_all_phases(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", )
        assert result.mode == "full"
        assert result.transcript_enhanced != ""
        assert result.structured_minutes is not None
        assert result.fact_check_report is not None
        assert result.markdown_output != ""
        assert mock_llm.generate.call_count == 2

    def test_full_mode_with_context_files(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run(
            SAMPLE_TRANSCRIPT_SMALL,
            mode="full",
            background_path="",
            glossary_tag="",
        )
        assert result.success is True
        assert result.structured_minutes is not None

    def test_full_mode_summarize_error(self, mock_config):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            RuntimeError("summarize failed"),
        ]
        mock_config.get_llm_client.return_value = mock_llm
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", )
        assert len(result.errors) >= 1
        assert any("结构化摘要" in e for e in result.errors)
        assert result.markdown_output != ""

    def test_full_mode_factcheck_error(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", )
        assert result.fact_check_report is not None

    def test_full_mode_confidence_in_summary(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", )
        s = result.summary
        assert "事实校验置信度" in s

    def test_full_mode_large_transcript(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run("Large " * 3000, mode="full")
        assert result.performance_tier == "large"

    def test_full_mode_builtin_template(self, mock_config, mock_llm):
        mock_llm.generate.return_value = '{"sections":[{"title":"S1","content":"C1","citations":["L1"]}],"qa_pairs":[],"takeaways":[]}'
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full")
        # Built-in default prompt is always used, so structured_minutes is generated
        assert result.structured_minutes is not None
        assert len(result.structured_minutes.sections) == 1


class TestPipelineErrorHandling:
    def test_preprocess_error(self, mock_config):
        p = Pipeline(mock_config)
        with patch.object(p._preprocessor, "clean", side_effect=ValueError("clean failed")):
            result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="fast")
            assert result.success is False
            assert "预处理失败" in result.errors[0]
            assert result.transcript_cleaned == SAMPLE_TRANSCRIPT_SMALL

    def test_render_error(self, mock_config):
        p = Pipeline(mock_config)
        with patch(
            "finminutes.core.renderer.MarkdownRenderer.render",
            side_effect=RuntimeError("render failed"),
        ):
            result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="fast")
            assert result.success is False
            assert "渲染失败" in result.errors[-1]
            assert "渲染过程中出错" in result.markdown_output

    def test_context_load_error(self, mock_config):
        p = Pipeline(mock_config)
        result = p.run(
            SAMPLE_TRANSCRIPT_SMALL,
            mode="fast",
            background_path="nonexistent.yaml",
        )
        assert not result.success
        assert any("加载数据失败" in e for e in result.errors)

    def test_multiple_errors_recorded(self, mock_config):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("rewrite fail")
        mock_config.get_llm_client.return_value = mock_llm
        p = Pipeline(mock_config)
        with patch.object(p._preprocessor, "clean", side_effect=ValueError("preprocess fail")):
            result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full")
            assert len(result.errors) >= 2
            assert result.success is False

    def test_timing_info_present(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", )
        assert result.elapsed_total >= 0
        assert result.elapsed_preprocess >= 0
        assert result.elapsed_rewrite >= 0
        assert result.elapsed_summarize >= 0
        assert result.elapsed_factcheck >= 0
        assert result.elapsed_render >= 0
