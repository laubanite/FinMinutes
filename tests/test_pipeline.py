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


class TestPipelineFullRewriteError:
    """standard 模式已删除：改写失败中止路径现仅在 full（speech/both 触发改写时）可达。"""

    def test_full_rewrite_error_aborts(self, mock_config):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM failed")
        mock_config.get_llm_client.return_value = mock_llm
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="speech")
        assert result.success is False
        assert len(result.errors) >= 1
        assert "改写失败" in result.errors[0]
        assert result.transcript_enhanced != ""  # falls back to cleaned
        assert result.aborted is True  # 关键阶段失败 → 中止
        assert result.markdown_output == ""  # 中止后不再渲染

    def test_full_rewrite_error_on_error_called(self, mock_config):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM failed")
        mock_config.get_llm_client.return_value = mock_llm
        p = Pipeline(mock_config)
        calls = []
        result = p.run(
            SAMPLE_TRANSCRIPT_SMALL, mode="full", format="speech",
            on_error=lambda stage, msg: calls.append((stage, msg)),
        )
        assert result.aborted is True
        assert calls and calls[0][0] == "rewrite"
        assert "改写失败" in calls[0][1]


class TestPipelineFullMode:
    def test_full_mode_calls_all_phases(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="both")
        assert result.mode == "full"
        assert result.transcript_enhanced != ""
        assert result.structured_minutes is not None
        assert result.fact_check_report is not None
        assert result.markdown_output != ""
        assert mock_llm.generate.call_count == 2

    def test_full_mode_rewrite_topic_blocks_become_sections(self, mock_config, mock_llm):
        """rewrite 产出的 topic 块直接派生 sections，QA 单独一次调用（调用数不翻倍）。"""
        mock_llm.generate.side_effect = [
            '[{"topic": "业绩", "text": "Q3营收约三亿元"}, {"topic": "毛利", "text": "毛利率40%"} ]',
            json.dumps({"qa_pairs": [], "takeaways": []}, ensure_ascii=False),
        ]
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="both")
        assert result.structured_minutes is not None
        titles = [s.title for s in result.structured_minutes.sections]
        assert titles == ["业绩", "毛利"], titles
        assert result.structured_minutes.sections[0].content == "Q3营收约三亿元"
        assert mock_llm.generate.call_count == 2  # rewrite 1 + QA 1（sections 零额外调用）

    def test_full_mode_with_context_files(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run(
            SAMPLE_TRANSCRIPT_SMALL,
            mode="full", format="both",
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
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="both")
        assert len(result.errors) >= 1
        assert any("结构化摘要" in e for e in result.errors)
        assert result.aborted is True  # 关键阶段失败 → 中止
        assert result.markdown_output == ""

    def test_full_mode_factcheck_error(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="both")
        assert result.fact_check_report is not None

    def test_full_mode_confidence_in_summary(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="both")
        s = result.summary
        assert "内容覆盖率" in s

    def test_full_mode_large_transcript(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run("Large " * 3000, mode="full", format="both")
        assert result.performance_tier == "large"

    def test_full_mode_builtin_template(self, mock_config, mock_llm):
        mock_llm.generate.return_value = '{"sections":[{"title":"S1","content":"C1","citations":["L1"]}],"qa_pairs":[],"takeaways":[]}'
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="both")
        # Built-in default prompt is always used, so structured_minutes is generated
        assert result.structured_minutes is not None
        assert len(result.structured_minutes.sections) == 1

    def test_full_mode_qa_skips_rewrite(self, mock_config, mock_llm):
        """format=qa：跳过 rewrite（省调用），校验稿只 Q&A、无主题要点。"""
        mock_llm.generate.return_value = json.dumps({"qa_pairs": [{"question": "Q1", "answer": "A1"}]}, ensure_ascii=False)
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="qa")
        assert mock_llm.generate.call_count == 1  # 只 QA 一次，rewrite 未调用
        assert result.structured_minutes is not None
        assert len(result.structured_minutes.sections) == 0
        assert len(result.structured_minutes.qa_pairs) == 1

    def test_full_mode_speech_skips_qa(self, mock_config, mock_llm):
        """format=speech：跳过 QA 提取，校验稿只主题要点。"""
        mock_llm.generate.return_value = '[{"type": "narration", "topic": "公司", "text": "介绍内容"}]'
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="speech")
        assert mock_llm.generate.call_count == 1  # 只 rewrite，QA 未调用
        assert result.structured_minutes is not None
        assert len(result.structured_minutes.sections) == 1
        assert len(result.structured_minutes.qa_pairs) == 0

    def test_full_mode_both_filters_qa_blocks_from_sections(self, mock_config, mock_llm):
        """format=both：rewrite 分片中的 qa 块不进主题要点（由 QA 提取覆盖）。"""
        mock_llm.generate.side_effect = [
            '[{"type": "narration", "topic": "公司概况", "text": "独白内容"}, {"type": "qa", "topic": "融资", "text": "问答内容"}]',
            json.dumps({"qa_pairs": [{"question": "融资多少？", "answer": "1.5亿"}]}, ensure_ascii=False),
        ]
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="both")
        titles = [s.title for s in result.structured_minutes.sections]
        assert titles == ["公司概况"]
        assert "问答内容" not in "".join(s.content for s in result.structured_minutes.sections)
        assert len(result.structured_minutes.qa_pairs) == 1

    def test_invalid_format(self, mock_config):
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="bogus")
        assert result.success is False
        assert any("format" in e.lower() for e in result.errors)


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
            result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="both")
            assert len(result.errors) >= 2
            assert result.success is False

    def test_timing_info_present(self, mock_config, mock_llm):
        mock_llm.generate.side_effect = [
            "增强后的文本内容",
            SAMPLE_STRUCTURED_JSON,
        ]
        p = Pipeline(mock_config)
        result = p.run(SAMPLE_TRANSCRIPT_SMALL, mode="full", format="both")
        assert result.elapsed_total >= 0
        assert result.elapsed_preprocess >= 0
        assert result.elapsed_rewrite >= 0
        assert result.elapsed_summarize >= 0
        assert result.elapsed_factcheck >= 0
        assert result.elapsed_render >= 0
