import json
from unittest.mock import MagicMock

import pytest

from finminutes.core.models import Section, Template
from finminutes.core.rewriter import parse_blocks
from finminutes.core.summarizer import (
    QAPair,
    SectionContent,
    StructuredGenerator,
    StructuredMinutes,
    _topics_similar,
    merge_blocks,
)


@pytest.fixture
def template():
    return Template(
        name="expert_interview",
        description="测试",
        sections=[
            Section(title="会议概况", prompt="简述会议信息"),
            Section(title="核心讨论", prompt="核心讨论内容"),
            Section(title="Q&A 精选", prompt="问答环节"),
        ],
    )


SECTIONS_BLOCKS = [
    [
        {"topic": "会议概况", "text": "会议于2024年举行"},
        {"topic": "核心讨论", "text": "讨论了增长策略"},
    ],
]


class TestDataClasses:
    def test_section_content_defaults(self):
        s = SectionContent()
        assert s.title == ""
        assert s.content == ""
        assert s.citations == []

    def test_qa_pair_defaults(self):
        q = QAPair()
        assert q.question == ""
        assert q.answer == ""
        assert q.asker == ""

    def test_structured_minutes_defaults(self):
        m = StructuredMinutes()
        assert m.sections == []
        assert m.qa_pairs == []


class TestStructuredGenerator:
    @pytest.fixture
    def llm_mock(self):
        m = MagicMock()
        data = {
            "qa_pairs": [
                {"question": "Q1?", "answer": "A1", "asker": "张三"},
            ],
        }
        m.generate.return_value = json.dumps(data, ensure_ascii=False)
        return m

    def test_generate_returns_structured_minutes(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        result = gen.generate("some transcript", section_blocks=SECTIONS_BLOCKS)
        assert isinstance(result, StructuredMinutes)

    def test_generate_has_sections(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        result = gen.generate("transcript", section_blocks=SECTIONS_BLOCKS)
        assert len(result.sections) == 2
        assert result.sections[0].title == "会议概况"
        assert result.sections[0].content == "会议于2024年举行"
        assert result.sections[1].title == "核心讨论"

    def test_generate_has_qa_pairs(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        result = gen.generate("transcript", section_blocks=SECTIONS_BLOCKS)
        assert len(result.qa_pairs) == 1
        assert result.qa_pairs[0].asker == "张三"

    def test_prompt_contains_qa_extraction(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        gen.generate("test transcript", section_blocks=SECTIONS_BLOCKS)
        prompt = llm_mock.generate.call_args[0][0]
        assert "Q&A提取专家" in prompt
        assert "qa_pairs" in prompt

    def test_prompt_contains_transcript(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        gen.generate("specific meeting text", section_blocks=SECTIONS_BLOCKS)
        prompt = llm_mock.generate.call_args[0][0]
        assert "specific meeting text" in prompt

    def test_prompt_narration_instruction(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        gen.generate("这是一段完全没有问号的单向陈述内容。" * 30)
        prompt = llm_mock.generate.call_args[0][0]
        assert "单向陈述" in prompt

    def test_prompt_qa_instruction(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        gen.generate("请问毛利率是多少？营收增长怎么样？" * 30)
        prompt = llm_mock.generate.call_args[0][0]
        assert "问答为主" in prompt

    def test_empty_transcript(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        result = gen.generate("")
        assert result.sections == []
        llm_mock.generate.assert_not_called()

    def test_raw_json_stored(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        result = gen.generate("transcript", section_blocks=SECTIONS_BLOCKS)
        assert "会议概况" in result.raw_json

    def test_generate_with_section_blocks_only_qa_call(self, template):
        llm = MagicMock()
        llm.generate.return_value = '{"qa_pairs": [], "takeaways": []}'
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text", section_blocks=[[{"topic": "T", "text": "C"}]])
        assert result.sections[0].title == "T"
        assert llm.generate.call_count == 1  # 只调 QA 一次

    def test_generate_without_section_blocks_chunks(self, template):
        llm = MagicMock()
        llm.generate.side_effect = [
            '[{"topic": "话题", "text": "内容1"}]',
            '{"qa_pairs": [], "takeaways": []}',
        ]
        gen = StructuredGenerator(llm, template)
        result = gen.generate("some short transcript")
        assert len(result.sections) == 1
        assert result.sections[0].title == "话题"
        assert llm.generate.call_count == 2  # 块提取 + QA

    def test_generate_qa_uses_qa_source_not_enhanced(self, template):
        """QA 提取应读取改写前原文（qa_source），而非 formalized 文本。"""
        llm = MagicMock()
        llm.generate.return_value = '{"qa_pairs": [{"question": "Q?", "answer": "A"}], "takeaways": []}'
        gen = StructuredGenerator(llm, template)
        gen.generate(
            "正式化的改写文本内容",
            section_blocks=[[{"topic": "T", "text": "C"}]],
            qa_source="原始的提问文本？这里问了一个问题",
        )
        prompt = llm.generate.call_args[0][0]
        assert "原始的提问文本？这里问了一个问题" in prompt
        assert "正式化的改写文本内容" not in prompt

    def test_qa_single_call_when_short(self, template):
        """未超过 qa_chunk_size → 单次调用（不额外分块）。"""
        llm = MagicMock()
        llm.generate.return_value = '{"qa_pairs": [], "takeaways": []}'
        gen = StructuredGenerator(llm, template, qa_chunk_size=200)
        gen.generate("短文本？" * 30, section_blocks=SECTIONS_BLOCKS)
        assert llm.generate.call_count == 1

    def test_qa_chunked_splits_long_transcript(self, template):
        """超过 qa_chunk_size → 按块提取并按序合并；每次调用只携带子块。"""
        llm = MagicMock()
        counter = {"n": 0}

        def fake_generate(prompt, **kw):
            counter["n"] += 1
            return json.dumps({
                "qa_pairs": [{"question": f"Q{counter['n']}？", "answer": f"A{counter['n']}"}],
            }, ensure_ascii=False)

        llm.generate.side_effect = fake_generate
        gen = StructuredGenerator(llm, template, qa_chunk_size=200)
        long_text = "我们讨论了毛利率和营收情况？这是一个很长的问题。" * 50
        result = gen.generate(long_text, section_blocks=SECTIONS_BLOCKS)
        assert counter["n"] >= 2
        assert len(result.qa_pairs) == counter["n"]
        assert result.qa_pairs[0].question == "Q1？"
        # 每次调用只带子块，不携带整篇长文本
        for call in llm.generate.call_args_list:
            assert len(call.args[0]) < len(long_text)

    def test_qa_chunked_dedupes_overlap(self, template):
        """重叠区重复提取的同一问题只保留一次（按问题文本判重）。"""
        llm = MagicMock()
        counter = {"n": 0}

        def fake_generate(prompt, **kw):
            counter["n"] += 1
            pairs = [
                [{"question": "Q1？", "answer": "A1"}],
                [{"question": "Q1？", "answer": "A1"}, {"question": "Q2？", "answer": "A2"}],
            ]
            return json.dumps({
                "qa_pairs": pairs[counter["n"] - 1] if counter["n"] <= 2 else [],
                "takeaways": [],
            }, ensure_ascii=False)

        llm.generate.side_effect = fake_generate
        gen = StructuredGenerator(llm, template, qa_chunk_size=200)
        result = gen.generate("长文本？" * 60, section_blocks=SECTIONS_BLOCKS)
        assert [q.question for q in result.qa_pairs] == ["Q1？", "Q2？"]

    def test_qa_chunked_partial_failure_keeps_good_chunks(self, template):
        """单块失败只跳过该块，不丢弃其余块的成功结果。"""
        llm = MagicMock()
        counter = {"n": 0}

        def fake_generate(prompt, **kw):
            counter["n"] += 1
            if counter["n"] == 2:
                raise RuntimeError("boom")
            return json.dumps({
                "qa_pairs": [{"question": f"Q{counter['n']}？", "answer": f"A{counter['n']}"}],
                "takeaways": [],
            }, ensure_ascii=False)

        llm.generate.side_effect = fake_generate
        gen = StructuredGenerator(llm, template, qa_chunk_size=200)
        result = gen.generate("长文本？" * 130, section_blocks=SECTIONS_BLOCKS)
        questions = [q.question for q in result.qa_pairs]
        assert counter["n"] >= 3
        assert "Q1？" in questions
        assert "Q2？" not in questions
        assert len(questions) == counter["n"] - 1

    def test_qa_chunked_all_fail_raises(self, template):
        """全部块都失败 → 与单次调用一致向上抛（fail-fast）。"""
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("boom")
        gen = StructuredGenerator(llm, template, qa_chunk_size=200)
        with pytest.raises(RuntimeError):
            gen.generate("长文本？" * 130, section_blocks=SECTIONS_BLOCKS)

    def test_qa_chunked_keeps_empty_answer(self, template):
        """去重只按问题判重；『有问无答』的对保留，不因答案为空被丢弃。"""
        llm = MagicMock()
        llm.generate.return_value = json.dumps({
            "qa_pairs": [{"question": "Q1？", "answer": ""}],
            "takeaways": [],
        }, ensure_ascii=False)
        gen = StructuredGenerator(llm, template, qa_chunk_size=200)
        result = gen.generate("长文本？" * 130, section_blocks=SECTIONS_BLOCKS)
        assert [q.question for q in result.qa_pairs] == ["Q1？"]
        assert result.qa_pairs[0].answer == ""


class TestParseResponse:
    def test_direct_json_parsing(self, template):
        llm = MagicMock()
        llm.generate.side_effect = [
            '[{"topic": "T", "text": "C"}]',
            '{"qa_pairs": [], "takeaways": []}',
        ]
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        assert len(result.sections) == 1
        assert result.sections[0].title == "T"

    def test_retry_on_invalid_json(self, template):
        llm = MagicMock()
        llm.generate.side_effect = [
            '[{"topic": "T", "text": "C"}]',
            "some text not json",
            '{"qa_pairs": [], "takeaways": []}',
        ]
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        assert len(result.sections) == 1
        assert llm.generate.call_count == 3  # 块 + QA 首次 + QA 重试

    def test_regex_extract_fallback(self, template):
        llm = MagicMock()
        llm.generate.side_effect = [
            '[{"topic": "T", "text": "C"}]',
            "some text not json",
            "Here is the JSON: " + '{"qa_pairs": [], "takeaways": []}' + " end",
        ]
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        assert len(result.sections) == 1

    def test_markdown_code_block_extraction(self, template):
        llm = MagicMock()
        block = '[{"topic": "T", "text": "C"}]'
        text = f"```json\n{block}\n```"
        llm.generate.side_effect = [
            text,
            '{"qa_pairs": [], "takeaways": []}',
        ]
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        assert len(result.sections) == 1

    def test_complete_fallback(self, template):
        llm = MagicMock()
        llm.generate.side_effect = [
            "block not json",  # 块解析失败 → 兜底原文块，内容不丢
            "qa not json at all",
            "still not json",
        ]
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        assert len(result.sections) >= 1
        assert result.qa_pairs == []

    def test_missing_optional_fields(self, template):
        llm = MagicMock()
        llm.generate.side_effect = [
            '[{"topic": "T", "text": "C"}]',
            '{"qa_pairs": [{"question": "Q"}]}',
        ]
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        assert result.sections[0].title == "T"
        assert result.qa_pairs[0].answer == ""
        assert result.qa_pairs[0].asker == ""


class TestParseBlocksAndMerge:
    def test_parse_blocks_json(self):
        text, blocks = parse_blocks('[{"topic": "A", "text": "x"}, {"topic": "B", "text": "y"}]')
        assert len(blocks) == 2
        assert blocks[0]["topic"] == "A"
        assert blocks[1]["text"] == "y"

    def test_parse_blocks_fallback(self):
        text, blocks = parse_blocks("not json")
        assert blocks == [{"type": "narration", "topic": "", "text": "not json"}]

    def test_parse_blocks_preserves_type(self):
        _, blocks = parse_blocks('[{"type": "qa", "topic": "A", "text": "x"}]')
        assert blocks[0]["type"] == "qa"

    def test_parse_blocks_defaults_type_to_narration(self):
        _, blocks = parse_blocks('[{"topic": "A", "text": "x"}]')
        assert blocks[0]["type"] == "narration"

    def test_merge_blocks_skips_qa_type(self):
        secs = merge_blocks([
            [
                {"type": "narration", "topic": "公司概况", "text": "独白内容"},
                {"type": "qa", "topic": "融资", "text": "问答内容"},
                {"type": "mixed", "topic": "产能", "text": "陈述部分"},
            ],
        ])
        titles = [s.title for s in secs]
        assert titles == ["公司概况", "产能"]
        assert "问答内容" not in "".join(s.content for s in secs)

    def test_parse_blocks_markdown_wrapped(self):
        _, blocks = parse_blocks('```json\n[{"topic": "A", "text": "x"}]\n```')
        assert len(blocks) == 1
        assert blocks[0]["topic"] == "A"

    def test_merge_adjacent_same_topic(self):
        secs = merge_blocks([
            [{"topic": "价格", "text": "A1"}],
            [{"topic": "价格", "text": "A2"}],
        ])
        assert len(secs) == 1
        assert secs[0].content == "A1\n\nA2"

    def test_merge_preserves_order_non_adjacent(self):
        secs = merge_blocks([
            [{"topic": "价格", "text": "A1"}, {"topic": "工艺", "text": "B1"}],
            [{"topic": "价格", "text": "A2"}],
        ])
        assert [s.title for s in secs] == ["价格", "工艺", "价格"]

    def test_merge_skips_empty_text(self):
        secs = merge_blocks([[{"topic": "价格", "text": ""}]])
        assert secs == []

    def test_merge_normalizes_topic_prefix(self):
        secs = merge_blocks([
            [{"topic": "1. 价格", "text": "A1"}],
            [{"topic": "价格", "text": "A2"}],
        ])
        assert len(secs) == 1
        assert secs[0].title == "价格"

    def test_topics_similar(self):
        assert _topics_similar("价格", "1. 价格")
        assert _topics_similar("先进封装", "先进封装技术")
        assert not _topics_similar("价格", "工艺")


class TestJsonParsingEdgeCases:
    def test_extract_json_fragment_truncated(self, template):
        from finminutes.core.summarizer import StructuredGenerator

        text = 'some text {"sections": [{"title": "T", "content": "C"}], "qa_pairs": [], "takeaways": []'
        result = StructuredGenerator._extract_json_fragment(text)
        assert result is not None
        assert result["sections"][0]["title"] == "T"

    def test_extract_json_fragment_no_match(self, template):
        from finminutes.core.summarizer import StructuredGenerator

        assert StructuredGenerator._extract_json_fragment("no json here") is None

    def test_json_list_instead_of_dict(self, template):
        from finminutes.core.summarizer import StructuredGenerator

        assert StructuredGenerator._extract_json_fragment('[1, 2, 3]') is None


class TestTranscriptTypeDetection:
    def test_narration_low_question_density(self):
        from finminutes.core.summarizer import detect_transcript_type
        text = "我们公司成立于2010年。主要做半导体封装。产品覆盖消费电子。" * 30
        assert detect_transcript_type(text) == "narration"

    def test_qa_high_question_density(self):
        from finminutes.core.summarizer import detect_transcript_type
        text = "请问毛利率是多少？营收增长怎么样？你们怎么看？" * 30
        assert detect_transcript_type(text) == "qa"

    def test_empty(self):
        from finminutes.core.summarizer import detect_transcript_type
        # 空文本无问号 → 归为陈述型（不影响流程，generate 对空输入提前返回）
        assert detect_transcript_type("") == "narration"
