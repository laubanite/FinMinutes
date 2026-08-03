import json
from unittest.mock import MagicMock

import pytest

from finminutes.core.models import Section, Template
from finminutes.core.summarizer import (
    QAPair,
    SectionContent,
    StructuredGenerator,
    StructuredMinutes,
    Takeaway,
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

    def test_takeaway_defaults(self):
        t = Takeaway()
        assert t.type == "事实"

    def test_structured_minutes_defaults(self):
        m = StructuredMinutes()
        assert m.sections == []
        assert m.qa_pairs == []
        assert m.takeaways == []


class TestStructuredGenerator:
    @pytest.fixture
    def llm_mock(self):
        m = MagicMock()
        data = {
            "sections": [
                {"title": "会议概况", "content": "会议于2024年举行", "citations": ["L1", "L2"]},
                {"title": "核心讨论", "content": "讨论了增长策略", "citations": ["L3"]},
            ],
            "qa_pairs": [
                {"question": "Q1?", "answer": "A1", "asker": "张三"},
            ],
            "takeaways": [
                {"content": "增长良好", "type": "事实"},
                {"content": "可能上市", "type": "推断"},
            ],
        }
        m.generate.return_value = json.dumps(data, ensure_ascii=False)
        return m

    def test_generate_returns_structured_minutes(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        result = gen.generate("some transcript")
        assert isinstance(result, StructuredMinutes)

    def test_generate_has_sections(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        result = gen.generate("transcript")
        assert len(result.sections) == 2
        assert result.sections[0].title == "会议概况"
        assert result.sections[1].citations == ["L3"]

    def test_generate_has_qa_pairs(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        result = gen.generate("transcript")
        assert len(result.qa_pairs) == 1
        assert result.qa_pairs[0].asker == "张三"

    def test_generate_has_takeaways(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        result = gen.generate("transcript")
        assert len(result.takeaways) == 2
        assert result.takeaways[1].type == "推断"

    def test_prompt_contains_qa_extraction(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        gen.generate("test transcript")
        prompt = llm_mock.generate.call_args[0][0]
        assert "Q&A提取专家" in prompt
        assert "qa_pairs" in prompt

    def test_prompt_contains_transcript(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        gen.generate("specific meeting text")
        prompt = llm_mock.generate.call_args[0][0]
        assert "specific meeting text" in prompt

    def test_empty_transcript(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        result = gen.generate("")
        assert result.sections == []
        llm_mock.generate.assert_not_called()

    def test_raw_json_stored(self, llm_mock, template):
        gen = StructuredGenerator(llm_mock, template)
        result = gen.generate("transcript")
        assert "会议概况" in result.raw_json


class TestParseResponse:
    def test_direct_json_parsing(self, template):
        llm = MagicMock()
        llm.generate.return_value = '{"sections": [{"title": "T", "content": "C", "citations": []}], "qa_pairs": [], "takeaways": []}'
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        assert len(result.sections) == 1

    def test_retry_on_invalid_json(self, template):
        llm = MagicMock()
        # First call returns invalid, second returns valid
        valid = '{"sections": [{"title": "T", "content": "C", "citations": []}], "qa_pairs": [], "takeaways": []}'
        llm.generate.side_effect = ["some text not json", valid]
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        assert len(result.sections) == 1
        assert llm.generate.call_count == 2

    def test_regex_extract_fallback(self, template):
        llm = MagicMock()
        data = '{"sections": [{"title": "T", "content": "C", "citations": []}], "qa_pairs": [], "takeaways": []}'
        llm.generate.side_effect = [
            "some text not json",
            "Here is the JSON: " + data + " end",
        ]
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        assert len(result.sections) == 1

    def test_markdown_code_block_extraction(self, template):
        llm = MagicMock()
        data = '{"sections": [{"title": "T", "content": "C", "citations": []}], "qa_pairs": [], "takeaways": []}'
        text = f"```json\n{data}\n```"
        llm.generate.return_value = text
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        assert len(result.sections) == 1

    def test_complete_fallback(self, template):
        llm = MagicMock()
        # All attempts fail
        llm.generate.side_effect = [
            "not json at all",
            "still not json",
        ]
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        # Should have fallback section with raw text
        assert len(result.sections) >= 1

    def test_missing_optional_fields(self, template):
        llm = MagicMock()
        llm.generate.return_value = '{"sections": [{"title": "T"}], "qa_pairs": [{"question": "Q"}], "takeaways": [{}]}'
        gen = StructuredGenerator(llm, template)
        result = gen.generate("text")
        assert result.sections[0].content == ""
        assert result.sections[0].citations == []
        assert result.qa_pairs[0].answer == ""
        assert result.qa_pairs[0].asker == ""
        assert result.takeaways[0].content == ""


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
