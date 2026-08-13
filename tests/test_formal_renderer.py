from unittest.mock import MagicMock

from finminutes.core.formal_renderer import FormalGenerator, _format_sections
from finminutes.core.models import Template, Section
from finminutes.core.summarizer import SectionContent


def test_format_sections():
    secs = [SectionContent(title="公司", content="营收1.5亿元")]
    out = _format_sections(secs)
    assert "公司" in out
    assert "营收1.5亿元" in out


def test_format_sections_empty():
    assert _format_sections([]) == "（无）"


def test_generate_includes_qa_and_sections_in_prompt():
    llm = MagicMock()
    llm.generate.return_value = "成品稿内容"
    gen = FormalGenerator(llm)
    template = Template(name="t", description="会议纪要", sections=[Section(title="总结")])
    qa = [{"question": "营收多少？", "answer": "1.5亿元", "timerange": ""}]
    secs = [SectionContent(title="公司概况", content="独白内容，出货14亿颗")]

    gen.generate(qa, template, sections=secs)

    prompt = llm.generate.call_args.kwargs.get("prompt") or llm.generate.call_args[0][0]
    assert "营收多少？" in prompt
    assert "1.5亿元" in prompt
    assert "公司概况" in prompt
    assert "14亿颗" in prompt


def test_generate_without_sections_marks_absent():
    llm = MagicMock()
    llm.generate.return_value = "成品稿内容"
    gen = FormalGenerator(llm)
    template = Template(name="t", sections=[Section(title="总结")])
    qa = [{"question": "Q", "answer": "A"}]

    gen.generate(qa, template, sections=[])

    prompt = llm.generate.call_args.kwargs.get("prompt") or llm.generate.call_args[0][0]
    assert "无主题要点" in prompt


def test_generate_speech_only_no_qa():
    """纯独白模式：无 Q&A、只有主题要点时，prompt 标记无问答。"""
    llm = MagicMock()
    llm.generate.return_value = "成品稿内容"
    gen = FormalGenerator(llm)
    template = Template(name="t", sections=[Section(title="总结")])
    secs = [SectionContent(title="公司概况", content="营收1.5亿元")]

    gen.generate([], template, sections=secs)

    prompt = llm.generate.call_args.kwargs.get("prompt") or llm.generate.call_args[0][0]
    assert "无问答" in prompt
    assert "营收1.5亿元" in prompt
