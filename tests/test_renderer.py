from finminutes.core.fact_checker import CitationCheck, CoverageReport, FactCheckReport, UnsupportedClaim
from finminutes.core.renderer import MarkdownRenderer
from finminutes.core.summarizer import QAPair, SectionContent, StructuredMinutes


class TestRenderSections:
    def test_empty_minutes(self):
        r = MarkdownRenderer(StructuredMinutes())
        result = r.render()
        assert "# 会议纪要" in result

    def test_section_with_citations(self):
        minutes = StructuredMinutes(
            sections=[SectionContent(title="概况", content="会议内容", citations=["L1", "L3"])],
        )
        r = MarkdownRenderer(minutes)
        result = r.render()
        assert "## 1. 概况 (L1) (L3)" in result
        assert "会议内容" in result

    def test_section_without_citations(self):
        minutes = StructuredMinutes(
            sections=[SectionContent(title="概况", content="会议内容")],
        )
        r = MarkdownRenderer(minutes)
        result = r.render()
        assert "## 1. 概况" in result
        assert "{[L" not in result

    def test_section_empty_content(self):
        minutes = StructuredMinutes(
            sections=[SectionContent(title="概况", content="")],
        )
        r = MarkdownRenderer(minutes)
        result = r.render()
        assert "（无内容）" in result

    def test_multiple_sections_numbered(self):
        minutes = StructuredMinutes(
            sections=[
                SectionContent(title="S1", content="C1"),
                SectionContent(title="S2", content="C2"),
                SectionContent(title="S3", content="C3"),
            ],
        )
        r = MarkdownRenderer(minutes)
        result = r.render()
        assert "## 1. S1" in result
        assert "## 2. S2" in result
        assert "## 3. S3" in result


class TestRenderQA:
    def test_qa_section_rendered(self):
        minutes = StructuredMinutes(
            qa_pairs=[QAPair(question="Q1?", answer="A1", asker="张三")],
        )
        r = MarkdownRenderer(minutes)
        result = r.render()
        assert "## Q&A 环节" in result
        assert "张三" in result
        assert "Q1?" in result
        assert "A1" in result

    def test_qa_without_asker(self):
        minutes = StructuredMinutes(
            qa_pairs=[QAPair(question="Q?", answer="A")],
        )
        r = MarkdownRenderer(minutes)
        result = r.render()
        assert "（未指明）" in result

    def test_qa_without_answer(self):
        minutes = StructuredMinutes(
            qa_pairs=[QAPair(question="Q?", answer="")],
        )
        r = MarkdownRenderer(minutes)
        result = r.render()
        assert "Q?" in result
        # No answer line after
        lines = result.split("\n")
        qa_idx = next(i for i, l in enumerate(lines) if "Q?" in l)
        assert qa_idx < len(lines) - 1

    def test_no_qa_omits_section(self):
        minutes = StructuredMinutes()
        r = MarkdownRenderer(minutes)
        result = r.render()
        assert "Q&A" not in result


class TestRenderValidationReport:
    def test_report_not_included_when_none(self):
        minutes = StructuredMinutes(sections=[SectionContent(title="S", content="C")])
        r = MarkdownRenderer(minutes, report=None)
        result = r.render()
        assert "事实校验报告" not in result

    def test_report_empty_passed(self):
        minutes = StructuredMinutes()
        report = FactCheckReport()
        r = MarkdownRenderer(minutes, report=report)
        result = r.render()
        assert "事实校验报告" in result
        assert "[通过]" in result
        assert "100%" in result

    def test_report_with_coverage_warning(self):
        minutes = StructuredMinutes()
        report = FactCheckReport(
            sections_checked=1,
            citation_checks=[CitationCheck(citation="L1", line_index=1, valid=True)],
            coverage=CoverageReport(total_chunks=10, uncovered_chunks=6, ratio=0.4),
        )
        r = MarkdownRenderer(minutes, report=report)
        result = r.render()
        assert "[异常]" in result
        assert "内容覆盖率" in result
        assert "40%" in result
        assert "L1" in result

    def test_report_with_invalid_citation(self):
        minutes = StructuredMinutes()
        report = FactCheckReport(
            citation_checks=[CitationCheck(citation="bad", line_index=-1, valid=False)],
        )
        r = MarkdownRenderer(minutes, report=report)
        result = r.render()
        assert "[无效]" in result
        assert "bad" in result

    def test_report_with_unsupported_claims(self):
        minutes = StructuredMinutes()
        report = FactCheckReport(
            unsupported_claims=[
                UnsupportedClaim(claim_snippet="some text", claim_type="section", title="S1", issue="无引用来源"),
            ],
        )
        r = MarkdownRenderer(minutes, report=report)
        result = r.render()
        assert "[待确认]" in result
        assert "无引用来源" in result
        assert "S1" in result

    def test_report_confidence_score_displayed(self):
        minutes = StructuredMinutes()
        report = FactCheckReport(sections_checked=2)
        r = MarkdownRenderer(minutes, report=report)
        result = r.render()
        assert "100%" in result

    def test_report_citation_valid_with_content(self):
        minutes = StructuredMinutes()
        report = FactCheckReport(
            citation_checks=[
                CitationCheck(citation="L5", line_index=5, content="some source text", valid=True),
            ],
        )
        r = MarkdownRenderer(minutes, report=report)
        result = r.render()
        assert "[通过]" in result
        assert "L5" in result
        assert "some source text" in result


class TestFullRendering:
    def test_full_document_structure(self):
        minutes = StructuredMinutes(
            sections=[
                SectionContent(title="概况", content="2024年营收300亿", citations=["L1"]),
            ],
            qa_pairs=[QAPair(question="增长如何？", answer="良好", asker="王")],
        )
        report = FactCheckReport(
            sections_checked=1,
            qa_pairs_checked=1,
            citation_checks=[CitationCheck(citation="L1", line_index=1, content="2024年营收300亿", valid=True)],
        )
        r = MarkdownRenderer(minutes, report=report)
        result = r.render()

        assert "# 会议纪要" in result
        assert "## 1. 概况" in result
        assert "## Q&A 环节" in result
        assert "## 事实校验报告" in result

    def test_no_report_sections_only(self):
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="C1")],
        )
        r = MarkdownRenderer(minutes)
        result = r.render()
        assert "# 会议纪要" in result
        assert "## 1. S1" in result
        assert "Q&A" not in result
        assert "核心要点" not in result
        assert "事实校验报告" not in result
