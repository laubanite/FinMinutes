from finminutes.core.fact_checker import (
    CitationCheck,
    FactChecker,
    FactCheckReport,
    NumericAnomaly,
    UnsupportedClaim,
)
from finminutes.core.summarizer import QAPair, SectionContent, StructuredMinutes, Takeaway


class TestDataClasses:
    def test_citation_check_defaults(self):
        c = CitationCheck(citation="L1", line_index=1, content="some text", valid=True)
        assert c.citation == "L1"
        assert c.line_index == 1
        assert c.content == "some text"
        assert c.valid is True

    def test_numeric_anomaly_defaults(self):
        a = NumericAnomaly(value_in_minutes="100", title="Test", citation="L1")
        assert a.value_in_minutes == "100"
        assert a.title == "Test"
        assert a.citation == "L1"
        assert a.source_text == ""

    def test_unsupported_claim_defaults(self):
        u = UnsupportedClaim(claim_snippet="text", claim_type="section", title="T", issue="no citation")
        assert u.claim_snippet == "text"
        assert u.claim_type == "section"
        assert u.title == "T"
        assert u.issue == "no citation"

    def test_report_defaults(self):
        r = FactCheckReport()
        assert r.sections_checked == 0
        assert r.qa_pairs_checked == 0
        assert r.takeaways_checked == 0
        assert r.citation_checks == []
        assert r.numeric_anomalies == []
        assert r.unsupported_claims == []

    def test_report_verified_with_no_issues(self):
        r = FactCheckReport(sections_checked=2, qa_pairs_checked=1, takeaways_checked=1)
        assert r.verified_overall is True
        assert r.confidence_score == 1.0

    def test_report_not_verified_with_anomalies(self):
        r = FactCheckReport(numeric_anomalies=[NumericAnomaly("100", "T", "L1")])
        assert r.verified_overall is False

    def test_report_confidence_score(self):
        r = FactCheckReport(
            sections_checked=3,
            qa_pairs_checked=1,
            takeaways_checked=1,
            unsupported_claims=[UnsupportedClaim("t", "takeaway", "事实", "无引用")],
        )
        assert r.confidence_score == 0.8

    def test_report_confidence_score_zero_division(self):
        r = FactCheckReport()
        assert r.confidence_score == 1.0


class TestCitationCheck:
    def test_valid_citation(self):
        transcript = "line one\nline two\nline three"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="text", citations=["L1", "L3"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.citation_checks) == 2
        assert report.citation_checks[0].citation == "L1"
        assert report.citation_checks[0].content == "line one"
        assert report.citation_checks[0].valid is True
        assert report.citation_checks[1].content == "line three"

    def test_invalid_citation_format(self):
        transcript = "line one"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="text", citations=["invalid_ref"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.citation_checks) == 1
        assert report.citation_checks[0].valid is False
        assert report.citation_checks[0].line_index == -1

    def test_citation_out_of_bounds_positive(self):
        transcript = "line one"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="text", citations=["L99"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert report.citation_checks[0].valid is False
        assert report.citation_checks[0].line_index == 99

    def test_citation_out_of_bounds_zero(self):
        transcript = "line one"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="text", citations=["L0"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert report.citation_checks[0].valid is False

    def test_multiple_sections_with_citations(self):
        transcript = "a\nb\nc"
        minutes = StructuredMinutes(
            sections=[
                SectionContent(title="S1", content="t1", citations=["L1"]),
                SectionContent(title="S2", content="t2", citations=["L2"]),
            ],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.citation_checks) == 2
        assert all(c.valid for c in report.citation_checks)


class TestNumbers:
    def test_no_numbers_returns_no_anomalies(self):
        transcript = "这是一段文字"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="没有数字", citations=["L1"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.numeric_anomalies) == 0

    def test_number_matches_source(self):
        transcript = "公司营收100万元"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="公司营收100万元", citations=["L1"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.numeric_anomalies) == 0

    def test_number_mismatch_source(self):
        transcript = "公司营收200万元"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="公司营收100万元", citations=["L1"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.numeric_anomalies) == 1
        assert report.numeric_anomalies[0].value_in_minutes == "100"
        assert report.numeric_anomalies[0].citation == "L1"
        assert report.numeric_anomalies[0].title == "S1"

    def test_percentage_number(self):
        transcript = "增长率25%"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="增长率25%", citations=["L1"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.numeric_anomalies) == 0

    def test_decimal_number(self):
        transcript = "营收3.5亿"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="营收3.5亿", citations=["L1"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.numeric_anomalies) == 0

    def test_multiple_numbers_partial_mismatch(self):
        transcript = "A股100点"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="A股100点港股200点", citations=["L1"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.numeric_anomalies) == 1
        assert report.numeric_anomalies[0].value_in_minutes == "200"

    def test_empty_content_skips_number_check(self):
        transcript = "some text"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="", citations=["L1"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.numeric_anomalies) == 0


class TestFactChecker:
    def test_check_empty_minutes(self):
        transcript = "some transcript"
        minutes = StructuredMinutes()
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert report.sections_checked == 0
        assert report.qa_pairs_checked == 0
        assert report.takeaways_checked == 0
        assert report.verified_overall is True

    def test_all_valid_returns_verified(self):
        transcript = "2024年营收300亿\n净利润50亿"
        minutes = StructuredMinutes(
            sections=[
                SectionContent(title="业绩", content="营收300亿", citations=["L1"]),
                SectionContent(title="利润", content="净利润50亿", citations=["L2"]),
            ],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert report.verified_overall is True
        assert report.confidence_score == 1.0

    def test_section_without_citations_flagged(self):
        transcript = "some text"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="重要内容", citations=[])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.unsupported_claims) == 1
        assert report.unsupported_claims[0].claim_type == "section"
        assert report.sections_checked == 1

    def test_section_empty_content_without_citations(self):
        transcript = "some text"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="", citations=[])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.unsupported_claims) == 1
        assert "内容为空" in report.unsupported_claims[0].issue

    def test_qa_pairs_flagged_as_unsupported(self):
        transcript = "some text"
        minutes = StructuredMinutes(qa_pairs=[QAPair(question="Q1", answer="A1", asker="张三")])
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.unsupported_claims) == 1
        assert report.unsupported_claims[0].claim_type == "qa_pair"

    def test_qa_pairs_empty_not_flagged(self):
        transcript = "some text"
        minutes = StructuredMinutes(qa_pairs=[QAPair()])
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.unsupported_claims) == 0

    def test_takeaways_flagged_as_unsupported(self):
        transcript = "some text"
        minutes = StructuredMinutes(takeaways=[Takeaway(content="重要", type="事实")])
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.unsupported_claims) == 1
        assert report.unsupported_claims[0].claim_type == "takeaway"

    def test_takeaways_empty_not_flagged(self):
        transcript = "some text"
        minutes = StructuredMinutes(takeaways=[Takeaway(content="")])
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.unsupported_claims) == 0

    def test_mixed_results(self):
        transcript = "实际营收150亿\n实际净利润30亿"
        minutes = StructuredMinutes(
            sections=[
                SectionContent(title="营收", content="营收150亿", citations=["L1"]),
                SectionContent(title="利润", content="净利润40亿", citations=["L2"]),
            ],
            qa_pairs=[QAPair(question="Q?", answer="重要内容")],
            takeaways=[Takeaway(content="核心要点")],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert report.sections_checked == 2
        assert report.qa_pairs_checked == 1
        assert report.takeaways_checked == 1
        assert len(report.citation_checks) == 2
        assert len(report.numeric_anomalies) == 1
        assert len(report.unsupported_claims) == 2  # 1 qa + 1 takeaway
        assert report.verified_overall is False
        assert report.confidence_score == 0.25

    def test_invalid_citation_does_not_trigger_number_check(self):
        transcript = "实际营收150亿"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="营收", content="营收150亿", citations=["invalid"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.citation_checks) == 1
        assert report.citation_checks[0].valid is False
        assert len(report.numeric_anomalies) == 0
