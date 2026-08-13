from finminutes.core.fact_checker import (
    CitationCheck,
    CoverageReport,
    FactChecker,
    FactCheckReport,
    UnsupportedClaim,
    _bigram_ratio,
    _char_bigrams,
    _number_ratio,
    _number_tokens,
    check_coverage,
)
from finminutes.core.summarizer import QAPair, SectionContent, StructuredMinutes


class TestDataClasses:
    def test_citation_check_defaults(self):
        c = CitationCheck(citation="L1", line_index=1, content="some text", valid=True)
        assert c.citation == "L1"
        assert c.line_index == 1
        assert c.content == "some text"
        assert c.valid is True

    def test_coverage_report_defaults(self):
        c = CoverageReport(total_chunks=10, uncovered_chunks=2, ratio=0.95)
        assert c.total_chunks == 10
        assert c.uncovered_chunks == 2
        assert c.ratio == 0.95
        assert c.uncovered_snippets == []
        assert c.ok is True
        assert c.ok_ratio == 0.90  # 默认阈值

    def test_coverage_report_not_ok(self):
        c = CoverageReport(total_chunks=10, uncovered_chunks=7, ratio=0.3)
        assert c.ok is False

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
        assert r.citation_checks == []
        assert r.coverage is None
        assert r.unsupported_claims == []

    def test_report_verified_with_no_issues(self):
        r = FactCheckReport(sections_checked=2, qa_pairs_checked=1,
                            coverage=CoverageReport(10, 0, 1.0))
        assert r.verified_overall is True
        assert r.confidence_score == 1.0

    def test_report_not_verified_with_low_coverage(self):
        r = FactCheckReport(coverage=CoverageReport(10, 8, 0.2))
        assert r.verified_overall is False
        assert r.confidence_score == 0.2

    def test_report_confidence_score_unsupported(self):
        r = FactCheckReport(
            sections_checked=3,
            qa_pairs_checked=1,
            unsupported_claims=[UnsupportedClaim("t", "section", "事实", "无引用")],
        )
        assert r.confidence_score == 0.75

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

    def test_citation_out_of_bounds(self):
        transcript = "line one"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="text", citations=["L99"])],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert report.citation_checks[0].valid is False
        assert report.citation_checks[0].line_index == 99

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


class TestCoverage:
    def test_full_verbatim_coverage(self):
        transcript = "公司营收100万元。净利润50万元。毛利率30%。"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content=transcript, citations=["L1"])],
        )
        report = check_coverage(transcript, minutes)
        assert report.ratio == 1.0
        assert report.uncovered_chunks == 0
        assert report.ok is True

    def test_empty_pool_no_coverage(self):
        transcript = "这是一段完全没有被提取内容覆盖的转录文本内容。"
        minutes = StructuredMinutes()  # 无 sections / qa
        report = check_coverage(transcript, minutes)
        assert report.ratio == 0.0
        assert report.uncovered_chunks == report.total_chunks
        assert report.ok is False

    def test_empty_transcript(self):
        report = check_coverage("", StructuredMinutes())
        assert report.total_chunks == 0
        assert report.ratio == 1.0

    def test_omitted_content_zero_coverage(self):
        # 提取内容与转录完全不同 → 该片未覆盖
        transcript = "甲" * 500
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content="乙" * 200)],
        )
        report = check_coverage(transcript, minutes)
        assert report.ratio == 0.0
        assert report.uncovered_chunks == 1
        assert report.ok is False

    def test_missing_tail_lowers_ratio(self):
        # 覆盖首段、遗漏大段尾内容 → 覆盖率低
        covered = "甲" * 700 + "。"
        omitted = "乙" * 8000
        transcript = covered + omitted
        minutes = StructuredMinutes(
            sections=[SectionContent(title="S1", content=covered)],
        )
        report = check_coverage(transcript, minutes)
        assert report.uncovered_chunks >= 1
        assert report.ratio < 0.60
        assert report.ok is False

    def test_bigram_ratio_full(self):
        assert _bigram_ratio("甲乙丙丁甲乙丙丁", set(_char_bigrams("甲乙丙丁甲乙丙丁内容"))) == 1.0

    def test_bigram_ratio_zero(self):
        assert _bigram_ratio("甲乙丙丁", set(_char_bigrams("子丑寅卯"))) == 0.0

    def test_number_tokens_units_and_years(self):
        toks = _number_tokens("营收1.8亿，毛利30%，目标26年上市，扩产至8000万")
        assert "1.8亿" in toks
        assert "30%" in toks
        assert "26" in toks and "2026" in toks  # 2 位年份补 20xx
        assert "8000万" in toks

    def test_number_ratio_rewritten_tolerated(self):
        # 书面化改写后措辞变化，但数字保留 → 数字保留率高
        pool = _number_tokens("公司营收1.8亿元，毛利率30%，计划2026年上市")
        assert _number_ratio("我们营收大概1.8个亿，毛利是30%，26年要上市", pool) == 1.0

    def test_number_ratio_missing_lowers(self):
        pool = _number_tokens("公司营收1.8亿元，毛利率30%")
        assert _number_ratio("我们营收大概1.8个亿，然后毛利是30%，另外一家规模5000万", pool) < 1.0


class TestFactChecker:
    def test_check_empty_minutes(self):
        transcript = "some transcript"
        minutes = StructuredMinutes()
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert report.sections_checked == 0
        assert report.qa_pairs_checked == 0
        # 无任何提取内容 → 覆盖率 0，未通过
        assert report.coverage is not None
        assert report.verified_overall is False
        assert report.confidence_score == 0.0

    def test_all_valid_returns_verified(self):
        transcript = "2024年营收300亿。\n净利润50亿。"
        minutes = StructuredMinutes(
            sections=[
                SectionContent(title="业绩", content="2024年营收300亿。", citations=["L1"]),
                SectionContent(title="利润", content="净利润50亿。", citations=["L2"]),
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

    def test_qa_pairs_flagged_as_unsupported(self):
        transcript = "some text"
        minutes = StructuredMinutes(qa_pairs=[QAPair(question="Q1", answer="A1", asker="张三")])
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert len(report.unsupported_claims) == 1
        assert report.unsupported_claims[0].claim_type == "qa_pair"

    def test_mixed_results(self):
        transcript = "实际营收150亿。\n实际净利润30亿。"
        minutes = StructuredMinutes(
            sections=[SectionContent(title="营收", content="实际营收150亿。", citations=["L1"])],
            qa_pairs=[QAPair(question="Q?", answer="实际净利润30亿。")],
        )
        fc = FactChecker(transcript, minutes)
        report = fc.check()
        assert report.sections_checked == 1
        assert report.qa_pairs_checked == 1
        assert len(report.citation_checks) == 1
        assert report.coverage is not None
        assert len(report.unsupported_claims) == 1  # qa 无引用
