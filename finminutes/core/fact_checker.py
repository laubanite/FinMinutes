import re

from finminutes.core.summarizer import StructuredMinutes

_TIMESTAMP_RE = re.compile(r"(\d{2}:\d{2}:\d{2})-(\d{2}:\d{2}:\d{2})")
_TIMESTAMP_SINGLE_RE = re.compile(r"(\d{2}:\d{2}:\d{2})")
_PARAGRAPH_RE = re.compile(r"§(\d+)")
_LINE_REF_RE = re.compile(r"L(\d+)", re.IGNORECASE)
_UNKNOWN_RE = re.compile(r"×(\d+)")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")


class CitationCheck:
    def __init__(self, citation: str, line_index: int, content: str | None = None, valid: bool = False):
        self.citation = citation
        self.line_index = line_index
        self.content = content
        self.valid = valid


class NumericAnomaly:
    def __init__(self, value_in_minutes: str, title: str, citation: str, source_text: str = ""):
        self.value_in_minutes = value_in_minutes
        self.title = title
        self.citation = citation
        self.source_text = source_text


class UnsupportedClaim:
    def __init__(self, claim_snippet: str, claim_type: str, title: str, issue: str):
        self.claim_snippet = claim_snippet
        self.claim_type = claim_type
        self.title = title
        self.issue = issue


class FactCheckReport:
    def __init__(
        self,
        sections_checked: int = 0,
        qa_pairs_checked: int = 0,
        takeaways_checked: int = 0,
        citation_checks: list[CitationCheck] | None = None,
        numeric_anomalies: list[NumericAnomaly] | None = None,
        unsupported_claims: list[UnsupportedClaim] | None = None,
    ):
        self.sections_checked = sections_checked
        self.qa_pairs_checked = qa_pairs_checked
        self.takeaways_checked = takeaways_checked
        self.citation_checks = citation_checks or []
        self.numeric_anomalies = numeric_anomalies or []
        self.unsupported_claims = unsupported_claims or []

    @property
    def verified_overall(self) -> bool:
        return len(self.numeric_anomalies) == 0 and len(self.unsupported_claims) == 0

    @property
    def confidence_score(self) -> float:
        total = self.sections_checked + self.qa_pairs_checked + self.takeaways_checked
        if total == 0:
            return 1.0
        issues = len(self.numeric_anomalies) + len(self.unsupported_claims)
        return max(0.0, 1.0 - issues / total)


class FactChecker:
    def __init__(self, transcript: str, minutes: StructuredMinutes):
        self._lines = transcript.split("\n")
        self._minutes = minutes
        self._citation_style = "paragraph"

    def check(self, citation_style: str = "paragraph") -> FactCheckReport:
        self._citation_style = citation_style if citation_style in ("timestamp", "paragraph") else "paragraph"
        citation_checks: list[CitationCheck] = []
        numeric_anomalies: list[NumericAnomaly] = []
        unsupported_claims: list[UnsupportedClaim] = []

        for section in self._minutes.sections:
            if section.citations:
                for c in section.citations:
                    check = self._check_citation(c)
                    citation_checks.append(check)
                    if check.valid and check.content:
                        content_text = section.content or ""
                        anomalies = self._check_numbers(
                            content_text, [check.content], section.title, c
                        )
                        numeric_anomalies.extend(anomalies)
            else:
                issue = "无引用来源"
                if not section.content:
                    issue = "内容为空且无引用"
                unsupported_claims.append(
                    UnsupportedClaim(
                        claim_snippet=(section.content or "")[:100],
                        claim_type="section",
                        title=section.title,
                        issue=issue,
                    )
                )

        for qa in self._minutes.qa_pairs:
            text = f"{qa.question} {qa.answer}".strip()
            if text:
                unsupported_claims.append(
                    UnsupportedClaim(
                        claim_snippet=text[:100],
                        claim_type="qa_pair",
                        title=f"Q: {qa.question[:50]}",
                        issue="无引用来源",
                    )
                )

        for t in self._minutes.takeaways:
            if t.content:
                unsupported_claims.append(
                    UnsupportedClaim(
                        claim_snippet=t.content[:100],
                        claim_type="takeaway",
                        title=t.type,
                        issue="无引用来源",
                    )
                )

        return FactCheckReport(
            sections_checked=len(self._minutes.sections),
            qa_pairs_checked=len(self._minutes.qa_pairs),
            takeaways_checked=len(self._minutes.takeaways),
            citation_checks=citation_checks,
            numeric_anomalies=numeric_anomalies,
            unsupported_claims=unsupported_claims,
        )

    def _check_citation(self, citation: str) -> CitationCheck:
        if self._citation_style == "timestamp":
            return self._check_timestamp_citation(citation)
        return self._check_paragraph_citation(citation)

    def _check_timestamp_citation(self, citation: str) -> CitationCheck:
        match = _TIMESTAMP_RE.match(citation)
        if match:
            ts_start, ts_end = match.group(1), match.group(2)
            return CitationCheck(
                citation=citation,
                line_index=-1,
                content=self._find_timestamp_line(ts_start),
                valid=True,
            )
        single = _TIMESTAMP_SINGLE_RE.match(citation)
        if single:
            return CitationCheck(
                citation=citation,
                line_index=-1,
                content=self._find_timestamp_line(single.group(1)),
                valid=True,
            )
        return CitationCheck(citation=citation, line_index=-1, valid=False)

    def _check_paragraph_citation(self, citation: str) -> CitationCheck:
        match = _PARAGRAPH_RE.match(citation)
        if match:
            para_idx = int(match.group(1))
            paras = self._get_paragraphs()
            if 1 <= para_idx <= len(paras):
                return CitationCheck(
                    citation=citation,
                    line_index=para_idx,
                    content=paras[para_idx - 1].strip()[:200],
                    valid=True,
                )
            return CitationCheck(citation=citation, line_index=para_idx, valid=False)
        line_ref = _LINE_REF_RE.match(citation)
        if line_ref:
            idx = int(line_ref.group(1))
            if 1 <= idx <= len(self._lines):
                return CitationCheck(
                    citation=citation,
                    line_index=idx,
                    content=self._lines[idx - 1].strip(),
                    valid=True,
                )
            return CitationCheck(citation=citation, line_index=idx, valid=False)
        unknown = _UNKNOWN_RE.match(citation)
        if unknown:
            return CitationCheck(
                citation=citation,
                line_index=0,
                content="",
                valid=True,
            )
        return CitationCheck(citation=citation, line_index=-1, valid=False)

    def _find_timestamp_line(self, ts: str) -> str:
        for line in self._lines:
            if ts in line:
                return line.strip()[:200]
        return ""

    def _get_paragraphs(self) -> list[str]:
        text = "\n".join(self._lines)
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paras or self._lines

    @staticmethod
    def _check_numbers(content: str, source_lines: list[str], title: str, citation: str) -> list[NumericAnomaly]:
        anomalies: list[NumericAnomaly] = []
        minutes_nums = _NUMBER_RE.findall(content)
        if not minutes_nums:
            return anomalies
        source_text = " ".join(source_lines)
        for num in minutes_nums:
            if num not in source_text:
                anomalies.append(
                    NumericAnomaly(
                        value_in_minutes=num,
                        title=title,
                        citation=citation,
                        source_text=source_text[:100],
                    )
                )
        return anomalies
