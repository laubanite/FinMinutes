import re

from finminutes.core.rewriter import chunk_text
from finminutes.core.summarizer import StructuredMinutes

_TIMESTAMP_RE = re.compile(r"(\d{2}:\d{2}:\d{2})-(\d{2}:\d{2}:\d{2})")
_TIMESTAMP_SINGLE_RE = re.compile(r"(\d{2}:\d{2}:\d{2})")
_PARAGRAPH_RE = re.compile(r"§(\d+)")
_LINE_REF_RE = re.compile(r"L(\d+)", re.IGNORECASE)
_UNKNOWN_RE = re.compile(r"×(\d+)")

# 内容覆盖检查参数
_COVERAGE_CHUNK_CHARS = 1000     # 转录文本切分粒度
_CHUNK_COVER_THRESHOLD = 0.5     # 单片得分 ≥0.5 视为已覆盖
_COVERAGE_OK_RATIO = 0.90        # 整体覆盖率阈值（低于则视为内容遗漏，可被 config coverage_threshold 覆盖）


class CitationCheck:
    def __init__(self, citation: str, line_index: int, content: str | None = None, valid: bool = False):
        self.citation = citation
        self.line_index = line_index
        self.content = content
        self.valid = valid


class CoverageReport:
    """内容完整度/覆盖率检查结果：转录文本有多大比例被 sections/QA 提取内容覆盖。"""

    def __init__(
        self,
        total_chunks: int,
        uncovered_chunks: int,
        ratio: float,
        uncovered_snippets: list[str] | None = None,
        ok_ratio: float = _COVERAGE_OK_RATIO,
    ):
        self.total_chunks = total_chunks
        self.uncovered_chunks = uncovered_chunks
        self.ratio = ratio
        self.uncovered_snippets = uncovered_snippets or []
        self.ok_ratio = _COVERAGE_OK_RATIO if ok_ratio is None else ok_ratio

    @property
    def ok(self) -> bool:
        return self.ratio >= self.ok_ratio


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
        citation_checks: list[CitationCheck] | None = None,
        coverage: CoverageReport | None = None,
        unsupported_claims: list[UnsupportedClaim] | None = None,
    ):
        self.sections_checked = sections_checked
        self.qa_pairs_checked = qa_pairs_checked
        self.citation_checks = citation_checks or []
        self.coverage = coverage
        self.unsupported_claims = unsupported_claims or []

    @property
    def verified_overall(self) -> bool:
        cov_ok = self.coverage is None or self.coverage.ok
        return cov_ok and len(self.unsupported_claims) == 0

    @property
    def confidence_score(self) -> float:
        if self.coverage is not None:
            return max(0.0, min(self.coverage.ratio, 1.0))
        total = self.sections_checked + self.qa_pairs_checked
        if total == 0:
            return 1.0
        issues = len(self.unsupported_claims)
        return max(0.0, 1.0 - issues / total)


class FactChecker:
    def __init__(self, transcript: str, minutes: StructuredMinutes):
        self._lines = transcript.split("\n")
        self._minutes = minutes
        self._citation_style = "paragraph"

    def check(self, citation_style: str = "paragraph", ok_ratio: float | None = None) -> FactCheckReport:
        self._citation_style = citation_style if citation_style in ("timestamp", "paragraph") else "paragraph"
        citation_checks: list[CitationCheck] = []
        unsupported_claims: list[UnsupportedClaim] = []

        for section in self._minutes.sections:
            if section.citations:
                for c in section.citations:
                    check = self._check_citation(c)
                    citation_checks.append(check)
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

        coverage = check_coverage("\n".join(self._lines), self._minutes, ok_ratio=ok_ratio)

        return FactCheckReport(
            sections_checked=len(self._minutes.sections),
            qa_pairs_checked=len(self._minutes.qa_pairs),
            citation_checks=citation_checks,
            coverage=coverage,
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


def check_coverage(
    transcript: str, minutes: StructuredMinutes, ok_ratio: float | None = None
) -> CoverageReport:
    """内容覆盖检查：按 chunk 粒度确认转录文本被 sections/QA 提取内容覆盖的比例。

    启发式（确定性、零 LLM 调用，对书面化改写鲁棒）：
    - 对每片转录提取「数字/金额/百分比」token（归一化：展开范围、2 位年份补 20xx），
      与文本池数字集求交集得数字保留率——金融纪要里数字就是事实，最不该丢。
    - 数字较少的片段退回字符 bigram 命中率（词汇重叠，容忍改写措辞）。
    - 单片得分 = max(数字保留率, bigram 命中率)，< 阈值（默认 0.5）计为未覆盖。
    - 整体覆盖率 = 各片得分均值；< ok_ratio（默认 0.90，可配置）提示可能存在内容遗漏。
    """
    chunks = chunk_text(transcript or "", max_chars=_COVERAGE_CHUNK_CHARS, overlap=100)
    non_empty = [c for c in chunks if c.strip()]
    if not non_empty:
        return CoverageReport(0, 0, 1.0, ok_ratio=ok_ratio)

    pool = _pool_text(minutes)
    if not pool.strip():
        return CoverageReport(len(non_empty), len(non_empty), 0.0, list(non_empty), ok_ratio=ok_ratio)

    pool_norm = _normalize_match(pool)
    pool_bigrams = _char_bigrams(pool_norm)
    pool_numbers = _number_tokens(pool_norm)

    uncovered: list[str] = []
    scores: list[float] = []
    for c in non_empty:
        n = _normalize_match(c)
        if len(n) < 10:
            scores.append(1.0)
            continue
        bigram_ratio = _bigram_ratio(n, pool_bigrams)
        num_ratio = _number_ratio(n, pool_numbers)
        score = max(bigram_ratio, num_ratio)
        scores.append(score)
        if score < _CHUNK_COVER_THRESHOLD:
            uncovered.append(c)

    ratio = sum(scores) / len(scores) if scores else 0.0
    return CoverageReport(len(non_empty), len(uncovered), ratio, list(uncovered), ok_ratio=ok_ratio)


def _pool_text(minutes: StructuredMinutes) -> str:
    parts = []
    for s in minutes.sections:
        parts.append(s.content or "")
    for q in minutes.qa_pairs:
        parts.append(q.question or "")
        parts.append(q.answer or "")
    return "\n".join(parts)


def _normalize_match(text: str) -> str:
    """去标点/空白/语气助词，只留中文、字母、数字与 %/-，便于跨改写措辞比对。"""
    return re.sub(r"[^一-鿿A-Za-z0-9%\-]", "", text)


def _char_bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _bigram_ratio(text: str, pool_bigrams: set[str]) -> float:
    bg = _char_bigrams(text)
    if not bg:
        return 1.0
    return len(bg & pool_bigrams) / len(bg)


def _number_tokens(text: str) -> set[str]:
    """提取数字 token：带单位/百分号优先，其次裸数字；展开范围、2 位年份补 20xx。"""
    toks: set[str] = set()
    # 年份归一化："26年" 与 "2026年" 双向对齐（转录常写 2 位，成果稿常写 4 位）
    for m in re.finditer(r"(\d{2,4})\s*年", text):
        y = m.group(1)
        toks.add(y)
        if len(y) == 2:
            toks.add("20" + y)
        elif len(y) == 4:
            toks.add(y[2:])
    # 带单位/百分号的数字：1.8亿 / 300万 / 30% / 1.2-1.5亿
    for m in re.finditer(r"\d+(?:\.\d+)?\s*[-~～]+\s*\d+(?:\.\d+)?\s*[万亿%]", text):
        range_full = m.group(0)
        unit = range_full[-1]
        for part in re.split(r"[-~～]+", range_full[:-1]):
            part = part.strip()
            if part:
                toks.add(part + unit)
    for m in re.finditer(r"\d+(?:\.\d+)?\s*[万亿%]", text):
        toks.add(re.sub(r"\s+", "", m.group(0)))
    for m in re.finditer(r"\d+(?:\.\d+)?", text):
        toks.add(m.group(0))
    return toks


def _number_ratio(text: str, pool_numbers: set[str]) -> float:
    """文本中数字被成果文本池覆盖的比例；无数字返回 0.0（交由 bigram 兜底）。"""
    nums = _number_tokens(text)
    if not nums:
        return 0.0
    return len(nums & pool_numbers) / len(nums)
