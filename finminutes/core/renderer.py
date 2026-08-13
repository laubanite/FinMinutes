from finminutes.core.fact_checker import FactCheckReport
from finminutes.core.summarizer import StructuredMinutes


class MarkdownRenderer:
    def __init__(self, minutes: StructuredMinutes, report: FactCheckReport | None = None):
        self._minutes = minutes
        self._report = report

    def render(self) -> str:
        parts = ["# 会议纪要\n"]

        for i, section in enumerate(self._minutes.sections, 1):
            parts.append(self._render_section(i, section))

        if self._minutes.qa_pairs:
            parts.append(self._render_qa_section())

        if self._report is not None:
            parts.append(self._render_validation_report())

        return "\n".join(parts)

    @staticmethod
    def _render_section(index: int, section) -> str:
        lines = [f"## {index}. {section.title}"]
        if section.citations:
            refs = " ".join(f"({c})" for c in section.citations)
            lines[-1] += f" {refs}"
        lines.append("")
        lines.append(section.content or "（无内容）")
        lines.append("")
        return "\n".join(lines)

    def _render_qa_section(self) -> str:
        lines = ["---", "", "## Q&A 环节", ""]
        for qa in self._minutes.qa_pairs:
            asker = qa.asker or "（未指明）"
            lines.append(f"- **{asker}**：{qa.question}")
            if qa.answer:
                lines.append(f"  - {qa.answer}")
            lines.append("")
        return "\n".join(lines)

    def _render_validation_report(self) -> str:
        r = self._report
        lines = ["---", "", "## 事实校验报告", ""]

        status = "[通过]" if r.verified_overall else "[异常]"
        lines.append(f"- **校验状态**：{status}")
        if r.coverage is not None:
            covered = r.coverage.total_chunks - r.coverage.uncovered_chunks
            lines.append(f"- **内容覆盖率**：{r.coverage.ratio * 100:.0f}%（{covered}/{r.coverage.total_chunks} 段被提取内容覆盖）")
            if not r.coverage.ok:
                lines.append(f"- ⚠️ 覆盖率低于阈值 {r.coverage.ok_ratio * 100:.0f}%，可能存在内容遗漏，请核对以下未覆盖片段：")
                for snip in r.coverage.uncovered_snippets[:5]:
                    lines.append(f"  - \"{snip[:80]}\"")
        else:
            lines.append(f"- **置信度**：{r.confidence_score * 100:.0f}%")
        lines.append(f"- 共检查 {r.sections_checked} 个章节、{r.qa_pairs_checked} 个问答")
        lines.append("")

        if r.citation_checks:
            lines.append("### 引用检查")
            for c in r.citation_checks:
                if c.valid:
                    if c.content:
                        snippet = c.content[:50]
                        lines.append(f"- [通过] `{c.citation}` 对应原文：{snippet}")
                    else:
                        lines.append(f"- [通过] `{c.citation}`（来源标记）")
                else:
                    lines.append(f"- [无效] `{c.citation}` 引用格式不识别")
            lines.append("")

        if r.unsupported_claims:
            lines.append("### 无来源声明")
            for u in r.unsupported_claims:
                snippet = u.claim_snippet[:60]
                lines.append(f"- [待确认] [{u.claim_type}] {u.title}：{u.issue}（\"{snippet}\"）")
            lines.append("")

        return "\n".join(lines)



