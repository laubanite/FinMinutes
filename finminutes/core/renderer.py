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

        if self._minutes.takeaways:
            parts.append(self._render_takeaways())

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

    def _render_takeaways(self) -> str:
        lines = ["---", "", "## 核心要点", ""]
        for t in self._minutes.takeaways:
            if t.content:
                lines.append(f"- [{t.type}] {t.content}")
        lines.append("")
        return "\n".join(lines)

    def _render_validation_report(self) -> str:
        r = self._report
        lines = ["---", "", "## 事实校验报告", ""]

        score_pct = f"{r.confidence_score * 100:.0f}%"
        status = "[通过]" if r.verified_overall else "[异常]"
        lines.append(f"- **校验状态**：{status}")
        lines.append(f"- **置信度**：{score_pct}")
        lines.append(f"- 共检查 {r.sections_checked} 个章节、{r.qa_pairs_checked} 个问答、{r.takeaways_checked} 个要点")
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

        if r.numeric_anomalies:
            lines.append("### 数值异常")
            for a in r.numeric_anomalies:
                lines.append(f"- [异常] 内容中的「{a.value_in_minutes}」未在 {a.citation} 的源文本中找到")
            lines.append("")

        if r.unsupported_claims:
            lines.append("### 无来源声明")
            for u in r.unsupported_claims:
                snippet = u.claim_snippet[:60]
                lines.append(f"- [待确认] [{u.claim_type}] {u.title}：{u.issue}（\"{snippet}\"）")
            lines.append("")

        return "\n".join(lines)



