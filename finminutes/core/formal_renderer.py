from finminutes.core.fact_checker import FactCheckReport
from finminutes.core.models import Template
from finminutes.core.summarizer import StructuredMinutes, SectionContent

SYSTEM_PROMPT = """你是一个专业的金融会议纪要精炼专家。

你的核心任务：将校验稿中的 Q&A 数据，转化为书面化、结构化、可直接交付的成品稿。


【你的能力边界】

- 你控制"怎么说"（表达方式、语言风格、组织结构）
- 你不控制"说什么"（内容来自校验稿 Q&A，不能编造或丢失）


【强制约束（必须遵守）】

1. 事实保真：不能添加校验稿中不存在的信息
2. 内容完整：校验稿中的每一个问答对都必须保留
3. 数值准确：所有百分比、金额、年份等数值必须原样呈现
4. 禁止编造：不要添加任何原文中没有的信息
5. 问答数量守恒：校验稿中有 N 个问答对，成品稿中必须有 N 个


【凝练 vs 概括（必须区分）】

"凝练"是用更精炼的语言表达同样的内容。
"概括"是用更少的篇幅表达更少的内容。

你需要做的是"凝练"，不是"概括"。

✅ 凝练的操作：
- 删除口语化填充词（"嗯"、"那个"、"就是"、"吧"等）
- 合并重复表达（同一意思在上下文中反复出现时，只保留一次）
- 精简冗余句式（"在这样一个情况之下" → "在此情况下"）
- 使用书面化词汇替代口头化表达（"弄" → "进行/实施"）

❌ 凝练的边界（禁止的操作）：
- 禁止浓缩内容：不能把多条信息合并成一条后丢失信息
- 禁止删除限定词：条件、程度、范围（"大概"、"约"、"可能"）必须保留
- 禁止简化数据：数值、百分比、年份必须原样呈现
- 禁止压缩问答数量：校验稿中有 N 个问答对，成品稿中必须有 N 个


【输出结构】

- 当模板提供了 sections 时，按模板的章节结构和顺序输出
- 每个章节的内容，根据该章节的 prompt 要求加工


【自检】

生成完成后，确认：
1. 问答对数量是否与校验稿一致？
2. 每个数值是否都在成品稿中出现了？
3. 是否有任何信息被遗漏？

如有遗漏，重新生成。"""


def _format_qa_pairs(qa_pairs: list) -> str:
    lines = []
    for i, qa in enumerate(qa_pairs, 1):
        ts = qa.get("timerange", "")
        q = qa.get("question", "")
        a = qa.get("answer", "")
        lines.append(f"[{i}] timerange: {ts}")
        lines.append(f"    Q：{q}")
        lines.append(f"    A：{a}")
        lines.append("")
    return "\n".join(lines)


def _format_template_sections(template: Template) -> str:
    lines = []
    for sec in template.sections:
        lines.append(f"## {sec.title}")
        if sec.prompt:
            lines.append(f"   prompt: {sec.prompt}")
        lines.append("")
    return "\n".join(lines)


class FormalGenerator:
    def __init__(self, llm_client):
        self._llm = llm_client

    def generate(self, qa_pairs: list, template: Template) -> str:
        qa_text = _format_qa_pairs(qa_pairs)
        sections_text = _format_template_sections(template)

        user_prompt = f"""校验稿中的 Q&A 数据（共 {len(qa_pairs)} 组）：

{qa_text}
模板要求：

{sections_text}
请按照系统指令的要求，生成书面化、凝练的成品稿。"""

        result = self._llm.generate(prompt=user_prompt, system=SYSTEM_PROMPT)
        return result.strip() or "（生成失败）"


def _build_minutes_from_data(data: dict) -> StructuredMinutes:
    sections = []
    for s in data.get("sections", []):
        sections.append(
            SectionContent(
                title=s.get("title", ""),
                content=s.get("content", ""),
                citations=s.get("citations", []),
            )
        )
    from finminutes.core.summarizer import QAPair, Takeaway
    qa_pairs = [
        QAPair(question=q.get("question", ""), answer=q.get("answer", ""), asker=q.get("asker", ""))
        for q in data.get("qa_pairs", [])
    ]
    takeaways = [
        Takeaway(content=t.get("content", ""), type=t.get("type", "事实"))
        for t in data.get("takeaways", [])
    ]
    return StructuredMinutes(sections=sections, qa_pairs=qa_pairs, takeaways=takeaways)


def _build_report_from_data(data: dict) -> FactCheckReport | None:
    fc = data.get("fact_check")
    if not fc:
        return None
    from finminutes.core.fact_checker import FactCheckReport, CitationCheck, NumericAnomaly, UnsupportedClaim
    report = FactCheckReport(
        sections_checked=fc.get("sections_checked", 0),
        qa_pairs_checked=fc.get("qa_pairs_checked", 0),
        takeaways_checked=fc.get("takeaways_checked", 0),
    )
    for c in fc.get("citation_checks", []):
        report.citation_checks.append(
            CitationCheck(citation=c.get("citation", ""), line_index=-1, valid=c.get("valid", False))
        )
    for a in fc.get("numeric_anomalies", []):
        report.numeric_anomalies.append(
            NumericAnomaly(value_in_minutes=a.get("value", ""), title=a.get("title", ""), citation="")
        )
    for u in fc.get("unsupported_claims", []):
        report.unsupported_claims.append(
            UnsupportedClaim(claim_snippet=u.get("snippet", ""), claim_type=u.get("type", ""), title=u.get("title", ""), issue=u.get("issue", ""))
        )
    return report


class FormalRenderer:
    def __init__(self, template: Template):
        self._template = template

    def render(self, minutes: StructuredMinutes, report: FactCheckReport | None = None) -> str:
        parts = [f"# {self._template.description or '会议纪要'}\n"]

        for i, tmpl_sec in enumerate(self._template.sections, 1):
            matched = self._find_section(minutes, tmpl_sec)
            header = f"## {i}. {tmpl_sec.title}"
            if matched and matched.citations:
                clean_citations = [
                    c for c in matched.citations
                    if not c.startswith("\u00d7")
                ]
                if clean_citations:
                    refs = " ".join(f"({c})" for c in clean_citations)
                    header += f" {refs}"
            parts.append(header)
            parts.append("")
            if matched and matched.content:
                parts.append(matched.content)
            else:
                parts.append("（无内容）")
            parts.append("")

        if self._template_needs_qa():
            qa_section = self._render_qa_section(minutes)
            if qa_section:
                parts.append(qa_section)

        if self._template_needs_takeaways():
            tk_section = self._render_takeaways_section(minutes)
            if tk_section:
                parts.append(tk_section)

        return "\n".join(parts).strip()

    def _find_section(self, minutes: StructuredMinutes, tmpl_sec) -> SectionContent | None:
        for s in minutes.sections:
            if s.title == tmpl_sec.title:
                return s
        for s in minutes.sections:
            if tmpl_sec.title in s.title or s.title in tmpl_sec.title:
                return s
        return None

    def _template_needs_qa(self) -> bool:
        titles = {s.title for s in self._template.sections}
        return "Q&A" in " ".join(titles) or "问答" in " ".join(titles)

    def _template_needs_takeaways(self) -> bool:
        titles = {s.title for s in self._template.sections}
        return "总结" in " ".join(titles) or "结论" in " ".join(titles) or "要点" in " ".join(titles)

    def _render_qa_section(self, minutes: StructuredMinutes) -> str:
        if not minutes.qa_pairs:
            return ""
        lines = ["---", "", "## Q&A 环节", ""]
        for qa in minutes.qa_pairs:
            asker = qa.asker or "（未指明）"
            lines.append(f"- **{asker}**：{qa.question}")
            if qa.answer:
                lines.append(f"  - {qa.answer}")
            lines.append("")
        return "\n".join(lines)

    def _render_takeaways_section(self, minutes: StructuredMinutes) -> str:
        if not minutes.takeaways:
            return ""
        lines = ["---", "", "## 核心要点", ""]
        for t in minutes.takeaways:
            if t.content:
                lines.append(f"- [{t.type}] {t.content}")
        lines.append("")
        return "\n".join(lines)
