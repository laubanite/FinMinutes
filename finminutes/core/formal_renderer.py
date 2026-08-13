from finminutes.core.fact_checker import FactCheckReport
from finminutes.core.models import Template
from finminutes.core.summarizer import StructuredMinutes, SectionContent

SYSTEM_PROMPT = """你是一个专业的金融会议纪要精炼专家。

你的核心任务：将校验稿中的 Q&A 数据和主题要点（独白内容），转化为书面化、结构化、可直接交付的成品稿。


【你的能力边界】

- 你控制"怎么说"（表达方式、语言风格、组织结构）
- 你不控制"说什么"（内容来自校验稿 Q&A 与主题要点，不能编造或丢失）


【强制约束（必须遵守）】

1. 事实保真：不能添加校验稿中不存在的信息
2. 内容完整：校验稿中的每一个问答对都必须保留
3. 数值准确：所有百分比、金额、年份等数值必须原样呈现
4. 禁止编造：不要添加任何原文中没有的信息
5. 问答数量守恒：校验稿中有 N 个问答对，成品稿中必须有 N 个
6. 主题要点守恒：校验稿中每一条主题要点（独白内容块）都必须呈现在成品稿对应章节，不允许丢弃；若模板没有对应章节，归入最相近的章节
7. 成品稿一律使用简体中文，禁止繁体字


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
- 禁止压缩主题要点：校验稿中有 M 条主题要点，成品稿中必须全部体现


【输出结构】

- 当模板提供了 sections 时，按模板的章节结构和顺序输出
- 每个章节的内容，根据该章节的 prompt 要求加工
- 若校验稿只有主题要点、没有问答（纯独白访谈），成品稿以主题要点为章节素材组织
- 若校验稿只有问答、没有主题要点（纯对话访谈），成品稿以问答为素材组织


【自检】

生成完成后，确认：
1. 问答对数量是否与校验稿一致？
2. 每一条主题要点是否都进入了成品稿？
3. 每个数值是否都在成品稿中出现了？
4. 是否有任何信息被遗漏？

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


def _format_sections(sections: list) -> str:
    """把校验稿主题要点（独白内容块）格式化为 LLM 输入。"""
    lines = []
    for i, sec in enumerate(sections, 1):
        title = sec.title or "（无标题）"
        content = (sec.content or "").strip()
        lines.append(f"[{i}] 主题：{title}")
        if content:
            lines.append(f"    {content}")
        lines.append("")
    return "\n".join(lines).strip() or "（无）"


class FormalGenerator:
    def __init__(self, llm_client):
        self._llm = llm_client

    def generate(self, qa_pairs: list, template: Template, sections: list | None = None) -> str:
        sections = sections or []
        qa_text = _format_qa_pairs(qa_pairs) if qa_pairs else "（无问答，纯独白访谈）"
        points_text = _format_sections(sections) if sections else "（无主题要点，纯对话访谈）"
        sections_text = _format_template_sections(template)

        user_prompt = f"""校验稿中的 Q&A 数据（共 {len(qa_pairs)} 组）：

{qa_text}
校验稿中的主题要点（独白内容，共 {len(sections)} 条）：

{points_text}
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
    from finminutes.core.summarizer import QAPair
    qa_pairs = [
        QAPair(question=q.get("question", ""), answer=q.get("answer", ""), asker=q.get("asker", ""))
        for q in data.get("qa_pairs", [])
    ]
    return StructuredMinutes(sections=sections, qa_pairs=qa_pairs)


def _build_report_from_data(data: dict) -> FactCheckReport | None:
    fc = data.get("fact_check")
    if not fc:
        return None
    from finminutes.core.fact_checker import FactCheckReport, CitationCheck, NumericAnomaly, UnsupportedClaim
    report = FactCheckReport(
        sections_checked=fc.get("sections_checked", 0),
        qa_pairs_checked=fc.get("qa_pairs_checked", 0),
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
