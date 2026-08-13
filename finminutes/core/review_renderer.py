import os
import re

import yaml

from finminutes.core.fact_checker import FactCheckReport
from finminutes.core.summarizer import StructuredMinutes, QAPair

# 仅渲染形如 "09:32" / "09:32:15" / "09:32-09:35" 的 timerange；
# LLM 在转录无时间戳时会脑补 "当前"、"2023-2028" 等垃圾值，一律不渲染。
_TIMESTAMP_RE = re.compile(
    r"^\d{1,2}:\d{2}(?::\d{2})?(?:\s*[-~～至]\s*\d{1,2}:\d{2}(?::\d{2})?)?$"
)


def _load_review_template() -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "templates", "review", "default.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _serialize_minutes(minutes: StructuredMinutes) -> dict:
    data = {
        "qa_pairs": [
            {"question": q.question, "answer": q.answer, "asker": q.asker, "timerange": q.timerange}
            for q in minutes.qa_pairs
        ],
    }
    if minutes.sections:
        data["sections"] = [
            {"title": s.title, "content": s.content, "citations": s.citations}
            for s in minutes.sections
        ]
    return data


class ReviewRenderer:
    def __init__(self, minutes: StructuredMinutes, report: FactCheckReport | None = None, transcript_raw: str = ""):
        self._minutes = minutes
        self._report = report
        self._transcript_raw = transcript_raw
        self._tmpl = _load_review_template()

    def render(self) -> str:
        front_matter = self._build_front_matter()
        body = self._build_body()
        return front_matter + "\n" + body

    def _build_front_matter(self) -> str:
        data = _serialize_minutes(self._minutes)
        yaml_str = yaml.safe_dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()
        return f"---\n{yaml_str}\n---"

    def _build_body(self) -> str:
        lines = ["", "# 校验稿", ""]
        self._render_sections(lines)
        self._render_qa(lines)
        return "\n".join(lines).strip()

    def _render_sections(self, lines: list):
        if not self._minutes.sections:
            return
        lines.append("## 主题要点")
        lines.append("")
        for sec in self._minutes.sections:
            if sec.title:
                lines.append(f"### {sec.title}")
            if sec.content:
                lines.append(sec.content)
            else:
                lines.append("（无内容）")
            lines.append("")

    def _render_qa(self, lines: list):
        for qa in self._minutes.qa_pairs:
            ts = qa.timerange.strip() if qa.timerange else ""
            if ts and _TIMESTAMP_RE.match(ts):
                lines.append(f"[{ts}]")
                lines.append("")

            lines.append(f"**Q**：{qa.question}")
            if qa.answer:
                lines.append(f"**A**：{qa.answer}")
            lines.append("")
