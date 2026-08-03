import os
import re

import yaml

from finminutes.core.fact_checker import FactCheckReport
from finminutes.core.summarizer import StructuredMinutes, QAPair


def _load_review_template() -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "templates", "review", "default.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _serialize_minutes(minutes: StructuredMinutes) -> dict:
    return {
        "qa_pairs": [
            {"question": q.question, "answer": q.answer, "asker": q.asker, "timerange": q.timerange}
            for q in minutes.qa_pairs
        ],
    }


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
        self._render_qa(lines)
        return "\n".join(lines).strip()

    def _render_qa(self, lines: list):
        for qa in self._minutes.qa_pairs:
            ts = qa.timerange
            if ts:
                lines.append(f"[{ts}]")
                lines.append("")

            lines.append(f"**Q**：{qa.question}")
            if qa.answer:
                annotated = self._annotate_numbers(qa.answer)
                lines.append(f"**A**：{annotated}")
            lines.append("")

    def _annotate_numbers(self, text: str) -> str:
        transcript = self._transcript_raw or ""
        def _replace(m):
            num = m.group(0)
            start, end = m.start(), m.end()
            if (start > 0 and re.match(r'\w', text[start-1])) or \
               (end < len(text) and re.match(r'\w', text[end])):
                return num
            if num in transcript:
                return num
            return f"**{num}** ❓[待确认]"
        return re.sub(r"\d+(?:[.,]\d+)*%?", _replace, text)
