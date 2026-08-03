import json
import re
import logging

logger = logging.getLogger(__name__)

from finminutes.core.models import Template

_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)

EXTRACT_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class QAPair:
    def __init__(self, question: str = "", answer: str = "", asker: str = "", timerange: str = ""):
        self.question = question
        self.answer = answer
        self.asker = asker
        self.timerange = timerange


class Takeaway:
    def __init__(self, content: str = "", type: str = "事实"):
        self.content = content
        self.type = type


class SectionContent:
    def __init__(self, title: str = "", content: str = "", citations: list[str] | None = None):
        self.title = title
        self.content = content
        self.citations = citations or []


class StructuredMinutes:
    def __init__(
        self,
        sections: list[SectionContent] | None = None,
        qa_pairs: list[QAPair] | None = None,
        takeaways: list[Takeaway] | None = None,
        raw_json: str = "",
    ):
        self.sections = sections or []
        self.qa_pairs = qa_pairs or []
        self.takeaways = takeaways or []
        self.raw_json = raw_json


class StructuredGenerator:
    PROMPT_TEMPLATE = """你是一个专业的Q&A提取专家。请根据以下会议转录文本，按照时间顺序逐条提取所有问答对。

## 转录文本
{transcript}

## 输出要求
1. 必须返回有效的JSON格式，不要包含任何其他文字或markdown标记
2. JSON结构如下：
{{
  "qa_pairs": [
    {{"question": "问题内容", "answer": "回答内容", "asker": "提问者姓名", "timerange": "00:01:23-00:02:24"}}
  ]
}}

3. 提取规则：
   - 按照转录文本的时间顺序，逐条提取每一个提问及其对应的回答
   - 每个提问对应一组 Q&A，包括重复提问、跟进提问
   - 提取数量应与转录文本中的提问数量基本一致
   - timerange 字段填写该问答对应的时间范围，格式为 "开始时间-结束时间"
   - 如果不知道具体时间范围，留空字符串
   - 只提取 qa_pairs，不要输出 sections 或 takeaways

请直接输出JSON："""

    def __init__(self, llm_client, template: Template | None = None):
        self._llm = llm_client
        self._template = template

    def generate(self, enhanced_transcript: str) -> StructuredMinutes:
        if not enhanced_transcript.strip():
            return StructuredMinutes()

        prompt = self._build_prompt(enhanced_transcript)
        raw = self._llm.generate(prompt)
        data = self._parse_with_retry(prompt, raw)
        return self._build_minutes(data, raw)

    def _build_prompt(self, transcript: str) -> str:
        return self.PROMPT_TEMPLATE.format(transcript=transcript)

    def _parse_with_retry(self, original_prompt: str, raw: str) -> dict | None:
        data = self._try_parse_json(raw)
        if data is not None:
            return data

        retry_prompt = (
            original_prompt
            + "\n\n【重要】上次输出格式不正确。必须返回有效的 JSON，不要包含任何其他文字。"
        )
        raw2 = self._llm.generate(retry_prompt)
        data = self._try_parse_json(raw2)
        if data is not None:
            return data

        data = self._extract_json_fragment(raw2)
        if data is not None:
            return data

        data = self._extract_json_fragment(raw)
        if data is not None:
            return data

        return None

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = EXTRACT_JSON_RE.search(text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None
    

    @staticmethod
    def _extract_json_fragment(text: str) -> dict | None:
        start = text.find("{")
        if start < 0:
            return None
        raw = text[start:]
        for i in range(len(raw), 0, -1):
            candidate = raw[:i]
            if not candidate.endswith("}"):
                candidate += "}"
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None
    

    @staticmethod
    def _build_minutes(data: dict | None, raw_json: str) -> StructuredMinutes:
        if data is None:
            return StructuredMinutes(
                sections=[SectionContent(title="原文", content=raw_json)],
                raw_json=raw_json,
            )

        sections = []
        for s in data.get("sections", []):
            sections.append(
                SectionContent(
                    title=s.get("title", ""),
                    content=s.get("content", ""),
                    citations=s.get("citations", []),
                )
            )

        qa_pairs = []
        for q in data.get("qa_pairs", []):
            qa_pairs.append(
                QAPair(
                    question=q.get("question", ""),
                    answer=q.get("answer", ""),
                    asker=q.get("asker", ""),
                    timerange=q.get("timerange", ""),
                )
            )

        takeaways = []
        for t in data.get("takeaways", []):
            takeaways.append(
                Takeaway(
                    content=t.get("content", ""),
                    type=t.get("type", "事实"),
                )
            )

        return StructuredMinutes(
            sections=sections,
            qa_pairs=qa_pairs,
            takeaways=takeaways,
            raw_json=json.dumps(data, ensure_ascii=False),
        )
