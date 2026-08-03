import re
import logging

logger = logging.getLogger(__name__)

from finminutes.core.models import Background, Glossary, Term


_SENTENCE_BOUNDARIES = ["\n", "。", "！", "？"]
_SECONDARY_BOUNDARIES = ["；", "，", "、", ")"]


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break

        split = _find_split(text, start, end)
        if split <= start:
            split = end

        chunks.append(text[start:split])
        next_start = split - overlap
        if next_start <= start:
            next_start = split
        start = next_start

    return chunks


def _find_split(text: str, start: int, end: int) -> int:
    region_start = max(end - 300, start)
    region_end = min(end, len(text))

    for punct in _SENTENCE_BOUNDARIES:
        pos = text.rfind(punct, region_start, region_end)
        if pos > start:
            return pos + len(punct)

    for punct in _SECONDARY_BOUNDARIES:
        pos = text.rfind(punct, region_start, region_end)
        if pos > start:
            return pos + len(punct)

    return end


def levenshtein_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 2:
        return 3
    a, b = a.lower(), b.lower()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x2E80 <= cp <= 0x2EFF
        or 0x3000 <= cp <= 0x303F
        or 0xFF00 <= cp <= 0xFFEF
    )


def _same_script(a: str, b: str) -> bool:
    a_cjk = any(_is_cjk(c) for c in a)
    b_cjk = any(_is_cjk(c) for c in b)
    return a_cjk == b_cjk


def find_corrections(text: str, term: Term) -> str | None:
    text_lower = text.lower()
    for wrong in term.corrections:
        wrong_lower = wrong.lower()
        if wrong_lower in text_lower:
            return term.term
        wlen = len(wrong_lower)
        for i in range(len(text_lower) - wlen + 1):
            segment = text_lower[i : i + wlen]
            if not _same_script(segment, wrong_lower):
                continue
            if levenshtein_distance(segment, wrong_lower) <= 2:
                return term.term
    return None


class Rewriter:
    PROMPT_TEMPLATE = """你是一个专业的金融会议纪要专家。请根据以下背景信息和术语纠错规则，对文本片段进行改写。

## 背景信息
{background}

## 术语纠错规则
{glossary_rules}

## 改写要求
1. 根据术语表纠正金融专业术语的ASR识别错误
2. 根据背景信息补充分析师注释（标注为[分析师注：...]）
3. 保持原文语义不变
4. 不要添加原文中没有的信息
5. 保持说话人标签格式

## 待改写文本
{chunk}

请直接输出改写后的文本："""

    def __init__(self, llm_client, glossary: Glossary, background: Background):
        self._llm = llm_client
        self._glossary = glossary
        self._background = background

    def rewrite(self, transcript: str) -> str:
        if not transcript.strip():
            return ""

        bg_str = self._format_background()
        glossary_rules = self._format_glossary_rules()
        chunks = chunk_text(transcript)

        processed = []
        for chunk in chunks:
            prompt = self.PROMPT_TEMPLATE.format(
                background=bg_str,
                glossary_rules=glossary_rules,
                chunk=chunk,
            )
            response = self._llm.generate(prompt)
            # # ===== 添加调试日志 =====
            # logger.warning(f"=== 分片 {chunks.index(chunk)+1}/{len(chunks)} 原始响应 ===")
            # logger.warning(f"响应长度: {len(response)} 字符")
            # logger.warning(f"响应内容 (前500字符): {response[:500]}")
            # # ========================
            processed.append(response)

        return self._merge_chunks(processed)

    def _format_background(self) -> str:
        bg = self._background
        parts = []
        for key, val in [
            ("公司", bg.company),
            ("行业", bg.industry),
            ("参与者", bg.participants),
            ("会议目的", bg.meeting_purpose),
        ]:
            if val:
                parts.append(f"{key}：{val}")
        if bg.known_consensus:
            for item in bg.known_consensus:
                parts.append(f"已知共识：{item}")
        return "\n".join(parts) or "（无）"

    def _format_glossary_rules(self) -> str:
        lines = []
        for term in self._glossary.terms:
            if term.corrections:
                corr_str = "、".join(term.corrections)
                lines.append(f"- 当出现「{corr_str}」时，应纠正为「{term.term}」（{term.context}）")
        return "\n".join(lines) or "（无）"

    @staticmethod
    def _merge_chunks(chunks: list[str]) -> str:
        if not chunks:
            return ""
        if len(chunks) == 1:
            return chunks[0]
        result = chunks[0]
        for i in range(1, len(chunks)):
            current = chunks[i]
            prev = chunks[i - 1]
            trimmed = _trim_overlap(current, prev)
            if trimmed:
                result += "\n" + trimmed
            else:
                result += "\n" + current
        return result


def _trim_overlap(current: str, prev: str) -> str:
    max_size = min(200, len(prev), len(current))
    for size in range(max_size, 5, -1):
        suffix = prev[-size:]
        idx = current.find(suffix)
        if 0 <= idx < len(current) - 10:
            return current[idx + len(suffix) :]
    return current
