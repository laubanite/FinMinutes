import re

DEFAULT_FILLERS = [
    "嗯", "啊", "那个", "这个", "就是", "然后", "反正",
    "就是说", "对吧", "是吧", "其实", "基本上",
    "那么", "的话", "的时候", "的话呢",
]

CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "两": 2,
}
_CN_NUM_CHARS = "".join(CN_DIGITS) + "十百千万亿"
_CN_NUM_RE = re.compile(f"[{_CN_NUM_CHARS}]{{2,}}")


class Preprocessor:
    def __init__(self, fillers: list[str] | None = None):
        self.fillers = fillers or list(DEFAULT_FILLERS)

    def merge_speakers(self, text: str) -> str:
        if not text.strip():
            return text
        speaker_re = re.compile(r"^([^:\n]+):\s*$", re.MULTILINE)
        parts = speaker_re.split(text)
        if len(parts) < 3:
            return text
        result_parts = [parts[0]]
        i = 1
        while i < len(parts) - 1:
            speaker = parts[i].strip()
            content = parts[i + 1].strip()
            merged = [content]
            j = i + 2
            while j < len(parts) - 1 and parts[j].strip() == speaker:
                merged.append(parts[j + 1].strip())
                j += 2
            merged_text = "".join(merged)
            result_parts.append(f"{speaker}:\n{merged_text}")
            i = j
        return "\n\n".join(p for p in result_parts if p)

    def remove_fillers(self, text: str) -> str:
        if not text:
            return text
        for filler in self.fillers:
            text = text.replace(filler, "")
        text = re.sub(r"  +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def normalize_numbers(self, text: str) -> str:
        if not text:
            return text

        def _replace(match):
            raw = match.group(0)
            arabic = self._chinese_to_arabic(raw)
            if arabic is not None:
                return f"{arabic}[原:{raw}]"
            return raw

        return _CN_NUM_RE.sub(_replace, text)

    @staticmethod
    def detect_citation_style(text: str) -> str:
        if re.search(r"\d{2}:\d{2}:\d{2}", text):
            return "timestamp"
        return "paragraph"

    def clean(self, text: str) -> str:
        text = self.normalize_export(text)
        text = self.merge_speakers(text)
        text = self.remove_fillers(text)
        text = self.normalize_numbers(text)
        return text

    # ------------------------------------------------------------------
    # ASR 导出格式归一化
    # ------------------------------------------------------------------

    def normalize_export(self, text: str) -> str:
        """把常见 ASR 导出格式归一化为纯文本：JSON-lines（{"text":...}）、SRT/VTT。

        识别失败时原样返回，不影响正常转录文本。
        """
        if not text or not text.strip():
            return text
        json_text = self._extract_json_lines(text)
        if json_text is not None:
            return json_text
        srt_text = self._extract_srt(text)
        if srt_text is not None:
            return srt_text
        return text

    @staticmethod
    def _extract_json_lines(text: str) -> str | None:
        """识别 JSON-lines 格式（每行一个 {"text": "..."}），提取 text 字段拼接。

        要求所有非空行都是带 "text" 键的 JSON 对象，否则判为不认识并返回 None。
        """
        import json as _json

        lines = [ln for ln in text.split("\n") if ln.strip()]
        if not lines:
            return None
        parts = []
        for ln in lines:
            stripped = ln.strip()
            if not stripped.startswith("{"):
                return None
            try:
                obj = _json.loads(stripped)
            except _json.JSONDecodeError:
                return None
            if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
                return None
            parts.append(obj["text"])
        if not parts:
            return None
        return "\n".join(parts)

    @staticmethod
    def _extract_srt(text: str) -> str | None:
        """识别 SRT/VTT 字幕格式（序号 + 时间戳行 + 文本行），抽取文本行。"""
        lines = text.split("\n")
        if not any("-->" in ln for ln in lines[:5]):
            return None
        text_lines = []
        for ln in lines:
            stripped = ln.strip()
            if not stripped:
                continue
            if stripped.isdigit():
                continue  # 序号
            if "-->" in stripped:
                continue  # 时间戳行
            if stripped.startswith("WEBVTT") or stripped.startswith("NOTE"):
                continue
            text_lines.append(stripped)
        if not text_lines:
            return None
        return "\n".join(text_lines)

    @staticmethod
    def _chinese_to_arabic(text: str) -> int | None:
        if not text:
            return None
        result = 0
        section = 0
        current = 0
        for ch in text:
            if ch in CN_DIGITS:
                current = CN_DIGITS[ch]
            elif ch == "十":
                section += max(current, 1) * 10
                current = 0
            elif ch == "百":
                section += max(current, 1) * 100
                current = 0
            elif ch == "千":
                section += max(current, 1) * 1000
                current = 0
            elif ch == "万":
                section += current
                result += (section or 1) * 10000
                section = 0
                current = 0
            elif ch == "亿":
                section += current
                result += (section or 1) * 100000000
                section = 0
                current = 0
            else:
                return None
        result += section + current
        return result if result > 0 else None
