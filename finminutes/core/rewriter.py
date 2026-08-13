import re
import time
import logging

logger = logging.getLogger(__name__)

from finminutes.core.models import Background, Glossary, Term


_SENTENCE_BOUNDARIES = ["\n", "。", "！", "？"]
_SECONDARY_BOUNDARIES = ["；", "，", "、", ")"]


def format_glossary_rules(glossary: Glossary) -> str:
    """把术语纠错规则格式化为 prompt 文本；无规则时返回「（无）」。"""
    lines = []
    for term in glossary.terms:
        if term.corrections:
            corr_str = "、".join(term.corrections)
            lines.append(f"- 当出现「{corr_str}」时，应纠正为「{term.term}」（{term.context}）")
    return "\n".join(lines) or "（无）"


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


def parse_blocks(response: str) -> tuple[str, list[dict]]:
    """宽松解析分片响应为 topic 块：`[{"type": "...", "topic": "...", "text": "..."}]`。

    type 表示块内容表达形态（narration/qa/mixed），缺省按 narration 处理
    （宁可多进主题要点，不丢信息——用户删比加容易）。
    返回 (拼接文本, 块列表)；解析失败则整段兜底（内容零丢失）。
    """
    import json

    text = (response or "").strip()
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            start = text.index("[")
            end = text.rindex("]")
            data = json.loads(text[start : end + 1])
        except (ValueError, json.JSONDecodeError):
            data = None
    if not isinstance(data, list) or not data:
        return text, [{"type": "narration", "topic": "", "text": text}]
    blocks = []
    parts = []
    for item in data:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic", "")).strip()
        body = str(item.get("text", "")).strip()
        if not body:
            continue
        btype = str(item.get("type") or "narration").strip() or "narration"
        blocks.append({"type": btype, "topic": topic, "text": body})
        parts.append(body)
    if not blocks:
        return text, [{"type": "narration", "topic": "", "text": text}]
    return "\n\n".join(parts), blocks


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
    PROMPT_TEMPLATE = """你是一个专业的金融会议纪要专家。请根据以下背景信息和术语纠错规则，把转录片段改写成书面化的纪要内容，并按话题聚合。

## 背景信息
{background}

## 术语纠错规则
{glossary_rules}

## 要求
1. 根据术语表纠正金融专业术语的ASR识别错误；必要时根据背景信息补充分析师注释（标注为[分析师注：...]）
2. **把口语化表达改写为书面语**：删除口头语和语气词（嗯、啊、就是说、蛮、其实、基本上、的话等），把口语短句重组为规范、简练的书面句子（例：「我们跟中芯绍兴合作还是蛮紧密的」→「与中芯绍兴合作紧密」）
3. **必须保留全部实质信息**：数字、百分比、金额、公司名、专有名词、事件、时间、因果逻辑一条都不能删减、概括或遗漏；禁止添加原文没有的信息；改写后应比原文明显简短（冗余被去除），但信息量不减
4. 如原文有说话人标签（如「张三：」），保留以辅助核对
5. **按出现顺序把内容聚合为 2-4 个较大的话题块**（如：市场与竞争格局、产品与技术、产能与供应链、融资与规划），不要切得过细
6. **判断每块内容的表达形态并标注 type**：
   - "narration"：单向陈述/介绍/论述，无问答互动（主讲人介绍、技术论述等）
   - "qa"：明确的提问-回答互动（有问方有答方，含"您觉得…呢"这种跟进提问）
   - "mixed"：同一块内既有陈述又有问答，此时 text 只写陈述部分，问答部分不要写入（问答会单独提取）
   - 注意区分修辞性自问自答（如"为什么要做模组化？因为客户要小型化"）：这是陈述，标 narration，不算问答
7. **全文必须使用简体中文**：标题与正文一律简体，禁止繁体字（如"为/产/与/务/经/营/规/划/财/务/预/测/验"等需写为简体）

## 输出格式
只输出有效的JSON数组（不要包含任何其他文字或markdown标记），按出现顺序：
[{{"type": "narration|qa|mixed", "topic": "话题标题", "text": "该话题的书面化内容（保留全部数字与事实）"}}]

## 待改写文本
{chunk}

请直接输出JSON："""

    def __init__(self, llm_client, glossary: Glossary, background: Background,
                 chunk_size: int = 4000, on_retry=None, on_progress=None):
        self._llm = llm_client
        self._glossary = glossary
        self._background = background
        self._chunk_size = chunk_size
        self._on_retry = on_retry
        self._on_progress = on_progress  # on_progress(index, total, elapsed_sec)
        self.topic_blocks: list[list[dict]] = []  # 每分片按序的 topic 块（供 sections 派生）

    def rewrite(self, transcript: str) -> str:
        if not transcript.strip():
            return ""

        bg_str = self._format_background()
        glossary_rules = self._format_glossary_rules()
        chunks = chunk_text(transcript, max_chars=self._chunk_size)
        total = len(chunks)
        t0 = time.time()

        processed = []
        self.topic_blocks = []
        for i, chunk in enumerate(chunks, 1):
            prompt = self.PROMPT_TEMPLATE.format(
                background=bg_str,
                glossary_rules=glossary_rules,
                chunk=chunk,
            )
            response = self._llm.generate(prompt, retry_callback=self._on_retry)
            if self._on_progress:
                self._on_progress(i, total, time.time() - t0)
            # # ===== 添加调试日志 =====
            # logger.warning(f"=== 分片 {i}/{total} 原始响应 ===")
            # logger.warning(f"响应长度: {len(response)} 字符")
            # logger.warning(f"响应内容 (前500字符): {response[:500]}")
            # # ========================
            text, blocks = parse_blocks(response)
            processed.append(text)
            self.topic_blocks.append(blocks)

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
        return format_glossary_rules(self._glossary)

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
