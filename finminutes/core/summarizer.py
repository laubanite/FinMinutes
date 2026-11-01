import json
import re
import time
import logging

logger = logging.getLogger(__name__)

from finminutes.core.models import Template, Glossary, Background
from finminutes.core.rewriter import (
    RELATIVE_TIME_RULE,
    chunk_text,
    format_current_date,
    format_glossary_rules,
    parse_blocks,
)

_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)

EXTRACT_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


# ---------------------------------------------------------------------------
# 提问者标识：姓名填不出时退到角色
# ---------------------------------------------------------------------------
#
# 转录稿是纯文本（ASR 固定 response_format="text"），没有 speaker 标签，
# 所以 `asker`（提问者姓名）实测永远是空串——某场访谈的 26 条问答 26 条全空。
# 但「这一问是访谈方提的还是项目方提的」这个判断 LLM 做得出来，
# 而对一级市场访谈来说这个区分已经够用（返投要求是政府方问的，产能是企业方答的）。
#
# 所以分两个字段：
#   asker      —— 姓名，有说话人信息时才填得出来
#   asker_role —— 角色枚举，纯文本也能判断
# 渲染时优先姓名，其次角色，都没有才显示「（未指明）」。

ROLE_INTERVIEWER = "interviewer"
ROLE_INTERVIEWEE = "interviewee"
ROLE_UNKNOWN = "unknown"

# LLM 会自由发挥（「提问方」「投资方」「调研方」…），统一收敛到两个枚举。
# 认不出来就返回空串而不是猜——猜错的归属比留空更糟。
_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    ROLE_INTERVIEWER: (
        "interviewer", "asker", "investor", "访谈", "提问", "投资方", "投资人",
        "买方", "调研", "尽调", "调研方", "访谈方", "访谈者",
    ),
    ROLE_INTERVIEWEE: (
        "interviewee", "answerer", "company", "项目方", "企业方", "公司方",
        "被访", "受访", "回答方", "创始人", "管理层", "路演方",
    ),
}

ROLE_LABELS = {ROLE_INTERVIEWER: "访谈方", ROLE_INTERVIEWEE: "项目方"}
_EMPTY_ASKER = "（未指明）"


def normalize_asker_role(raw) -> str:
    """把 LLM 输出的角色描述归一化为枚举；认不出返回空串。"""
    text = str(raw or "").strip().lower()
    if not text or text in ("unknown", "none", "null", "无", "未知", "不确定"):
        return ""
    for role, aliases in _ROLE_ALIASES.items():
        if any(alias in text for alias in aliases):
            return role
    return ""


def asker_label(asker: str, role: str) -> str:
    """渲染用的提问者标签：姓名 > 角色 > （未指明）。"""
    name = (asker or "").strip()
    if name:
        return name
    return ROLE_LABELS.get((role or "").strip(), _EMPTY_ASKER)


class QAPair:
    def __init__(
        self,
        question: str = "",
        answer: str = "",
        asker: str = "",
        timerange: str = "",
        asker_role: str = "",
    ):
        self.question = question
        self.answer = answer
        self.asker = asker
        self.timerange = timerange
        self.asker_role = asker_role


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
        raw_json: str = "",
    ):
        self.sections = sections or []
        self.qa_pairs = qa_pairs or []
        self.raw_json = raw_json


def detect_transcript_type(text: str) -> str:
    """启发式识别转录类型：问号密度（？/? 每千字）< 1.5 → 单向陈述型。

    已用真实数据标定：访谈问答样本 3.02/千字 → qa；路演陈述样本 0.92/千字 → narration。
    """
    text = text or ""
    total = max(len(text), 1)
    q_marks = len(re.findall(r"[？?]", text))
    per_1000 = q_marks * 1000 / total
    if per_1000 < 1.5:
        return "narration"
    return "qa"


_TYPE_INSTRUCTIONS = {
    "narration": "- 本转录以单向陈述为主（演讲/路演/项目介绍）：问答对仅当存在明显提问-回答时提取",
    "qa": "- 本转录以问答为主：请完整提取所有问答对",
}


BLOCK_EXTRACT_TEMPLATE = """你是一个专业的金融会议纪要整理专家。以下是会议转录的一部分，请把它改写成书面化的纪要内容，并按话题聚合。

## 当前日期
{current_date}

## 要求
1. **把口语化表达改写为书面语**：删除口头语和语气词（嗯、啊、就是说、蛮、其实、基本上、的话等），把口语短句重组为规范、简练的书面句子
2. **必须保留全部实质信息**：数字、百分比、金额、公司名、专有名词、事件、时间、因果逻辑一条都不能删减、概括或遗漏；改写后应比原文明显简短，但信息量不减
3. 如原文有说话人标签（如「张三：」），保留以辅助核对
4. **按出现顺序把内容聚合为 2-4 个较大的话题块**，不要切得过细
5. **判断每块内容的表达形态并标注 type**：
   - "narration"：单向陈述/介绍/论述，无问答互动
   - "qa"：明确的提问-回答互动（有问方有答方）
   - "mixed"：同一块内既有陈述又有问答，此时 text 只写陈述部分，问答部分不要写入（问答会单独提取）
6. **全文必须使用简体中文**：标题与正文一律简体，禁止繁体字
7. {relative_time_rule}

## 输出格式
只输出有效的JSON数组（不要包含任何其他文字或markdown标记），按出现顺序：
[{{"type": "narration|qa|mixed", "topic": "话题标题", "text": "该话题的书面化内容（保留全部数字与事实）"}}]

## 转录片段
{chunk}

请直接输出JSON："""


QA_PROMPT_TEMPLATE = """你是一个专业的Q&A提取专家。请根据以下会议转录文本，提取问答对。

## 会议背景
{background}

## 当前日期
{current_date}

## 术语纠错规则
{glossary_rules}

## 转录文本
{transcript}

## 输出要求
1. 必须返回有效的JSON格式，不要包含任何其他文字或markdown标记
2. JSON结构如下：
{{
  "qa_pairs": [
    {{"question": "问题内容", "answer": "回答内容", "asker": "", "asker_role": "interviewer|interviewee|unknown", "timerange": ""}}
  ]
}}

3. 提取规则：
   - qa_pairs：按时间顺序提取每一个提问及其对应的回答，包括重复提问、跟进提问；数量应与转录中的提问数量基本一致
   - qa_pairs 的 answer 用书面语改写，去除口语化表达，但保留全部数字、金额、公司名与事实；转录中的 ASR 错词按上述术语纠错规则纠正
   - 问题与回答一律使用简体中文，禁止繁体字
   - timerange：仅当转录文本中含有明确时间戳（如 09:32 或 09:32-09:35）时填写；否则必须返回空字符串 ""，禁止编造或填入非时间戳内容
   - **asker / asker_role 的填写规则（重要）**：
     * `asker` 只填**原文明确出现的姓名或机构名**。转录稿通常没有说话人标记，此时必须留空字符串 ""，
       **绝对不要根据上下文推测或编造提问者姓名**
     * `asker_role` 填提问方的角色，三选一：
       - "interviewer"：访谈方/投资方/调研方提出问题（"你们公司……""能否介绍一下……"）
       - "interviewee"：项目方/企业方反向提问（"你们基金……""政府能给到什么支持"）
       - "unknown"：无法判断，或属于双方互相确认的追问
     * 判断依据是**提问内容指向谁**：问对方公司情况的是访谈方，问对方机构政策/流程的是项目方
   - **不要把单向陈述强行包装成问答对**：只有原文确实存在提问方与回答方时才提取；主讲人的单向陈述/介绍属于主题要点范畴，重复提取会造成同一段内容既出现在主题要点又出现在问答里
   - {relative_time_rule}
{type_instruction}

请直接输出JSON："""


def _normalize_topic(t: str) -> str:
    """去掉话题标题前导序号（如「1.」「2、」「（3）」）。"""
    t = (t or "").strip()
    t = re.sub(r"^(?:\d+[\.、）)]\s*|（\d+）\s*|[一二三四五六七八九十]+[、.．]?\s*)", "", t)
    return t.strip()


def _topics_similar(a: str, b: str) -> bool:
    a, b = _normalize_topic(a), _normalize_topic(b)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def merge_blocks(blocks: list[list[dict]]) -> list[SectionContent]:
    """把分片提取的 topic 块按序合并为 sections：仅合并相邻同话题，保持访谈顺序。

    type == "qa" 的块跳过（问答块不进主题要点，由 QA 提取单独覆盖）；
    type 缺省按 narration 处理（宁可多进主题要点，不丢信息）。
    """
    merged: list[SectionContent] = []
    for chunk_blocks in blocks:
        for b in chunk_blocks:
            if not isinstance(b, dict):
                continue
            if str(b.get("type") or "narration") == "qa":
                continue
            title = _normalize_topic(b.get("topic")) or "未命名"
            text = (b.get("text") or "").strip()
            if not text:
                continue
            if merged and _topics_similar(merged[-1].title, title):
                merged[-1].content += "\n\n" + text
            else:
                merged.append(SectionContent(title=title, content=text))
    return merged


# both 模式去重阈值：问答对内容被主题要点覆盖的 bigram 比例达到此值即判定重复。
# 取偏高值（0.7）以保守为先——宁可漏剔，不可误删真实问答。
_QA_DEDUP_THRESHOLD = 0.7
_QA_DEDUP_MIN_CHARS = 20


def _bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text or "")
    if len(t) < 2:
        return set()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _drop_qa_redundant_with_sections(qa_data: dict, sections: list,
                                     threshold: float = _QA_DEDUP_THRESHOLD) -> dict:
    """剔除与主题要点重复的问答对（both 模式专用）。

    根因：both 模式下 sections 来自 rewrite 的 narration 判定、QA 来自另一次独立提取，
    两次判断互不通信，同一段陈述可能被 rewrite 收进主题要点、又被 QA 提取器当成问答抽出。

    做法：确定性（零 LLM）按字符 bigram 计算「问答对内容被主题要点覆盖的比例」，
    超过阈值即判定该问答实为陈述，予以剔除——使每段内容只归属一处。
    主题要点优先保留（rewrite 已明确判定其非问答互动）。
    """
    pairs = qa_data.get("qa_pairs") or []
    if not pairs or not sections:
        return qa_data

    sec_bigrams: set[str] = set()
    for s in sections:
        sec_bigrams |= _bigrams(getattr(s, "content", "") or "")
    if not sec_bigrams:
        return qa_data

    kept: list[dict] = []
    dropped = 0
    for q in pairs:
        if not isinstance(q, dict):
            continue
        text = f"{q.get('question') or ''}{q.get('answer') or ''}"
        if len(re.sub(r"\s+", "", text)) < _QA_DEDUP_MIN_CHARS:
            kept.append(q)
            continue
        b = _bigrams(text)
        if b and len(b & sec_bigrams) / len(b) >= threshold:
            dropped += 1
            continue
        kept.append(q)

    if dropped:
        logger.info(
            "both 模式去重：剔除与主题要点重复的问答对 %d 条（阈值 %.2f，剩余 %d 条）",
            dropped, threshold, len(kept),
        )
    qa_data["qa_pairs"] = kept
    return qa_data


class StructuredGenerator:
    """分块保真：sections 来自 rewrite 的 topic 块（保序合并）；QA 单独一次调用提取。"""

    def __init__(self, llm_client, template: Template | None = None, on_retry=None,
                 block_chunk_size: int = 4000, qa_chunk_size: int = 8000,
                 glossary: Glossary | None = None, on_progress=None,
                 background: Background | None = None,
                 dedup_qa_with_sections: bool = True):
        self._llm = llm_client
        self._template = template
        self._on_retry = on_retry
        self._block_chunk_size = block_chunk_size
        self._qa_chunk_size = qa_chunk_size
        self._glossary = glossary or Glossary()
        self._on_progress = on_progress  # on_progress(index, total, elapsed_sec)
        self._background = background or Background()
        # both 模式下 sections 与 QA 由两次独立 LLM 判断产出，同一段可能两边都要；
        # 开启后按内容重叠度确定性剔除与主题要点重复的问答对（每段内容只归属一处）。
        self._dedup_qa_with_sections = dedup_qa_with_sections

    def generate(self, transcript: str, section_blocks: list[list[dict]] | None = None,
                 transcript_type: str | None = None, qa_source: str | None = None,
                 build_sections: bool = True, extract_qa: bool = True) -> StructuredMinutes:
        """生成结构化纪要。

        输入一律用改写前的原文（清洗转录）：改写会 formalize 掉隐性问题结构，
        sections 分块与 QA 识别都需要原文问法；rewrite 的书面化结果不参与结构化
        （主题要点来自 rewrite 分片标注的 section_blocks）。

        - sections：优先用 rewrite 阶段产出的 section_blocks（书面化、按访谈顺序）；
          无则自行对原文分块提取（如 --skip-rewrite 场景）；build_sections=False 时不生成。
        - qa_pairs：单独一次 LLM 调用提取，用 qa_source（默认等于 transcript）；
          extract_qa=False 时不提取（纯独白模式）。
        """
        source = qa_source or transcript
        if not source.strip():
            return StructuredMinutes()
        t = transcript_type or detect_transcript_type(source)
        sections = self._build_sections(source, section_blocks, t) if build_sections else []
        qa_data = self._extract_qa(source, t) if extract_qa else {}
        # both 模式：sections 与 QA 来自两次独立判断，同一段可能两边都收。
        # 以 rewrite 的「是否为问答互动」判定为准，把误抽成问答的陈述内容剔除。
        if sections and qa_data and self._dedup_qa_with_sections:
            qa_data = _drop_qa_redundant_with_sections(qa_data, sections)
        return self._build_minutes(sections, qa_data)

    def _format_background(self) -> str:
        """把会议背景信息格式化为 prompt 文本；无背景时返回「（无）」。

        背景此前只在 rewrite 阶段注入，而 qa 模式（默认）不跑 rewrite，
        导致 `-b` 被静默加载却从未使用——这里补上 QA 提取路径的注入。
        """
        bg = self._background
        parts = []
        for key, val in [
            ("公司", getattr(bg, "company", "")),
            ("行业", getattr(bg, "industry", "")),
            ("参与者", getattr(bg, "participants", "")),
            ("会议目的", getattr(bg, "meeting_purpose", "")),
        ]:
            if val:
                parts.append(f"{key}：{val}")
        for item in getattr(bg, "known_consensus", None) or []:
            parts.append(f"已知共识：{item}")
        return "\n".join(parts) or "（无）"

    def _build_sections(self, transcript: str, section_blocks, t: str) -> list[SectionContent]:
        if section_blocks:
            return merge_blocks(section_blocks)
        return merge_blocks(self._extract_blocks(transcript))

    def _extract_blocks(self, transcript: str) -> list[list[dict]]:
        """无 rewrite 分片标注时（如 --skip-rewrite）：自行分块提取 topic 块（宽解析无重试）。"""
        chunks = chunk_text(transcript, max_chars=self._block_chunk_size)
        current_date = format_current_date()
        results = []
        for chunk in chunks:
            prompt = BLOCK_EXTRACT_TEMPLATE.format(
                chunk=chunk,
                current_date=current_date,
                relative_time_rule=RELATIVE_TIME_RULE,
            )
            raw = self._llm.generate(prompt, retry_callback=self._on_retry)
            _text, blocks = parse_blocks(raw)
            results.append(blocks)
        return results

    def _extract_qa(self, transcript: str, t: str) -> dict:
        """提取 QA。

        文本超过 `qa_chunk_size` 时按块提取并合并——长文本（如 2 小时录音 4 万字符）
        单次调用会把整篇塞进一个 prompt，免费档 LLM 极易超时；分块后每个 prompt 都有界。
        """
        if not transcript:
            return {}
        if len(transcript) <= self._qa_chunk_size:
            return self._extract_qa_once(transcript, t)
        return self._extract_qa_chunked(transcript, t)

    def _extract_qa_once(self, transcript: str, t: str) -> dict:
        prompt = QA_PROMPT_TEMPLATE.format(
            transcript=transcript,
            type_instruction=_TYPE_INSTRUCTIONS.get(t, ""),
            glossary_rules=format_glossary_rules(self._glossary),
            background=self._format_background(),
            current_date=format_current_date(),
            relative_time_rule=RELATIVE_TIME_RULE,
        )
        raw = self._llm.generate(prompt, retry_callback=self._on_retry)
        data = self._parse_with_retry(prompt, raw)
        return data or {}

    @staticmethod
    def _norm_text(s: str) -> str:
        return re.sub(r"\s+", "", s or "").lower()

    def _extract_qa_chunked(self, transcript: str, t: str) -> dict:
        """按块提取 QA，按序合并去重。

        - overlap 放大到 400：跨块边界的问答对更容易完整落入同一次提取。
        - 单块失败只跳过该块（记录 warning，保留其余部分）；全部失败才向上抛，
          与单次调用的 fail-fast 语义一致。
        """
        chunks = chunk_text(transcript, max_chars=self._qa_chunk_size, overlap=400)
        total = len(chunks)
        t0 = time.time()
        qa_pairs: list[dict] = []
        seen_q: set[str] = set()
        failures = 0
        last_exc: Exception | None = None

        for idx, chunk in enumerate(chunks, 1):
            try:
                data = self._extract_qa_once(chunk, t)
            except Exception as e:  # 单块失败不拖垮整篇
                failures += 1
                last_exc = e
                logger.warning("QA 分块提取失败（跳过该块）: %s", e)
                continue
            if self._on_progress:
                self._on_progress(idx, total, time.time() - t0)
            for q in data.get("qa_pairs", []):
                if not isinstance(q, dict):
                    continue
                question = (q.get("question") or "").strip()
                if not question:
                    continue
                key = self._norm_text(question)
                if key in seen_q:
                    continue
                seen_q.add(key)
                qa_pairs.append(q)

        if not qa_pairs and failures and len(chunks) == failures:
            if last_exc is not None:
                raise last_exc

        return {"qa_pairs": qa_pairs}

    def _parse_with_retry(self, original_prompt: str, raw: str) -> dict | None:
        data = self._try_parse_json(raw)
        if data is not None:
            return data

        retry_prompt = (
            original_prompt
            + "\n\n【重要】上次输出格式不正确。必须返回有效的 JSON，不要包含任何其他文字。"
        )
        raw2 = self._llm.generate(retry_prompt, retry_callback=self._on_retry)
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
    def _build_minutes(sections: list[SectionContent], qa_data: dict) -> StructuredMinutes:
        qa_pairs = []
        for q in qa_data.get("qa_pairs", []):
            if isinstance(q, dict):
                qa_pairs.append(
                    QAPair(
                        question=q.get("question", ""),
                        answer=q.get("answer", ""),
                        asker=q.get("asker", ""),
                        timerange=q.get("timerange", ""),
                        # LLM 可能写「提问方」「投资方」等自由文本，这里收敛成枚举
                        asker_role=normalize_asker_role(q.get("asker_role")),
                    )
                )

        raw_json = json.dumps(
            {
                "sections": [
                    {"title": s.title, "content": s.content, "citations": s.citations}
                    for s in sections
                ],
                "qa_pairs": qa_data.get("qa_pairs", []),
            },
            ensure_ascii=False,
        )
        return StructuredMinutes(
            sections=sections,
            qa_pairs=qa_pairs,
            raw_json=raw_json,
        )
