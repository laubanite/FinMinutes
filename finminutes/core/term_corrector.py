"""术语纠错：确定性部分（精确替换 + 音近告警），不依赖 LLM。

分工原则（按误伤代价从低到高）：

1. **精确命中** `corrections` → 直接替换。术语表里已经写明「错写成这样」，
   替换是安全的、可审计的，不需要任何模型参与。
2. **拼音近似命中** → 只告警，不自动改写。中文同音字误伤率极高
   （「结合/键合」「六篇/留片」都是同音），自动替换会把正确内容改坏；
   告警进入校验稿供人工确认，确认后的错写再回流成新的 corrections，
   形成「人工核对 → 词表扩充」的闭环。

这样设计的好处：自动改写那部分 100% 可解释、可回归测试；
模型只负责它擅长的泛化（生成候选词表），不负责拍板改字。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from finminutes.core.models import Glossary
from finminutes.core.rewriter import _same_script

try:  # 可选依赖：未安装时音近检测自动降级为不可用，精确替换照常工作
    from pypinyin import lazy_pinyin as _lazy_pinyin
except ImportError:  # pragma: no cover - 取决于环境
    _lazy_pinyin = None


_ASCII_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .\-_+/%]*$")


@dataclass
class CorrectionHit:
    """一次纠错命中。kind 区分可信度：exact 已改写，phonetic 仅告警。"""

    term: str
    matched: str
    start: int
    end: int
    kind: str
    context: str = ""

    @property
    def is_applied(self) -> bool:
        return self.kind == "exact"


def phonetic_available() -> bool:
    """音近检测是否可用（需要 pypinyin，可选安装：pip install pypinyin）。"""
    return _lazy_pinyin is not None


def pinyin_syllables(text: str) -> list[str] | None:
    """逐字无声调拼音（公开入口，供混淆集扩展复用）。不可用时返回 None。"""
    return _pinyin_syllables(text)


def context_snippet(text: str, start: int, end: int, width: int = 12) -> str:
    """截取命中位置前后的上下文，用于告警可读性。"""
    left = text[max(0, start - width) : start]
    right = text[end : end + width]
    return f"{left}【{text[start:end]}】{right}".replace("\n", " ")


_context = context_snippet


# ---------------------------------------------------------------------------
# 精确替换
# ---------------------------------------------------------------------------


def apply_exact(text: str, glossary: Glossary) -> tuple[str, list[CorrectionHit]]:
    """把术语表中登记的错误写法精确替换为正确写法。

    纯确定性、零模型调用。ASCII 词条加词边界避免「GMB」误伤「GMBX」；
    中文词条无词边界概念，直接替换（错写本身已经足够特异）。
    """
    if not text or not glossary or not glossary.terms:
        return text, []

    mapping: dict[str, str] = {}
    for term in glossary.terms:
        for wrong in term.corrections:
            wrong = (wrong or "").strip()
            # 用精确比较而非大小写归一：`finfet → FinFET` 是合法的大小写规范化条目
            if wrong and wrong != term.term:
                mapping.setdefault(wrong.lower(), term.term)
    if not mapping:
        return text, []

    # 长词优先，避免短错写先吃掉长错写的一部分
    parts = []
    for wrong in sorted(mapping, key=len, reverse=True):
        escaped = re.escape(wrong)
        if _ASCII_RE.match(wrong):
            parts.append(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
        else:
            parts.append(escaped)
    pattern = re.compile("|".join(parts), re.IGNORECASE)

    hits: list[CorrectionHit] = []

    def _replace(match: re.Match) -> str:
        correct = mapping.get(match.group(0).lower())
        if not correct:
            return match.group(0)
        hits.append(
            CorrectionHit(
                term=correct,
                matched=match.group(0),
                start=match.start(),
                end=match.end(),
                kind="exact",
                context=_context(text, match.start(), match.end()),
            )
        )
        return correct

    return pattern.sub(_replace, text), hits


# ---------------------------------------------------------------------------
# 音近告警（只报告，不改写）
# ---------------------------------------------------------------------------


def _seq_distance(a: list[str], b: list[str]) -> int:
    """序列级编辑距离（拼音音节为单位，不是字符）。"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _distance_threshold(length: int) -> int:
    """按术语长度收紧容错：2 字必须完全同音，3 字以上才容 1 个音节偏差。

    这个阈值是 golden 评测逼出来的，不是拍脑袋：
    - 2 字术语容 1 个音节 = 允许一半音节不同，于是「政府投资」里的「府投」、
      「的投资」里的「的投」全部命中「返投」，单场访谈就刷出 35 条误报。
    - 4 字术语容 2 个音节，半数字符不同也会命中，误报明显增加。
    实测所有真实错听（公司名与技术术语共 5 类）的音节距离都 ≤ 1，
    收紧阈值不损失召回，只砍误报。
    """
    return 0 if length <= 2 else 1


def _pinyin_syllables(text: str) -> list[str] | None:
    """逐字取无声调拼音；含非中文时保留原字符，保证与原文等长可对齐。"""
    if _lazy_pinyin is None:
        return None
    try:
        syllables = _lazy_pinyin(text, errors=lambda chars: list(chars))
    except TypeError:  # 老版本 pypinyin 无 errors 参数
        syllables = _lazy_pinyin(text)
    if len(syllables) != len(text):
        return None
    return [s.lower() for s in syllables]


def scan_phonetic(
    text: str,
    glossary: Glossary,
    max_hits: int = 50,
    min_term_len: int = 2,
) -> list[CorrectionHit]:
    """扫描文本中的「拼音近似但字形不同」片段，作为疑似 ASR 错词告警。

    不做任何改写——中文同音字太多，自动替换的误伤代价远高于漏报。
    约束（缺一不可，用来压误报）：
      - 同字形体系（中文不跨到拉丁）
      - 与术语至少共享 1 个汉字（挡掉「六篇 / 留片」这类零字重合的纯同音）
      - 拼音音节编辑距离 ≤ 阈值（2 字术语 0，即完全同音；3 字以上 1）

    锚定策略：只从「与术语共享字符」的位置反推窗口，避免全文滑窗的 O(n·m) 开销。
    """
    if not text or not glossary or not glossary.terms or _lazy_pinyin is None:
        return []

    text_py = _pinyin_syllables(text)
    if text_py is None:
        return []

    hits: list[CorrectionHit] = []
    seen: set[tuple[int, str]] = set()

    for term in glossary.terms:
        word = (term.term or "").strip()
        length = len(word)
        if length < min_term_len:
            continue
        # 音近检测只对中文术语有意义；纯拉丁术语靠精确替换覆盖
        if _ASCII_RE.match(word):
            continue
        term_py = _pinyin_syllables(word)
        if term_py is None:
            continue

        for offset, anchor in enumerate(word):
            pos = text.find(anchor)
            while pos != -1:
                start = pos - offset
                if start >= 0 and start + length <= len(text):
                    key = (start, word)
                    if key not in seen:
                        seen.add(key)
                        window = text[start : start + length]
                        if _is_suspicious(window, word, text_py[start : start + length], term_py):
                            hits.append(
                                CorrectionHit(
                                    term=word,
                                    matched=window,
                                    start=start,
                                    end=start + length,
                                    kind="phonetic",
                                    context=_context(text, start, start + length),
                                )
                            )
                            if len(hits) >= max_hits:
                                return _sorted(hits)
                pos = text.find(anchor, pos + 1)

    return _sorted(hits)


def _is_han(char: str) -> bool:
    """真汉字（不含全角标点）。`rewriter._is_cjk` 把「，。」也算 CJK，
    拿来做窗口校验会把「，投」当成合法窗口。"""
    return 0x4E00 <= ord(char) <= 0x9FFF


def _is_suspicious(window: str, word: str, window_py: list[str], term_py: list[str]) -> bool:
    if window == word:
        return False
    if not _same_script(window, word):
        return False
    # 纯汉字术语要求窗口也全是汉字：挡掉「，投」「的投」这类靠标点/助词
    # 凑出来的假命中（音节距离为 1 时它们能蒙混过关）
    if all(_is_han(c) for c in word) and not all(_is_han(c) for c in window):
        return False
    length = len(word)
    shared = len(set(window) & set(word))
    if shared < max(1, length // 2):
        return False
    return _seq_distance(window_py, term_py) <= _distance_threshold(length)


def _sorted(hits: list[CorrectionHit]) -> list[CorrectionHit]:
    return sorted(hits, key=lambda h: (h.start, h.term))


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def format_report(hits: list[CorrectionHit], max_items: int = 20) -> str:
    """把命中列表渲染为校验稿里的 markdown 片段。无命中返回空串。

    三类命中分节呈现，可信度依次递减：exact（已改写）> variant（混淆集）
    > phonetic（拼音近似）。后两类只提示、不动正文。
    """
    if not hits:
        return ""

    sections = [
        ("exact", "已自动纠正", "术语表精确命中，确定性替换"),
        ("variant", "疑似同音错写", "混淆集扩展命中，仅提示未改动正文"),
        ("phonetic", "疑似音近错词", "拼音近似，仅提示未改动正文"),
    ]

    lines: list[str] = ["## 术语纠错", ""]
    for kind, title, note in sections:
        group = [h for h in hits if h.kind == kind]
        if not group:
            continue
        lines.append(f"**{title} {len(group)} 处**（{note}）")
        lines.append("")
        # 同一「错写 → 正确词」聚合成一行并标次数，避免同一个错词刷 14 行
        grouped: dict[tuple[str, str], list[CorrectionHit]] = {}
        for h in group:
            grouped.setdefault((h.matched, h.term), []).append(h)
        for (matched, term), items in list(grouped.items())[:max_items]:
            suffix = f"（{len(items)} 处）" if len(items) > 1 else ""
            ctx = items[0].context
            if kind == "exact":
                lines.append(f"- `{matched}` → **{term}**{suffix}　{ctx}")
            else:
                lines.append(f"- `{matched}` 疑似 **{term}**{suffix}　{ctx}")
        if len(grouped) > max_items:
            lines.append(f"- …… 另有 {len(grouped) - max_items} 类")
        lines.append("")
    return "\n".join(lines).rstrip()
