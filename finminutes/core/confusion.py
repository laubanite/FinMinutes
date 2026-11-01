"""混淆集自动扩展（L3）：用同音字把术语扩写成候选错写形态。

## 为什么扩展结果只告警、不自动替换

上一版设想是「扩出来的错写全部走精确替换」，真做的时候站不住脚：

    术语「键合」→ 同音扩展 → 「结合」
    → 自动替换 → 「技术与市场结合」被改成「技术与市场键合」

同音字里混着大量合法常用词，自动替换的误伤代价远高于漏报（改错一个公司名比漏一个
错词严重）。所以分两档：

- `corrections`：人工/LLM 确认过的错写 → 确定性替换
- `auto_variants`：程序生成的同音候选 → 只进告警清单

人工在校验稿里看到告警、确认某条确实会错之后，把它从 `auto_variants` 挪进
`corrections`，下次就走确定性替换。这就是「扩展 → 告警 → 确认 → 自动化」的闭环。

## 为什么放在离线做

拼音转换和组合展开是一次性成本。`glossary expand` 跑完把结果落进 YAML，
运行时只需要字符串匹配，**不依赖拼音库**，也不用每次跑流水线都重新算。
"""

from __future__ import annotations

from itertools import zip_longest

from finminutes.core import term_corrector as _tc
from finminutes.core.models import Glossary
from finminutes.core.term_corrector import CorrectionHit, context_snippet

# 常用汉字区间（基本区）。用公共 API 反查同音字，避免依赖 pypinyin 内部字典结构
CJK_START = 0x4E00
CJK_END = 0x9FA5

DEFAULT_MAX_VARIANTS = 120
# 同一候选出现次数超过该阈值 → 更像合法常用词而非 ASR 偶发错听，不告警
DEFAULT_REPEAT_THRESHOLD = 5

_HOMOPHONE_CACHE: dict[str, list[str]] | None = None


def _is_han(char: str) -> bool:
    return CJK_START <= ord(char) <= CJK_END


def _is_common_char(char: str) -> bool:
    """用「能否用 GB2312 编码」当作常用字代理。

    GB2312 收 6763 个汉字，基本覆盖现代汉语常用字；异体字（俴/倹/儉）、
    生僻字、日韩汉字都编不进去。比引入一份词频表便宜，也不增加依赖。
    """
    try:
        char.encode("gb2312")
    except UnicodeEncodeError:
        return False
    return True


def homophone_map() -> dict[str, list[str]]:
    """构建「无声调拼音 → 同音字列表」反查表（进程内缓存）。

    遍历常用汉字区间，一次批量取拼音后按音节分组。整表约 2 万字，
    一次性构建成本可忽略；pypinyin 不可用时返回空表（扩展功能降级）。

    每个音节内部**常用字排前面**。这一点不是锦上添花：`expand_term` 有候选上限，
    按码位排序会让「件/俭/俴/倹/健/僭/儉/兼/冿/减/剑/剣…」里的生僻异体字占满配额，
    实测某个四字公司名，其常用同音候选被挤到第 126 位，超出默认 120 上限，
    整条正例直接漏检。
    """
    global _HOMOPHONE_CACHE
    if _HOMOPHONE_CACHE is not None:
        return _HOMOPHONE_CACHE

    chars = [chr(cp) for cp in range(CJK_START, CJK_END + 1)]
    syllables = _tc.pinyin_syllables("".join(chars))
    if not syllables:
        _HOMOPHONE_CACHE = {}
        return _HOMOPHONE_CACHE

    table: dict[str, list[str]] = {}
    for char, syllable in zip(chars, syllables):
        table.setdefault(syllable, []).append(char)
    # sort 是稳定排序，同一档内仍按码位，结果可复现
    for group in table.values():
        group.sort(key=lambda ch: not _is_common_char(ch))
    _HOMOPHONE_CACHE = table
    return table


def reset_cache() -> None:
    """清空同音字缓存（测试与切换拼音库时使用）。"""
    global _HOMOPHONE_CACHE
    _HOMOPHONE_CACHE = None


def expand_term(
    term: str,
    homophones: dict[str, list[str]] | None = None,
    max_variants: int = DEFAULT_MAX_VARIANTS,
    max_positions: int = 1,
    min_term_len: int = 2,
) -> list[str]:
    """把单个术语扩写成同音候选错写。

    只替换**部分位置**（默认 1 个字），不做全组合——四字术语全组合是
    同音字数^4 量级，几十万个候选既算不动也没意义。ASR 错听绝大多数
    落在单字上，替换 1 个字已经覆盖主要情形。

    生成顺序按位置轮转交错（第 1 字、第 2 字、第 3 字……再回到第 1 字），
    这样即使因上限被截断，也不会只保留前几个位置的候选。

    `min_term_len` 用于过滤过短术语：2 字术语扩出来的候选（如「键合」→「结合」）
    极易撞上日常词，误报率高，可调到 3 只扩展三字以上术语。
    """
    term = (term or "").strip()
    if len(term) < max(2, min_term_len):
        return []

    table = homophones if homophones is not None else homophone_map()
    if not table:
        return []

    syllables = _tc.pinyin_syllables(term)
    if not syllables or len(syllables) != len(term):
        return []

    # 每个位置各自的替换候选（排除原字）
    per_position: list[list[str]] = []
    for i, syllable in enumerate(syllables):
        # 只替换汉字位。拉丁/数字/空格位取到的「拼音」就是字符本身，
        # 拿去查同音表会撞出「啊A」这种废候选
        if not _is_han(term[i]):
            per_position.append([])
            continue
        options = [ch for ch in table.get(syllable, []) if ch != term[i]]
        per_position.append([term[:i] + ch + term[i + 1 :] for ch in options])

    variants: list[str] = []
    seen: set[str] = {term}
    for group in zip_longest(*per_position):
        for candidate in group:
            if candidate and candidate not in seen:
                seen.add(candidate)
                variants.append(candidate)
                if len(variants) >= max_variants:
                    return variants

    # 双字替换：在单字候选基础上再替换一个未动过的位置，上限保护
    if max_positions >= 2:
        for base in list(variants):
            for i, syllable in enumerate(syllables):
                if base[i] != term[i] or not _is_han(term[i]):
                    continue
                for ch in table.get(syllable, []):
                    if ch == term[i]:
                        continue
                    candidate = base[:i] + ch + base[i + 1 :]
                    if candidate not in seen:
                        seen.add(candidate)
                        variants.append(candidate)
                        if len(variants) >= max_variants:
                            return variants

    return variants


def expand_glossary(
    glossary: Glossary,
    homophones: dict[str, list[str]] | None = None,
    max_variants: int = DEFAULT_MAX_VARIANTS,
    max_positions: int = 1,
    min_term_len: int = 2,
) -> dict[str, int]:
    """就地填充每个术语的 auto_variants，返回 {term: 生成数量}。"""
    stats: dict[str, int] = {}
    for term in glossary.terms:
        variants = expand_term(
            term.term,
            homophones=homophones,
            max_variants=max_variants,
            max_positions=max_positions,
            min_term_len=min_term_len,
        )
        # 已被人工登记为 corrections 的错写不重复放进 auto_variants
        confirmed = {c.strip() for c in term.corrections if c and c.strip()}
        term.auto_variants = [v for v in variants if v not in confirmed]
        stats[term.term] = len(term.auto_variants)
    return stats


def scan_variants(
    text: str,
    glossary: Glossary,
    max_hits: int = 50,
    repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD,
) -> list[CorrectionHit]:
    """在文本中查找混淆集候选，产出「疑似同音错写」告警（纯字符串匹配，无需拼音库）。

    频次过滤：候选出现次数 > max(repeat_threshold, 正确词出现次数) 时跳过——
    出现太频繁的多半是合法常用词（「中心」），不是偶发错听。
    """
    if not text or not glossary or not glossary.terms:
        return []

    index: dict[str, str] = {}
    for term in glossary.terms:
        for variant in getattr(term, "auto_variants", None) or []:
            variant = (variant or "").strip()
            if variant and variant != term.term:
                index.setdefault(variant, term.term)
    if not index:
        return []

    hits: list[CorrectionHit] = []
    for variant, correct in index.items():
        variant_count = _count_occurrences(text, variant)
        if variant_count == 0:
            continue
        correct_count = _count_occurrences(text, correct)
        # 频次过滤：候选明显比正确写法更常见时，多半是合法常用词而非错听。
        # 但正确写法在全文一次都没出现时不能过滤——那说明候选就是这段录音里
        # 唯一的写法，典型的「公司名被持续听错」（实测：错误写法 14 次 / 正确写法 0 次），
        # 恰恰是最该告警的情形。
        if correct_count > 0 and variant_count > max(repeat_threshold, correct_count):
            continue
        # 逐处产出（而非只报第一次），告警清单才能指到每个位置；
        # 展示层由 format_report 按 (matched, term) 聚合，不会刷屏。
        pos = text.find(variant)
        while pos != -1 and len(hits) < max_hits:
            hits.append(
                CorrectionHit(
                    term=correct,
                    matched=variant,
                    start=pos,
                    end=pos + len(variant),
                    kind="variant",
                    context=context_snippet(text, pos, pos + len(variant)),
                )
            )
            pos = text.find(variant, pos + len(variant))

    hits.sort(key=lambda h: (h.start, h.term))
    return hits[:max_hits]


def scan_all(
    text: str,
    glossary: Glossary,
    max_hits: int = 50,
    repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD,
) -> list[CorrectionHit]:
    """混淆集告警 + 拼音近似告警合并去重。

    同一片段两边都命中时保留 variant（离线预生成，更具体），
    拼音扫描在未装 pypinyin 时自动不产出。
    """
    hits = scan_variants(text, glossary, max_hits=max_hits, repeat_threshold=repeat_threshold)

    phonetic = _tc.scan_phonetic(text, glossary, max_hits=max_hits)
    if phonetic:
        covered = {(h.start, h.matched) for h in hits}
        hits.extend(h for h in phonetic if (h.start, h.matched) not in covered)

    hits.sort(key=lambda h: (h.start, h.term))
    return hits[:max_hits]


def _count_occurrences(text: str, needle: str) -> int:
    if not needle:
        return 0
    count = 0
    pos = text.find(needle)
    while pos != -1:
        count += 1
        pos = text.find(needle, pos + len(needle))
    return count
