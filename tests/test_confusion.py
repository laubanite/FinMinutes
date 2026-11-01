import finminutes.core.term_corrector as tc
from finminutes.core.confusion import expand_glossary, expand_term, scan_all, scan_variants
from finminutes.core.glossary_loader import dump_glossary
from finminutes.core.models import Glossary, Term

# 测试用同音表：只覆盖需要的字，避免依赖真实拼音库
_HOMO = {
    "shi": ["示", "事", "世", "市"],
    "li": ["例", "立", "力", "利"],
    "dian": ["电", "垫", "店", "点"],
    "zi": ["子", "字", "资", "紫"],
    "liu": ["流", "留", "六", "刘"],
    "pian": ["片", "篇", "偏", "便"],
}

_PY = {
    "示": "shi", "事": "shi", "世": "shi", "市": "shi",
    "例": "li", "立": "li", "力": "li", "利": "li",
    "电": "dian", "垫": "dian", "店": "dian", "点": "dian",
    "子": "zi", "字": "zi", "资": "zi", "紫": "zi",
    "流": "liu", "留": "liu", "六": "liu", "刘": "liu",
    "片": "pian", "篇": "pian", "偏": "pian", "便": "pian",
}


def _fake_pinyin(text, errors=None):
    return [_PY.get(c, c) for c in text]


def _glossary(*terms: Term) -> Glossary:
    return Glossary(industry="test", terms=list(terms))


# ---------------------------------------------------------------------------
# expand_term
# ---------------------------------------------------------------------------


def test_expand_term_single_position_substitutions(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    variants = expand_term("示例电子", homophones=_HOMO)
    assert "事例电子" in variants
    assert "示立电子" in variants
    assert "示例垫子" in variants
    assert "示例电字" in variants


def test_expand_term_excludes_original(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    variants = expand_term("示例电子", homophones=_HOMO)
    assert "示例电子" not in variants


def test_expand_term_interleaves_positions(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    """前 4 个候选应分别落在第 1/2/3/4 个字上，避免截断只保留前几位。"""
    variants = expand_term("示例电子", homophones=_HOMO, max_variants=4)
    changed = {i for v in variants for i in range(4) if v[i] != "示例电子"[i]}
    assert changed == {0, 1, 2, 3}


def test_expand_term_respects_max_variants(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    variants = expand_term("示例电子", homophones=_HOMO, max_variants=5)
    assert len(variants) == 5


def test_expand_term_skips_single_char(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    assert expand_term("芯", homophones=_HOMO) == []


def test_expand_term_without_homophone_table(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    assert expand_term("示例电子", homophones={}) == []


def test_expand_term_two_positions_covers_double_substitution(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    variants = expand_term("示例电子", homophones=_HOMO, max_variants=500, max_positions=2)
    assert "事例垫子" in variants


# ---------------------------------------------------------------------------
# expand_glossary
# ---------------------------------------------------------------------------


def test_expand_glossary_fills_auto_variants(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    g = _glossary(Term(term="示例电子", corrections=[]))
    stats = expand_glossary(g, homophones=_HOMO)
    assert stats["示例电子"] == len(g.terms[0].auto_variants)
    assert "示例垫子" in g.terms[0].auto_variants


def test_expand_glossary_skips_confirmed_corrections(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    """已人工登记的错写不应重复出现在 auto_variants 里。"""
    g = _glossary(Term(term="示例电子", corrections=["示例垫子"]))
    expand_glossary(g, homophones=_HOMO)
    assert "示例垫子" not in g.terms[0].auto_variants


# ---------------------------------------------------------------------------
# scan_variants
# ---------------------------------------------------------------------------


def test_scan_variants_finds_candidate():
    g = _glossary(Term(term="示例电子", auto_variants=["示例垫子"]))
    hits = scan_variants("我们跟示例垫子聊了良率", g)
    assert [h.matched for h in hits] == ["示例垫子"]
    assert hits[0].kind == "variant"
    assert hits[0].term == "示例电子"
    assert not hits[0].is_applied


def test_scan_variants_skips_frequent_variant_when_correct_form_present():
    """正确写法也在文中出现时，候选高频多半是合法常用词，不告警。"""
    g = _glossary(Term(term="示例电子", auto_variants=["示例垫子"]))
    text = "示例电子" + "示例垫子" * 10
    assert scan_variants(text, g, repeat_threshold=5) == []


def test_scan_variants_alerts_consistent_mishearing():
    """正确写法全文 0 次、候选高频 → 持续误听，必须逐处告警（实测：错误写法 14 次）。"""
    g = _glossary(Term(term="示例电子", auto_variants=["示例垫子"]))
    text = "示例垫子" * 14
    hits = scan_variants(text, g, repeat_threshold=5)
    assert len(hits) == 14
    assert {h.kind for h in hits} == {"variant"}


def test_scan_variants_emits_each_occurrence():
    g = _glossary(Term(term="示例电子", auto_variants=["示例垫子"]))
    hits = scan_variants("示例垫子，示例垫子", g)
    assert [h.start for h in hits] == [0, 5]


def test_scan_variants_keeps_when_correct_term_more_frequent():
    """正确词本身高频出现时，少量候选仍应告警。"""
    g = _glossary(Term(term="示例电子", auto_variants=["示例垫子"]))
    text = "示例电子" * 10 + "示例垫子"
    hits = scan_variants(text, g, repeat_threshold=5)
    assert len(hits) == 1


def test_scan_variants_empty_without_auto_variants():
    g = _glossary(Term(term="示例电子", corrections=["示立电子"]))
    assert scan_variants("示立电子", g) == []


def test_scan_variants_no_match_returns_empty():
    g = _glossary(Term(term="示例电子", auto_variants=["示例垫子"]))
    assert scan_variants("完全无关的一段话", g) == []


# ---------------------------------------------------------------------------
# scan_all
# ---------------------------------------------------------------------------


def test_scan_all_dedupes_variant_over_phonetic(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    g = _glossary(Term(term="示例电子", auto_variants=["示例垫子"]))
    hits = scan_all("我们跟示例垫子聊", g)
    assert len(hits) == 1
    assert hits[0].kind == "variant"


def test_scan_all_keeps_phonetic_when_no_variant(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    g = _glossary(Term(term="示例电子", auto_variants=[]))
    hits = scan_all("我们跟示例垫子聊", g)
    assert [h.kind for h in hits] == ["phonetic"]


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------


def test_dump_glossary_keeps_auto_variants():
    g = _glossary(Term(term="示例电子", context="晶圆厂", corrections=["示立电子"],
                       auto_variants=["示例垫子", "事例电子"]))
    data = dump_glossary(g)
    term = data["terms"][0]
    assert term["auto_variants"] == ["示例垫子", "事例电子"]
    assert term["corrections"] == ["示立电子"]
    assert term["context"] == "晶圆厂"


def test_format_report_renders_variant_section():
    g = _glossary(Term(term="示例电子", auto_variants=["示例垫子"]))
    hits = scan_variants("我们跟示例垫子聊", g)
    report = tc.format_report(hits)
    assert "疑似同音错写 1 处" in report
    assert "混淆集扩展命中" in report


def test_expand_term_respects_min_term_len(monkeypatch):
    """2 字术语候选易撞日常词，可调高 min_term_len 跳过。"""
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    assert expand_term("示例", homophones=_HOMO) != []
    assert expand_term("示例", homophones=_HOMO, min_term_len=3) == []


def test_expand_term_skips_latin_positions(monkeypatch):
    """拉丁位没有同音字可换。用字符本身当拼音去查表会撞出「啊rokia」这种废候选。"""
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    table = {"a": ["啊", "阿"], "r": ["日"], "o": ["哦"], "k": ["科"]}
    assert expand_term("ACME", homophones=table) == []


def test_expand_term_keeps_common_variant_within_budget():
    """真实拼音表回归：同音字按码位排序时，「事例电子」被生僻异体字挤到第 126 位，
    超出默认 120 上限，整条正例直接漏检。常用字必须排前面。

    依赖 pypinyin，未安装时跳过（本用例的价值就在于跑真实字表）。
    """
    if not tc.phonetic_available():
        return
    variants = expand_term("示例电子", max_variants=120)
    assert "事例电子" in variants
