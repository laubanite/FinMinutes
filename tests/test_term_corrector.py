import finminutes.core.term_corrector as tc
from finminutes.core.models import Glossary, Term


def _glossary(*terms: Term) -> Glossary:
    return Glossary(industry="test", terms=list(terms))


def test_apply_exact_replaces_known_misspelling():
    g = _glossary(Term(term="示例电子", corrections=["示例电字"]))
    out, hits = tc.apply_exact("我们跟示例电字合作很久了", g)
    assert out == "我们跟示例电子合作很久了"
    assert len(hits) == 1
    assert hits[0].kind == "exact"
    assert hits[0].is_applied
    assert hits[0].matched == "示例电字"


def test_apply_exact_respects_word_boundary_for_latin():
    """ASCII 词条必须加词边界，避免 GMB 误伤 GMBX。"""
    g = _glossary(Term(term="GMV", corrections=["GMB"]))
    out, _ = tc.apply_exact("GMB 同比增长，GMBX 不变", g)
    assert out == "GMV 同比增长，GMBX 不变"


def test_apply_exact_is_case_insensitive_for_latin():
    g = _glossary(Term(term="FinFET", corrections=["finfet"]))
    out, _ = tc.apply_exact("采用 Finfet 工艺", g)
    assert out == "采用 FinFET 工艺"


def test_apply_exact_noop_without_corrections():
    g = _glossary(Term(term="流片", corrections=[]))
    text = "我们下个月流片"
    out, hits = tc.apply_exact(text, g)
    assert out == text
    assert hits == []


def test_apply_exact_skips_identity_mapping():
    """corrections 与 term 相同时不应产生替换记录。"""
    g = _glossary(Term(term="流片", corrections=["流片"]))
    out, hits = tc.apply_exact("流片", g)
    assert out == "流片"
    assert hits == []


def test_apply_exact_longer_correction_wins():
    g = _glossary(
        Term(term="示例电子", corrections=["示例电字"]),
        Term(term="示例集团", corrections=["示例"]),
    )
    out, _ = tc.apply_exact("示例电字那边", g)
    assert out == "示例电子那边"


def test_phonetic_flags_homophone_misspelling(monkeypatch):
    """同音不同字（邵/绍）应被识别为疑似错词，但不改写正文。"""
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    g = _glossary(Term(term="示例电子", corrections=[]))
    text = "我们跟示例垫子聊了良率问题"
    hits = tc.scan_phonetic(text, g)
    assert [h.matched for h in hits] == ["示例垫子"]
    assert hits[0].kind == "phonetic"
    assert not hits[0].is_applied


def test_phonetic_ignores_zero_char_overlap(monkeypatch):
    """「六篇」与「留片」拼音完全相同但零字重合，必须压掉，否则误报爆炸。"""
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    g = _glossary(Term(term="流片", corrections=[]))
    assert tc.scan_phonetic("我写了六篇文章", g) == []


def test_phonetic_ignores_correct_text(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    g = _glossary(Term(term="示例电子", corrections=[]))
    assert tc.scan_phonetic("示例电子的良率", g) == []


def test_phonetic_unavailable_without_pypinyin(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", None)
    g = _glossary(Term(term="示例电子", corrections=[]))
    assert tc.phonetic_available() is False
    assert tc.scan_phonetic("示例垫子", g) == []


def test_phonetic_skips_latin_terms(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    g = _glossary(Term(term="FinFET", corrections=[]))
    assert tc.scan_phonetic("finfet 工艺", g) == []


def test_format_report_lists_both_sections(monkeypatch):
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    g = _glossary(
        Term(term="示例电子", corrections=["示例电字"]),
    )
    text = "示例电字与示例垫子"
    corrected, exact_hits = tc.apply_exact(text, g)
    suspect_hits = tc.scan_phonetic(corrected, g)
    report = tc.format_report(exact_hits + suspect_hits)
    assert "已自动纠正 1 处" in report
    assert "疑似音近错词" in report
    assert "示例电字" in report


def test_format_report_empty_when_no_hits():
    assert tc.format_report([]) == ""


# ---------------------------------------------------------------------------
# 以下三条是 golden 评测（docs/golden）跑出来的收紧规则，改动前先看 docs/golden/eval_report.md
# ---------------------------------------------------------------------------


def test_phonetic_requires_exact_homophone_for_two_char_term(monkeypatch):
    """2 字术语容 1 个音节 = 允许一半不同，于是「政府投资」里的「府投」会命中「返投」。

    单场访谈就靠这个刷出 35 条误报，precision 掉到 36%。
    """
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    g = _glossary(Term(term="返投"))
    assert tc.scan_phonetic("政府投资和的投资额度", g) == []


def test_phonetic_still_flags_two_char_homophone(monkeypatch):
    """收紧阈值不能把真错词一起砍掉：反/返 完全同音，必须还能报出来。"""
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    g = _glossary(Term(term="返投"))
    hits = tc.scan_phonetic("还有反投任务", g)
    assert [h.matched for h in hits] == ["反投"]


def test_phonetic_rejects_window_with_punctuation(monkeypatch):
    """「示例，电」音节距离只有 1，但夹着标点的窗口不是词，必须挡掉。"""
    monkeypatch.setattr(tc, "_lazy_pinyin", _fake_pinyin)
    g = _glossary(Term(term="示例电子"))
    assert tc.scan_phonetic("示例，电的良率", g) == []


_FAKE_TABLE = {
    "示": "shi", "事": "shi", "例": "li",
    "电": "dian", "垫": "dian", "子": "zi", "字": "zi",
    "流": "liu", "留": "liu", "片": "pian", "六": "liu", "篇": "pian",
    "返": "fan", "反": "fan", "投": "tou",
}


def _fake_pinyin(text, errors=None):
    return [_FAKE_TABLE.get(c, c) for c in text]
