"""提问者角色兜底（asker_role）的解析、渲染与序列化。

背景：转录稿是纯文本、无 speaker 标签，`asker`（姓名）实测永远为空
（某场访谈的 26 条问答全空）。这里验证「姓名填不出时退到角色」这条链路。
"""

from finminutes.core.formal_renderer import _format_qa_pairs
from finminutes.core.renderer import MarkdownRenderer
from finminutes.core.review_renderer import _serialize_minutes
from finminutes.core.summarizer import (
    ROLE_INTERVIEWEE,
    ROLE_INTERVIEWER,
    QAPair,
    StructuredGenerator,
    StructuredMinutes,
    asker_label,
    normalize_asker_role,
)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------


def test_normalize_role_maps_common_aliases():
    """LLM 会自由发挥，中文别名必须都能收敛到枚举。"""
    for raw in ("interviewer", "提问方", "访谈方", "投资方", "调研方"):
        assert normalize_asker_role(raw) == ROLE_INTERVIEWER, raw
    for raw in ("interviewee", "项目方", "企业方", "被访方", "创始人"):
        assert normalize_asker_role(raw) == ROLE_INTERVIEWEE, raw


def test_normalize_role_rejects_unknown_instead_of_guessing():
    """认不出来要返回空串——猜错的归属比留空更糟。"""
    for raw in ("", None, "unknown", "无", "双方", "说不清", "张三"):
        assert normalize_asker_role(raw) == "", raw


def test_normalize_role_is_case_insensitive():
    assert normalize_asker_role("Interviewer") == ROLE_INTERVIEWER


# ---------------------------------------------------------------------------
# 渲染标签
# ---------------------------------------------------------------------------


def test_asker_label_prefers_name_over_role():
    assert asker_label("张总", ROLE_INTERVIEWER) == "张总"


def test_asker_label_falls_back_to_role():
    assert asker_label("", ROLE_INTERVIEWER) == "访谈方"
    assert asker_label("", ROLE_INTERVIEWEE) == "项目方"


def test_asker_label_empty_when_nothing_known():
    assert asker_label("", "") == "（未指明）"
    assert asker_label("", "unknown") == "（未指明）"


# ---------------------------------------------------------------------------
# 解析链路
# ---------------------------------------------------------------------------


def test_build_minutes_parses_and_normalizes_role():
    minutes = StructuredGenerator._build_minutes(
        [],
        {"qa_pairs": [{"question": "产能多少？", "answer": "六千万元", "asker_role": "访谈方"}]},
    )
    assert minutes.qa_pairs[0].asker_role == ROLE_INTERVIEWER


def test_build_minutes_keeps_asker_role_empty_when_absent():
    """老产物没有这个字段，不能因为缺字段就报错或瞎填。"""
    minutes = StructuredGenerator._build_minutes(
        [], {"qa_pairs": [{"question": "产能多少？", "answer": "六千万元"}]}
    )
    assert minutes.qa_pairs[0].asker_role == ""
    assert minutes.qa_pairs[0].asker == ""


# ---------------------------------------------------------------------------
# 渲染与序列化
# ---------------------------------------------------------------------------


def test_renderer_prints_role_instead_of_unknown():
    minutes = StructuredMinutes(
        qa_pairs=[QAPair(question="返投比例多少？", answer="1.5 倍", asker_role=ROLE_INTERVIEWEE)]
    )
    out = MarkdownRenderer(minutes).render()
    assert "**项目方**：返投比例多少？" in out
    assert "（未指明）" not in out


def test_serialize_minutes_keeps_asker_role():
    minutes = StructuredMinutes(
        qa_pairs=[QAPair(question="q", answer="a", asker_role=ROLE_INTERVIEWER)]
    )
    data = _serialize_minutes(minutes)
    assert data["qa_pairs"][0]["asker_role"] == ROLE_INTERVIEWER


def test_formal_qa_text_marks_role():
    """成品稿的 LLM 输入要看到归属，否则写出来的问答还是没有主语。"""
    text = _format_qa_pairs([{"question": "产能多少？", "answer": "六千万元", "asker_role": ROLE_INTERVIEWER}])
    assert "Q（访谈方）：产能多少？" in text


def test_formal_qa_text_has_no_label_without_role():
    """角色未知时不要塞「（未指明）」这种噪音给下游模型。"""
    text = _format_qa_pairs([{"question": "产能多少？", "answer": "六千万元"}])
    assert "Q：产能多少？" in text
    assert "未指明" not in text
