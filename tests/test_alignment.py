"""ASR 时间戳解析 + 说话人对齐。

真实调 API 的路径没法在本机验证（没 key），但「响应怎么解析」「段怎么归人」
这两块是纯逻辑，可以完全离线覆盖——而它们恰恰是最容易出错的地方。
"""

import json
from types import SimpleNamespace

import pytest

from finminutes.core import diarization as d
from finminutes.core.asr_client import ASRClient, Segment, TimestampUnsupportedError


def _seg(start, end, text):
    return Segment(start=start, end=end, text=text)


# ---------------------------------------------------------------------------
# verbose_json 响应解析
# ---------------------------------------------------------------------------


def test_extract_segments_from_sdk_objects():
    """openai SDK 解析后是对象，字段靠属性访问。"""
    response = SimpleNamespace(
        segments=[
            SimpleNamespace(start=0.0, end=5.2, text=" 第一句 "),
            SimpleNamespace(start=5.2, end=9.8, text="第二句"),
        ]
    )
    segs = ASRClient._extract_segments(response)
    assert [s.text for s in segs] == ["第一句", "第二句"]
    assert segs[0].start == 0.0 and segs[1].end == 9.8


def test_extract_segments_from_real_sdk_type():
    """用 openai SDK 真实的 TranscriptionVerbose 构造响应，确认字段名没记错。

    SimpleNamespace 的用例只能证明「我按自己以为的字段名写了代码」，
    这个用例才能证明「字段名和服务端返回的一致」。
    """
    from openai.types.audio.transcription_verbose import TranscriptionVerbose

    response = TranscriptionVerbose.model_validate({
        "duration": 9.8,
        "language": "zh",
        "text": "第一句第二句",
        "segments": [
            {"id": 0, "start": 0.0, "end": 5.2, "text": " 第一句 ",
             "seek": 0, "temperature": 0.0, "avg_logprob": -0.1,
             "compression_ratio": 1.0, "no_speech_prob": 0.01, "tokens": []},
            {"id": 1, "start": 5.2, "end": 9.8, "text": "第二句",
             "seek": 0, "temperature": 0.0, "avg_logprob": -0.1,
             "compression_ratio": 1.0, "no_speech_prob": 0.01, "tokens": []},
        ],
    })
    segs = ASRClient._extract_segments(response)
    assert [s.text for s in segs] == ["第一句", "第二句"]
    assert (segs[0].start, segs[1].end) == (0.0, 9.8)


def test_extract_segments_from_raw_dict():
    """有些兼容实现直接返回原始 JSON dict。"""
    payload = {"segments": [{"start": 1.0, "end": 2.0, "text": "hi"}]}
    assert ASRClient._extract_segments(payload)[0].text == "hi"


def test_extract_segments_from_json_string():
    payload = json.dumps({"segments": [{"start": 0, "end": 1, "text": "字符串形态"}]})
    assert ASRClient._extract_segments(payload)[0].text == "字符串形态"


def test_extract_segments_applies_offset():
    """分片模式下要把片首时间加回去，否则所有分片都从 0 开始。"""
    payload = {"segments": [{"start": 0.0, "end": 2.0, "text": "x"}]}
    seg = ASRClient._extract_segments(payload, offset=600.0)[0]
    assert (seg.start, seg.end) == (600.0, 602.0)


def test_extract_segments_rejects_plain_text():
    """返回纯文本说明服务不支持 verbose_json，必须报错而不是静默降级。"""
    with pytest.raises(TimestampUnsupportedError, match="纯文本"):
        ASRClient._extract_segments("这是一整段纯文本转写")


def test_extract_segments_rejects_missing_field():
    with pytest.raises(TimestampUnsupportedError, match="probe_asr_timestamps"):
        ASRClient._extract_segments({"text": "只有全文，没有分段"})


def test_extract_segments_skips_empty_and_broken_items():
    payload = {
        "segments": [
            {"start": 0, "end": 1, "text": "  "},
            {"start": None, "end": 1, "text": "缺起点"},
            {"start": 1, "end": 2, "text": "有效"},
        ]
    }
    segs = ASRClient._extract_segments(payload)
    assert [s.text for s in segs] == ["有效"]


# ---------------------------------------------------------------------------
# 段归人
# ---------------------------------------------------------------------------


def _turns():
    return [d.Turn(0.0, 56.6, 0), d.Turn(57.1, 119.7, 1)]


def test_group_by_speaker_merges_consecutive_same_speaker():
    segments = [_seg(0, 10, "甲"), _seg(10, 20, "乙"), _seg(60, 70, "丙")]
    blocks = d.group_by_speaker(segments, _turns())
    assert [(b[0], b[1]) for b in blocks] == [(0, "甲乙"), (1, "丙")]
    assert blocks[0][3] == 20  # end 取块内最大


def test_group_by_speaker_returns_none_outside_turns():
    blocks = d.group_by_speaker([_seg(200, 210, "空隙")], _turns())
    assert blocks[0][0] is None


def test_group_by_speaker_skips_blank_text():
    assert d.group_by_speaker([_seg(0, 1, "  ")], _turns()) == []


# ---------------------------------------------------------------------------
# 生成带标签的转录稿
# ---------------------------------------------------------------------------


def test_labeled_transcript_uses_merge_speakers_format():
    """`preprocessor.merge_speakers()` 认的是「名字独占一行 + 冒号」。"""
    segments = [_seg(0, 20, "我们做滤波器封装。"), _seg(60, 90, "产能要扩多少？")]
    text = d.to_labeled_transcript(segments, _turns())
    assert text == "说话人0:\n我们做滤波器封装。\n\n说话人1:\n产能要扩多少？"


def test_labeled_transcript_absorbs_gap_segments_into_previous_block():
    """空隙里的段落并给前一块，不另立「未知」块——否则下游会误以为换人了。"""
    segments = [_seg(0, 20, "前半句"), _seg(200, 210, "后半句")]
    text = d.to_labeled_transcript(segments, _turns())
    assert text == "说话人0:\n前半句后半句"
    assert "未知" not in text


def test_labeled_transcript_accepts_custom_names():
    segments = [_seg(0, 20, "甲说"), _seg(60, 90, "乙说")]
    text = d.to_labeled_transcript(segments, _turns(), speaker_names={0: "访谈方", 1: "项目方"})
    assert text.startswith("访谈方:\n甲说")
    assert "项目方:\n乙说" in text


def test_labeled_transcript_merges_back_to_back_same_speaker():
    """0 → 空隙 → 0 这种序列，处理完仍然是同一个块。"""
    segments = [_seg(0, 20, "一"), _seg(200, 210, "二"), _seg(210, 220, "三")]
    text = d.to_labeled_transcript(segments, _turns())
    assert text == "说话人0:\n一二三"


def test_labeled_transcript_empty_input():
    assert d.to_labeled_transcript([], _turns()) == ""


def test_labeled_transcript_survives_preprocessor():
    """和 preprocessor 真的接得上，不只是格式看着像。"""
    from finminutes.core.preprocessor import Preprocessor

    segments = [_seg(0, 20, "第一句"), _seg(60, 90, "第二句")]
    text = d.to_labeled_transcript(segments, _turns(), speaker_names={0: "甲方", 1: "乙方"})
    merged = Preprocessor().merge_speakers(text)
    assert "甲方:" in merged and "乙方:" in merged


# ---------------------------------------------------------------------------
# 分片模式的时间偏移
# ---------------------------------------------------------------------------


class _FakeAudio:
    """pydub.AudioSegment 的最小替身：只需要 len()、切片和 export()。"""

    def __init__(self, ms):
        self._ms = ms

    def __len__(self):
        return self._ms

    def __getitem__(self, item):
        return _FakeAudio(item.stop - item.start)

    def export(self, path, format=None, bitrate=None):
        with open(path, "wb"):
            pass


def _fake_pydub(monkeypatch, total_ms):
    import sys
    import types

    mod = types.ModuleType("pydub")
    mod.AudioSegment = types.SimpleNamespace(from_file=lambda path: _FakeAudio(total_ms))
    monkeypatch.setitem(sys.modules, "pydub", mod)


def _bare_client(chunk_ms, overlap_ms):
    client = object.__new__(ASRClient)
    client.chunk_duration_ms = chunk_ms
    client.overlap_ms = overlap_ms
    client.max_retries = 1
    return client


def test_chunked_segments_accumulate_absolute_offsets(monkeypatch):
    """30 秒音频、10 秒一片、5 秒重叠 → 3 片，核心起点应是 0/10/20。

    每片都返回同样的本地时间 0-4s 和 6-9s，累加后必须变成三组不重叠的绝对时间。
    这是最容易算错的地方：漏加偏移会让所有分片都堆在 0 秒附近。
    """
    _fake_pydub(monkeypatch, total_ms=30_000)
    client = _bare_client(chunk_ms=10_000, overlap_ms=5_000)
    client._transcribe_chunk_segments = lambda chunk, lang, idx, hotwords=None: [
        Segment(0.0, 4.0, f"a{idx}"),
        Segment(6.0, 9.0, f"b{idx}"),
    ]

    segs = client._transcribe_chunked_segments("fake.mp3", "zh")

    assert len(segs) == 6
    assert [round(s.start, 1) for s in segs] == [0.0, 6.0, 10.0, 16.0, 20.0, 26.0]
    assert [s.text for s in segs] == ["a0", "b0", "a1", "b1", "a2", "b2"]


def test_chunked_segments_do_not_duplicate_overlap_region(monkeypatch):
    """落在重叠区（中点超出本片核心区间）的段要被丢掉，否则拼接后会重复。"""
    _fake_pydub(monkeypatch, total_ms=30_000)
    client = _bare_client(chunk_ms=10_000, overlap_ms=5_000)
    client._transcribe_chunk_segments = lambda chunk, lang, idx, hotwords=None: [
        Segment(0.0, 2.0, "核心"),      # 中点 1s，在核心区间内
        Segment(12.0, 14.0, "重叠区"),  # 中点 13s，超出本片核心 0-10s
    ]

    segs = client._transcribe_chunked_segments("fake.mp3", "zh")
    texts = [s.text for s in segs]
    assert texts.count("重叠区") == 0
    assert texts.count("核心") == 3  # 每片各留一个


def test_chunked_segments_sorted_by_start(monkeypatch):
    _fake_pydub(monkeypatch, total_ms=30_000)
    client = _bare_client(chunk_ms=10_000, overlap_ms=5_000)
    client._transcribe_chunk_segments = lambda chunk, lang, idx, hotwords=None: [
        Segment(6.0, 9.0, f"后{idx}"),
        Segment(0.0, 4.0, f"前{idx}"),
    ]
    segs = client._transcribe_chunked_segments("fake.mp3", "zh")
    assert [s.start for s in segs] == sorted(s.start for s in segs)


def test_chunked_segments_does_not_retry_timestamp_error(monkeypatch):
    """服务不支持时间戳时重试三次只是浪费时间和配额。"""
    _fake_pydub(monkeypatch, total_ms=30_000)
    client = _bare_client(chunk_ms=10_000, overlap_ms=5_000)
    client.max_retries = 3
    calls = {"n": 0}

    def boom(chunk, lang, idx, hotwords=None):
        calls["n"] += 1
        raise TimestampUnsupportedError("没有分段")

    client._transcribe_chunk_segments = boom
    with pytest.raises(TimestampUnsupportedError):
        client._transcribe_chunked_segments("fake.mp3", "zh")
    assert calls["n"] == 1
