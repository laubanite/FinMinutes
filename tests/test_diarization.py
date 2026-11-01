"""说话人分离模块的纯逻辑部分（不依赖真模型）。

对齐、碎片合并、模型路径解析这些逻辑不需要跑推理，可以离线验证；
真正跑推理的 `diarize()` 在缺模型时会抛明确错误，这里也覆盖到。
"""

import os

from finminutes.core import diarization as d


# ---------------------------------------------------------------------------
# 对齐
# ---------------------------------------------------------------------------


def _turns():
    return [d.Turn(0.0, 56.6, 0), d.Turn(57.1, 119.7, 1)]


def test_assign_speaker_picks_max_overlap():
    assert d.assign_speaker(10, 12, _turns()) == 0
    assert d.assign_speaker(70, 72, _turns()) == 1


def test_assign_speaker_crosses_boundary_picks_longer_side():
    """跨边界的片段归给重叠更长的那一侧，而不是按起点。"""
    assert d.assign_speaker(56.0, 58.0, _turns()) == 1  # 0.6s vs 0.9s
    assert d.assign_speaker(55.0, 58.0, _turns()) == 0  # 1.6s vs 0.9s


def test_assign_speaker_returns_none_outside_turns():
    """落在所有人都不说话的空隙里，返回 None 而不是瞎猜。"""
    assert d.assign_speaker(200, 201, _turns()) is None


# ---------------------------------------------------------------------------
# 碎片合并
# ---------------------------------------------------------------------------


def test_merge_short_turns_absorbs_fragment():
    """实测 0.6s 的插话会被单独聚成一类，合并后应归给最近的邻居。"""
    turns = [d.Turn(0.0, 56.6, 0), d.Turn(30.3, 30.9, 1), d.Turn(57.1, 119.7, 2)]
    merged = d.merge_short_turns(turns, min_duration=1.0)
    assert d.speaker_count(merged) == 2
    assert [t.speaker for t in merged] == [0, 0, 2]


def test_merge_short_turns_keeps_long_turns_untouched():
    turns = [d.Turn(0.0, 10.0, 0), d.Turn(10.5, 20.0, 1)]
    assert d.merge_short_turns(turns) == turns


def test_merge_short_turns_handles_empty():
    assert d.merge_short_turns([]) == []


def test_merge_short_turns_all_fragments_returns_original():
    """全是碎片时不能返回空——宁可保留原样，也别把整段音频丢掉。"""
    turns = [d.Turn(0.0, 0.2, 0), d.Turn(0.3, 0.4, 1)]
    assert d.merge_short_turns(turns, min_duration=1.0) == turns


# ---------------------------------------------------------------------------
# 收口：重叠碎片 + 相邻同人
# ---------------------------------------------------------------------------


def test_coalesce_drops_nested_same_speaker_fragment():
    """实测输出里 56.6s 长段内嵌了一个 0.6s 同标签碎片，应收成一段。"""
    turns = [d.Turn(0.03, 56.61, 0), d.Turn(30.32, 30.95, 0)]
    assert d.coalesce(turns) == [d.Turn(0.03, 56.61, 0)]


def test_coalesce_merges_adjacent_same_speaker_within_gap():
    turns = [d.Turn(57.12, 89.33, 1), d.Turn(89.70, 104.71, 1)]  # 间隔 0.37s
    assert d.coalesce(turns) == [d.Turn(57.12, 104.71, 1)]


def test_coalesce_respects_custom_gap():
    turns = [d.Turn(0.0, 10.0, 0), d.Turn(12.0, 20.0, 0)]
    assert len(d.coalesce(turns)) == 2
    assert len(d.coalesce(turns, max_gap=3.0)) == 1


def test_coalesce_keeps_gap_larger_than_threshold():
    """停顿超过阈值说明是两段话，不要糊成一段。"""
    turns = [d.Turn(0.0, 10.0, 0), d.Turn(20.0, 30.0, 0)]
    assert len(d.coalesce(turns)) == 2


def test_coalesce_keeps_overlap_between_different_speakers():
    """不同人的重叠可能是真抢话，保留而不隐藏。"""
    turns = [d.Turn(0.0, 10.0, 0), d.Turn(5.0, 12.0, 1)]
    assert d.coalesce(turns) == turns


def test_coalesce_sorts_by_start_time():
    turns = [d.Turn(50.0, 60.0, 1), d.Turn(0.0, 10.0, 0)]
    assert [t.start for t in d.coalesce(turns)] == [0.0, 50.0]


def test_coalesce_handles_empty():
    assert d.coalesce([]) == []


# ---------------------------------------------------------------------------
# 可信度：单方独白时标签没有信息量
# ---------------------------------------------------------------------------


def test_dominant_share_balanced_is_about_half():
    """000-300s / 600-900s 两个真实窗口的形态：63/37、55/45。"""
    assert abs(d.dominant_share([d.Turn(0, 63, 0), d.Turn(63, 100, 1)]) - 0.63) < 1e-6


def test_dominant_share_detects_monologue():
    """300-600s 窗口的真实形态：289s vs 12s，占比 96%。"""
    share = d.dominant_share([d.Turn(0, 289, 0), d.Turn(289, 301, 1)])
    assert share > d.DOMINANCE_WARN


def test_dominant_share_handles_empty_and_zero_length():
    assert d.dominant_share([]) == 0.0
    assert d.dominant_share([d.Turn(1.0, 1.0, 0)]) == 0.0


def test_default_speaker_count_is_two():
    """访谈场景的领域先验。自动推断在实测中会分出 9-25 个说话人，不能当默认值。"""
    assert d.DEFAULT_NUM_SPEAKERS == 2
    assert d.DEFAULT_THRESHOLD >= 0.9


# ---------------------------------------------------------------------------
# 模型目录
# ---------------------------------------------------------------------------


def test_model_dir_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv(d.MODEL_DIR_ENV, str(tmp_path))
    open(os.path.join(tmp_path, d.SEGMENTATION_FILE), "wb").close()
    open(os.path.join(tmp_path, d.EMBEDDING_FILE), "wb").close()
    assert d.model_dir() == str(tmp_path)
    assert d.models_ready()


def test_model_dir_reports_not_ready_when_empty(monkeypatch, tmp_path):
    monkeypatch.setenv(d.MODEL_DIR_ENV, str(tmp_path))
    assert not d.models_ready()


def test_diarize_raises_clear_error_without_models(monkeypatch, tmp_path):
    """缺模型时必须给「去跑下载脚本」这种可执行的提示，而不是底层 onnx 报错。"""
    monkeypatch.setenv(d.MODEL_DIR_ENV, str(tmp_path))
    try:
        d.diarize("whatever.wav")
    except RuntimeError as exc:
        assert "download_diarization_models" in str(exc)
    else:
        raise AssertionError("应当抛出 RuntimeError")
