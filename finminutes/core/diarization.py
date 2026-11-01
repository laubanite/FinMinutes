"""说话人分离（diarization）：本地离线，零 API 调用。

## 定位

这一步**只解决「谁在什么时候说话」**，不做转写。转写仍由 ASR（Groq/SiliconFlow）负责，
两条流水线按时间轴对齐——业界几乎不把 ASR 和 diarization 做成一个模型。

这样设计的好处是 ASR 的免费额度照用，只多一个 36MB 的本地模型：
- pyannote segmentation-3.0（onnx，5.8MB）：切出「有人在说话」的片段
- 3D-Speaker CAM++（onnx，27MB）：给每个片段算声纹向量，再聚类成说话人

## 依赖

`sherpa-onnx` + `numpy`，都是可选依赖：

    pip install finminutes[diarization]

模型不进仓库（33MB），首次使用跑：

    python docs/tools/download_diarization_models.py

## 实测（某场 32 分钟访谈录音，4 线程 CPU，300 秒窗口）

耗时很不稳定：同一份 300 秒音频，机器空闲时 20 秒，忙的时候量到过 133 秒
（0.07x-0.44x 实时）。引擎构建只要 0.3-1.8 秒，瓶颈全在推理，是纯 CPU 单进程。
按区间算，1 小时音频约 4-15 分钟。

三个不重叠的 5 分钟窗口，对比「指定 2 人」和「自动推断」：

| 窗口 | 指定 2 人 | 自动 th=0.5 | 自动 th=0.9 | 自动 th=0.95 |
| --- | --- | --- | --- | --- |
| 000-300s | 2 人 63%/37% | **9 人** | 4 人 | 3 人 |
| 300-600s | 2 人 **96%/4%** | **13 人** | 4 人 | 4 人 |
| 600-900s | 2 人 55%/45% | **25 人** | 10 人 | 8 人 |

三条结论，都是踩过才知道的：

1. **自动推断（`num_speakers=-1`）在这个数据上根本不可用**。默认阈值 0.5 会分出
   9-25 个说话人；把阈值抬到 0.95 也还要 3-8 个。调阈值救不回来，只能指定人数。
   所以 `DEFAULT_NUM_SPEAKERS = 2`——访谈/调研录音的领域先验就是双方对话。
2. **指定 2 人在「有来有回」的窗口上很准**：000-300s 是 63%/37%、600-900s 是 55%/45%，
   和转录稿里「市局提问 → 项目方回答」的交替节奏对得上。
3. **指定 2 人在「单方长段」的窗口上会退化成 96%/4%**。这不是 bug：拿 300-600s 那个
   116 秒长段切成 8 秒窗做声纹相似度，任意两窗最低 0.74（同一人），而与另一方的
   相似度只有 0.16-0.25——那真的是同一个人连讲两分钟。
   但这意味着**标签在这种段落里没有信息量**，所以 `diarize()` 会在最大占比 > 90% 时告警，
   `dominant_share()` 供上层自行判断。

## 典型用法

    turns = diarize("会议.mp3")                  # 谁在什么时候说话
    turns = merge_short_turns(turns)             # 碎片改标签
    turns = coalesce(turns)                      # 相邻同人合成块、去掉嵌套碎片

    spk = assign_speaker(seg.start, seg.end, turns)   # 把一段 ASR 文本分给某人

或者直接用命令行：

    finminutes diarize -a 会议.mp3 -s 2

## 已知局限

- 抢话场景 DER 会升高；本模块只保证「区分问方/答方」，不保证人名准确
- 聚类编号（speaker 0/1）每次运行可能不同，不能跨文件比较
- 强制 2 人时若真实有 3 方（含 FA），会有两方被合并；宁可这样也不要 25 个标签
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MODEL_DIR_ENV = "FINMINUTES_MODEL_DIR"
SEGMENTATION_FILE = "seg_model.onnx"
EMBEDDING_FILE = "campplus.onnx"

# 阈值只在 num_speakers=-1 时生效。0.5 会分出 9-25 个说话人（实测），必须抬高。
DEFAULT_THRESHOLD = 0.9
DEFAULT_MIN_DURATION_ON = 0.3
DEFAULT_MIN_DURATION_OFF = 0.5
DEFAULT_THREADS = 4
# 访谈/调研录音的领域先验：双方对话。自动推断在这个数据上不可用，见模块 docstring。
DEFAULT_NUM_SPEAKERS = 2
# 单方发言占比超过这个值，就认为说话人标签没有区分价值。
DOMINANCE_WARN = 0.9


@dataclass
class Turn:
    """一段连续发言。speaker 是聚类编号，不代表真实身份。"""

    start: float
    end: float
    speaker: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def _sherpa():
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            "说话人分离需要 sherpa-onnx：pip install finminutes[diarization]"
        ) from exc
    import sherpa_onnx

    return sherpa_onnx


def available() -> bool:
    """依赖是否就位（不检查模型文件）。"""
    try:
        _sherpa()
        import numpy  # noqa: F401
    except Exception:
        return False
    return True


def model_dir() -> str:
    """模型目录：环境变量 > 用户缓存 > 项目内 models/。

    显式设置了 `FINMINUTES_MODEL_DIR` 就**只用它**，不静默回落到缓存目录——
    否则用户指错了路径会莫名其妙跑起来，排查起来很痛苦。
    """
    env = os.environ.get(MODEL_DIR_ENV)
    if env:
        return env
    cache = os.path.join(os.path.expanduser("~"), ".cache", "finminutes", "models")
    for path in (cache, os.path.join(os.getcwd(), "models")):
        if os.path.exists(os.path.join(path, SEGMENTATION_FILE)) and os.path.exists(
            os.path.join(path, EMBEDDING_FILE)
        ):
            return path
    return cache


def models_ready() -> bool:
    path = model_dir()
    return os.path.exists(os.path.join(path, SEGMENTATION_FILE)) and os.path.exists(
        os.path.join(path, EMBEDDING_FILE)
    )


def _load_waveform(audio_path: str) -> "object":
    """用 ffmpeg 统一转 16k 单声道，再用标准库 wave 读——不引入 soundfile/librosa。"""
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("需要 ffmpeg 解码音频（项目已有该依赖）")
    tmpdir = tempfile.mkdtemp(prefix="fm_diar_")
    wav_path = os.path.join(tmpdir, "audio.wav")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", audio_path,
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path],
            check=True,
        )
        with wave.open(wav_path) as w:
            frames = w.readframes(w.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def diarize(
    audio_path: str,
    num_speakers: int = DEFAULT_NUM_SPEAKERS,
    threshold: float = DEFAULT_THRESHOLD,
    min_duration_on: float = DEFAULT_MIN_DURATION_ON,
    min_duration_off: float = DEFAULT_MIN_DURATION_OFF,
    threads: int = DEFAULT_THREADS,
) -> list[Turn]:
    """对音频做说话人分离，返回按时间排序的 turn 列表。

    默认按双人对话切分。`num_speakers=3` 用于带 FA 的三方沟通，
    `num_speakers=-1` 是自动推断——**在本项目的访谈数据上会过度聚类**，
    只在音频不是访谈（比如多人圆桌）时才用。
    """
    if not available():
        raise RuntimeError("说话人分离需要 sherpa-onnx + numpy")
    if not models_ready():
        raise RuntimeError(
            f"模型未就位（{model_dir()}）。先运行：python docs/tools/download_diarization_models.py"
        )

    sherpa_onnx = _sherpa()
    samples = _load_waveform(audio_path)
    root = model_dir()

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=os.path.join(root, SEGMENTATION_FILE)
            ),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=os.path.join(root, EMBEDDING_FILE), num_threads=threads
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=threshold
        ),
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    if not config.validate():
        raise RuntimeError("diarization 配置校验失败，检查模型文件是否完整")

    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    result = engine.process(samples).sort_by_start_time()
    turns = [Turn(start=s.start, end=s.end, speaker=s.speaker) for s in result]

    share = dominant_share(turns)
    if share > DOMINANCE_WARN:
        logger.warning(
            "说话人分离退化：单方占比 %.0f%%。这段音频可能是单人长段发言，"
            "说话人标签无法区分问方/答方，下游不要据此判断角色。",
            share * 100,
        )
    return turns


def dominant_share(turns: list[Turn]) -> float:
    """说话时长最多的那个人占全部发言时长的比例，0~1。空输入返回 0。

    用来判断这次的说话人标签是否可信：接近 1 说明音频基本是单人独白，
    「谁是问方」这种判断没有依据。
    """
    if not turns:
        return 0.0
    totals: dict[int, float] = {}
    for turn in turns:
        totals[turn.speaker] = totals.get(turn.speaker, 0.0) + turn.duration
    total = sum(totals.values())
    if total <= 0:
        return 0.0
    return max(totals.values()) / total


def merge_short_turns(turns: list[Turn], min_duration: float = 1.0) -> list[Turn]:
    """把过短的碎片并给时间上最近的邻居，减少聚类噪声。

    实测 120 秒音频会多出 0.6s / 2.5s 的碎片被单独聚成一类；
    这些片段对「问方/答方」判断没有价值，合并掉更稳。
    """
    if not turns:
        return []
    keep = [t for t in turns if t.duration >= min_duration]
    if not keep:
        return turns
    merged: list[Turn] = []
    for turn in turns:
        if turn.duration >= min_duration:
            merged.append(Turn(turn.start, turn.end, turn.speaker))
            continue
        # 优先归给「时间上完全包住它」的 turn；没有再按端点距离找最近的。
        # 只按端点距离算会出错：碎片落在长 turn 内部时，它的起点离长 turn 的起点很远，
        # 反而更靠近后面那个 turn 的起点，会被错误地并过去。
        containing = [k for k in keep if k.start <= turn.start and turn.end <= k.end]
        if containing:
            nearest = min(containing, key=lambda k: k.duration)
        else:
            nearest = min(
                keep, key=lambda k: min(abs(k.start - turn.end), abs(turn.start - k.end))
            )
        merged.append(Turn(turn.start, turn.end, nearest.speaker))
    merged.sort(key=lambda t: t.start)
    return merged


def coalesce(turns: list[Turn], max_gap: float = 0.5) -> list[Turn]:
    """把同一说话人相邻的 turn 合成连续块，并丢掉被完全包住的碎片。

    原始输出会出现「56.6 秒的长段里嵌一个 0.6 秒同标签碎片」这种重叠——
    那是 segmentation 把一次咳嗽或气口切了出来，`merge_short_turns()` 只改标签、
    不动时间范围，所以看起来还是两行。给人看和拿去做对齐都别扭，这里收一次。

    不同说话人的重叠**保留**，那可能是真的抢话，藏起来反而失真。
    """
    if not turns:
        return []
    ordered = sorted(turns, key=lambda t: (t.start, -t.duration))
    out: list[Turn] = []
    for turn in ordered:
        if out:
            prev = out[-1]
            if turn.speaker == prev.speaker:
                if turn.start >= prev.start and turn.end <= prev.end:
                    continue  # 被同一人的长段完全包住
                if turn.start - prev.end <= max_gap:
                    out[-1] = Turn(prev.start, max(prev.end, turn.end), prev.speaker)
                    continue
        out.append(Turn(turn.start, turn.end, turn.speaker))
    return out


def assign_speaker(start: float, end: float, turns: list[Turn]) -> int | None:
    """按重叠时长把一段时间分配给某个说话人；无重叠返回 None。"""
    best: int | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best, best_overlap = turn.speaker, overlap
    return best


def speaker_count(turns: list[Turn]) -> int:
    return len({t.speaker for t in turns})


# ---------------------------------------------------------------------------
# 和 ASR 对齐
# ---------------------------------------------------------------------------


def group_by_speaker(segments, turns: list[Turn]) -> list[tuple[int | None, str, float, float]]:
    """把带时间戳的 ASR 分段按说话人合并成块。

    返回 `[(speaker, text, start, end)]`。speaker 为 None 表示这一段跟所有 turn
    都不重叠——通常是分片边界或纯噪声，调用方自己决定怎么处理。

    合并是「连续同人即合块」，所以一段长发言会变成一个块，而不是几十条碎句。
    """
    blocks: list[list] = []
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        speaker = assign_speaker(seg.start, seg.end, turns)
        if blocks and blocks[-1][0] == speaker:
            blocks[-1][1] += text
            blocks[-1][3] = max(blocks[-1][3], seg.end)
        else:
            blocks.append([speaker, text, seg.start, seg.end])
    return [(b[0], b[1], b[2], b[3]) for b in blocks]


def to_labeled_transcript(segments, turns: list[Turn],
                          speaker_names: dict[int, str] | None = None) -> str:
    """生成带说话人标签的转录稿，格式与 `preprocessor.merge_speakers()` 对齐。

    形如：

        说话人0:
        我们这边主要做滤波器封装……

        说话人1:
        产能扩张大概需要多少场地？

    落在空隙里的段落并给前一块——它多半是上一句话被切开的尾巴，
    单独立一个「未知」块反而让下游误以为换人了。
    """
    names = speaker_names or {}
    lines: list[str] = []
    last_speaker: int | None = None
    for speaker, text, _start, _end in group_by_speaker(segments, turns):
        if speaker is None:
            speaker = last_speaker
        if speaker is None:
            continue
        last_speaker = speaker
        label = names.get(speaker, f"说话人{speaker}")
        if lines and lines[-1][0] == label:
            lines[-1][1] += text
        else:
            lines.append([label, text])
    return "\n\n".join(f"{label}:\n{text}" for label, text in lines)
