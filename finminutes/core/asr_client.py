import os
import shutil
import tempfile
import time
from dataclasses import dataclass

from openai import OpenAI

from finminutes.core.exceptions import FinMinutesError


class FFmpegMissingError(FinMinutesError):
    """切分转录需要 ffmpeg，但系统未安装。"""


class TimestampUnsupportedError(FinMinutesError):
    """ASR 服务没有返回分段/时间戳，无法和说话人分离结果对齐。"""


@dataclass
class Segment:
    """ASR 返回的一段转写，时间是相对整条音频的绝对秒数。"""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class ASRClient:
    """OpenAI 兼容 ASR 客户端基类。

    切分参数（文件大小、分片时长、重叠时长、重试次数）从配置读取，
    不再硬编码。
    """

    DEFAULT_MAX_FILE_SIZE_MB = 25
    DEFAULT_CHUNK_DURATION_MIN = 10
    DEFAULT_OVERLAP_SEC = 5
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_URL = ""

    # Whisper 系模型支持 prompt 条件偏置（解码阶段把术语当"前文上下文"），
    # 是非 Whisper 模型（如 SenseVoice）不支持的参数，故按 provider 默认关闭、可配置开启。
    DEFAULT_HOTWORD_PROMPT = False
    # Whisper prompt 上限 224 token，中文按 1 字≈1.5 token 估，留安全余量取 180 字符
    HOTWORD_PROMPT_MAX_CHARS = 180

    def __init__(self, config: dict):
        self.max_file_size_mb = config.get("max_file_size_mb", self.DEFAULT_MAX_FILE_SIZE_MB)
        self.chunk_duration_ms = config.get("chunk_duration_minutes", self.DEFAULT_CHUNK_DURATION_MIN) * 60 * 1000
        self.overlap_ms = config.get("overlap_seconds", self.DEFAULT_OVERLAP_SEC) * 1000
        self.max_retries = config.get("max_retries", self.DEFAULT_MAX_RETRIES)
        self.hotword_prompt_enabled = bool(
            config.get("hotword_prompt", self.DEFAULT_HOTWORD_PROMPT)
        )
        self._client = OpenAI(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url") or self.DEFAULT_BASE_URL,
        )
        self._model = config.get("model", "")

    def transcribe(self, audio_path: str, language: str = "zh", progress_callback=None,
                   hotwords: list[str] | None = None) -> str:
        """转写音频。

        hotwords — 术语表词条，用于解码阶段偏置（仅当 provider 支持 prompt 时生效）。
        上游偏置优于下游纠错：ASR 一旦把「热压键合」听成「热压结合」，
        后续只能靠猜；把正确词形提前喂给解码器，错字根本不产生。
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        ext = os.path.splitext(audio_path)[1].lower()
        supported = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm")
        if ext not in supported:
            raise ValueError(f"不支持的音频格式: {ext}，支持的格式: {', '.join(supported)}")

        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb <= self.max_file_size_mb:
            return self._transcribe_single(audio_path, language, hotwords=hotwords)
        return self._transcribe_chunked(
            audio_path, language, file_size_mb, progress_callback, hotwords=hotwords
        )

    @staticmethod
    def _check_ffmpeg():
        if shutil.which("ffmpeg") is None:
            raise FFmpegMissingError()

    # ------------------------------------------------------------------
    # 带时间戳的转写（给说话人分离对齐用）
    # ------------------------------------------------------------------

    def transcribe_segments(self, audio_path: str, language: str = "zh",
                            progress_callback=None,
                            hotwords: list[str] | None = None) -> list["Segment"]:
        """转写并返回带时间戳的分段，时间是相对整条音频的绝对秒数。

        和 `transcribe()` 走同一套切分逻辑，区别只是要 `verbose_json` 而不是纯文本。
        服务端不支持时会抛 `TimestampUnsupportedError`——别猜，直接告诉调用方。
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        ext = os.path.splitext(audio_path)[1].lower()
        supported = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm")
        if ext not in supported:
            raise ValueError(f"不支持的音频格式: {ext}，支持的格式: {', '.join(supported)}")

        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb <= self.max_file_size_mb:
            return self._transcribe_single_segments(audio_path, language, hotwords=hotwords)
        return self._transcribe_chunked_segments(
            audio_path, language, progress_callback, hotwords=hotwords
        )

    def _transcribe_single_segments(self, audio_path: str, language: str,
                                    hotwords: list[str] | None = None) -> list["Segment"]:
        kwargs = {}
        prompt = self._build_hotword_prompt(hotwords)
        if prompt:
            kwargs["prompt"] = prompt
        with open(audio_path, "rb") as f:
            response = self._client.audio.transcriptions.create(
                model=self._model,
                file=f,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                **kwargs,
            )
        return self._extract_segments(response)

    @staticmethod
    def _extract_segments(response, offset: float = 0.0) -> list["Segment"]:
        """从 verbose_json 响应里取出分段。

        三种形态都要认：SDK 解析后的对象、原始 dict、以及被当字符串返回的 JSON。
        拿不到 segments 就抛 TimestampUnsupportedError——静默降级成纯文本会让
        下游以为对齐成功，实际全部归到同一个人。
        """
        payload = response
        if isinstance(payload, str):
            text = payload.strip()
            if not text.startswith("{"):
                raise TimestampUnsupportedError(
                    "ASR 返回的是纯文本，没有分段。该服务不支持 verbose_json / "
                    "timestamp_granularities，无法做说话人对齐。"
                )
            import json

            try:
                payload = json.loads(text)
            except ValueError as exc:
                raise TimestampUnsupportedError(f"ASR 返回的 JSON 无法解析: {exc}") from exc

        if isinstance(payload, dict):
            raw = payload.get("segments")
        else:
            raw = getattr(payload, "segments", None)

        if not raw:
            raise TimestampUnsupportedError(
                "ASR 响应里没有 segments 字段。先跑 python docs/tools/probe_asr_timestamps.py "
                "确认这个服务支持 verbose_json + timestamp_granularities。"
            )

        out: list[Segment] = []
        for item in raw:
            if isinstance(item, dict):
                start, end, text = item.get("start"), item.get("end"), item.get("text", "")
            else:
                start, end, text = item.start, item.end, item.text
            if start is None or end is None:
                continue
            text = (text or "").strip()
            if not text:
                continue
            out.append(Segment(start=float(start) + offset, end=float(end) + offset, text=text))
        return out

    def _transcribe_chunked_segments(self, audio_path: str, language: str,
                                     progress_callback=None,
                                     hotwords: list[str] | None = None) -> list["Segment"]:
        """分片转写并累加时间偏移。

        分片之间有 overlap，直接拼接会出现重复段落。用「片段中点落在本片核心区间」
        来判定归属：每片的核心区间是 [片首, 片首+chunk_duration)，重叠区不重复计入。
        比按文本重叠去重稳，因为这里本来就有准确时间。
        """
        self._check_ffmpeg()

        try:
            from pydub import AudioSegment as PydubSegment
        except ImportError:
            raise ImportError("pydub 未安装，请执行: pip install pydub")

        if progress_callback:
            progress_callback("loading", 0, 0, 0, 0)

        audio = PydubSegment.from_file(audio_path)
        duration_ms = len(audio)
        total_chunks = (duration_ms + self.chunk_duration_ms - 1) // self.chunk_duration_ms

        collected: list[Segment] = []
        start = 0
        idx = 0
        while start < duration_ms:
            end = min(start + self.chunk_duration_ms, duration_ms)
            if end < duration_ms:
                end = min(end + self.overlap_ms, duration_ms)

            if progress_callback:
                progress_callback("chunk", idx + 1, total_chunks, start, end)

            local = self._transcribe_chunk_segments(
                audio[start:end], language, idx, hotwords=hotwords
            )
            # 注意：local 里的时间是**相对本片片首**的，所以核心区间也要用局部坐标。
            # 拿局部中点去比绝对时间是个很容易犯的错——从第二片起就全判错。
            offset_s = start / 1000.0
            core_len_s = self.chunk_duration_ms / 1000.0
            for seg in local:
                mid = (seg.start + seg.end) / 2
                if mid < core_len_s:
                    collected.append(
                        Segment(start=seg.start + offset_s,
                                end=seg.end + offset_s,
                                text=seg.text)
                    )

            start = end - self.overlap_ms if end < duration_ms else duration_ms
            idx += 1

        if progress_callback:
            progress_callback("merging", 0, 0, 0, 0)

        collected.sort(key=lambda s: s.start)
        return collected

    def _transcribe_chunk_segments(self, chunk_audio, language: str, chunk_index: int,
                                   hotwords: list[str] | None = None) -> list["Segment"]:
        tmp_path = None
        for attempt in range(self.max_retries):
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name
                chunk_audio.export(tmp_path, format="mp3", bitrate="64k")
                return self._transcribe_single_segments(tmp_path, language, hotwords=hotwords)
            except TimestampUnsupportedError:
                raise  # 重试没意义，是服务能力问题
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                raise
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except PermissionError:
                        pass

    def test_connection(self) -> tuple:
        start = time.time()
        try:
            self._client.models.list()
            latency_ms = (time.time() - start) * 1000
            return (True, "OK", round(latency_ms, 1))
        except Exception as e:
            return (False, str(e), None)

    def _transcribe_single(self, audio_path: str, language: str,
                           hotwords: list[str] | None = None) -> str:
        kwargs = {}
        prompt = self._build_hotword_prompt(hotwords)
        if prompt:
            kwargs["prompt"] = prompt
        with open(audio_path, "rb") as f:
            response = self._client.audio.transcriptions.create(
                model=self._model,
                file=f,
                language=language,
                response_format="text",
                **kwargs,
            )
        return self._unwrap_response(response)

    def _build_hotword_prompt(self, hotwords: list[str] | None) -> str:
        """按 provider 能力决定是否做热词偏置；不支持时返回空串（不传 prompt）。"""
        if not self.hotword_prompt_enabled:
            return ""
        return self._format_hotwords(hotwords)

    @classmethod
    def _format_hotwords(cls, hotwords: list[str] | None) -> str:
        """把术语词条拼成 Whisper 的 prompt（解码偏置）。

        返回空串表示无词条可用。按字符预算截断，避免超过 Whisper prompt
        的 224 token 上限后被服务端截断或报错。
        """
        if not hotwords:
            return ""
        prefix = "本次访谈涉及以下专有名词，请准确转写："
        budget = cls.HOTWORD_PROMPT_MAX_CHARS - len(prefix) - 1
        picked: list[str] = []
        used = 0
        for word in hotwords:
            word = (word or "").strip()
            if not word or word in picked:
                continue
            if used + len(word) + 1 > budget:
                break
            picked.append(word)
            used += len(word) + 1
        if not picked:
            return ""
        return f"{prefix}{'、'.join(picked)}。"

    @staticmethod
    def _unwrap_response(response) -> str:
        """部分 ASR（如硅基流动 SenseVoice）忽略 response_format 返回 JSON 包裹，
        需要解包 text 字段，否则后续 pipeline 会把 JSON 结构当正文。"""
        if isinstance(response, str) and response.strip().startswith("{"):
            try:
                import json

                data = json.loads(response)
                text = data.get("text", "")
                if text:
                    return text
            except Exception:
                pass
        return response

    def _transcribe_chunked(self, audio_path: str, language: str, file_size_mb: float,
                            progress_callback=None, hotwords: list[str] | None = None) -> str:
        self._check_ffmpeg()

        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pydub 未安装，请执行: pip install pydub")

        if progress_callback:
            progress_callback("loading", 0, 0, 0, 0)

        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)

        total_chunks = (duration_ms + self.chunk_duration_ms - 1) // self.chunk_duration_ms

        chunks = []
        start = 0
        idx = 0
        while start < duration_ms:
            end = min(start + self.chunk_duration_ms, duration_ms)
            if end < duration_ms:
                end = min(end + self.overlap_ms, duration_ms)

            chunk_audio = audio[start:end]

            if progress_callback:
                progress_callback("chunk", idx + 1, total_chunks, start, end)

            text = self._transcribe_chunk(chunk_audio, language, idx, hotwords=hotwords)
            chunks.append({"text": text})

            start = end - self.overlap_ms if end < duration_ms else duration_ms
            idx += 1

        if progress_callback:
            progress_callback("merging", 0, 0, 0, 0)

        return self._merge_chunks(chunks)

    def _transcribe_chunk(self, chunk_audio, language: str, chunk_index: int,
                          hotwords: list[str] | None = None) -> str:
        tmp_path = None
        for attempt in range(self.max_retries):
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name
                chunk_audio.export(tmp_path, format="mp3", bitrate="64k")
                return self._transcribe_single(tmp_path, language, hotwords=hotwords)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                raise
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except PermissionError:
                        pass

    @staticmethod
    def _merge_chunks(chunks: list) -> str:
        merged = []
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            if i == 0:
                merged.append(text)
            else:
                overlap = ASRClient._find_overlap(merged[-1], text)
                if overlap:
                    text = text[len(overlap):].lstrip()
                merged.append(text)
        return "\n".join(merged)

    @staticmethod
    def _find_overlap(text1: str, text2: str, min_overlap: int = 10) -> str:
        tail = text1[-200:]
        head = text2[:200]
        for i in range(min(len(tail), len(head)), min_overlap - 1, -1):
            if tail[-i:] == head[:i]:
                return head[:i]
        return ""


class GroqASRClient(ASRClient):
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    # Whisper 系模型支持 prompt 条件偏置，默认开启热词注入
    DEFAULT_HOTWORD_PROMPT = True

    def __init__(self, config: dict):
        cfg = dict(config)
        cfg.setdefault("max_file_size_mb", 25)
        cfg.setdefault("chunk_duration_minutes", 10)
        cfg.setdefault("overlap_seconds", 5)
        cfg.setdefault("max_retries", 3)
        cfg.setdefault("model", "whisper-large-v3-turbo")
        super().__init__(cfg)


class SiliconFlowASRClient(ASRClient):
    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

    def __init__(self, config: dict):
        cfg = dict(config)
        cfg.setdefault("max_file_size_mb", 50)
        cfg.setdefault("chunk_duration_minutes", 10)
        cfg.setdefault("overlap_seconds", 5)
        cfg.setdefault("max_retries", 3)
        cfg.setdefault("model", "FunAudioLLM/SenseVoiceSmall")
        super().__init__(cfg)
