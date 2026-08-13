import os
import shutil
import tempfile
import time

from openai import OpenAI

from finminutes.core.exceptions import FinMinutesError


class FFmpegMissingError(FinMinutesError):
    """切分转录需要 ffmpeg，但系统未安装。"""


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

    def __init__(self, config: dict):
        self.max_file_size_mb = config.get("max_file_size_mb", self.DEFAULT_MAX_FILE_SIZE_MB)
        self.chunk_duration_ms = config.get("chunk_duration_minutes", self.DEFAULT_CHUNK_DURATION_MIN) * 60 * 1000
        self.overlap_ms = config.get("overlap_seconds", self.DEFAULT_OVERLAP_SEC) * 1000
        self.max_retries = config.get("max_retries", self.DEFAULT_MAX_RETRIES)
        self._client = OpenAI(
            api_key=config.get("api_key", ""),
            base_url=config.get("base_url") or self.DEFAULT_BASE_URL,
        )
        self._model = config.get("model", "")

    def transcribe(self, audio_path: str, language: str = "zh", progress_callback=None) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        ext = os.path.splitext(audio_path)[1].lower()
        supported = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm")
        if ext not in supported:
            raise ValueError(f"不支持的音频格式: {ext}，支持的格式: {', '.join(supported)}")

        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb <= self.max_file_size_mb:
            return self._transcribe_single(audio_path, language)
        return self._transcribe_chunked(audio_path, language, file_size_mb, progress_callback)

    @staticmethod
    def _check_ffmpeg():
        if shutil.which("ffmpeg") is None:
            raise FFmpegMissingError()

    def test_connection(self) -> tuple:
        start = time.time()
        try:
            self._client.models.list()
            latency_ms = (time.time() - start) * 1000
            return (True, "OK", round(latency_ms, 1))
        except Exception as e:
            return (False, str(e), None)

    def _transcribe_single(self, audio_path: str, language: str) -> str:
        with open(audio_path, "rb") as f:
            response = self._client.audio.transcriptions.create(
                model=self._model,
                file=f,
                language=language,
                response_format="text",
            )
        return self._unwrap_response(response)

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

    def _transcribe_chunked(self, audio_path: str, language: str, file_size_mb: float, progress_callback=None) -> str:
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

            text = self._transcribe_chunk(chunk_audio, language, idx)
            chunks.append({"text": text})

            start = end - self.overlap_ms if end < duration_ms else duration_ms
            idx += 1

        if progress_callback:
            progress_callback("merging", 0, 0, 0, 0)

        return self._merge_chunks(chunks)

    def _transcribe_chunk(self, chunk_audio, language: str, chunk_index: int) -> str:
        tmp_path = None
        for attempt in range(self.max_retries):
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name
                chunk_audio.export(tmp_path, format="mp3", bitrate="64k")
                return self._transcribe_single(tmp_path, language)
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
