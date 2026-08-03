import os
import tempfile
from unittest.mock import MagicMock, patch

from finminutes.core.asr_client import ASRClient, GroqASRClient, SiliconFlowASRClient


class TestASRClientConfig:
    def test_groq_defaults(self):
        client = GroqASRClient({"api_key": "gsk_test"})
        assert client.max_file_size_mb == 25
        assert client.chunk_duration_ms == 10 * 60 * 1000
        assert client.overlap_ms == 5 * 1000
        assert client.max_retries == 3

    def test_groq_custom_config(self):
        client = GroqASRClient({
            "api_key": "gsk_test",
            "max_file_size_mb": 30,
            "chunk_duration_minutes": 15,
            "overlap_seconds": 8,
            "max_retries": 5,
        })
        assert client.max_file_size_mb == 30
        assert client.chunk_duration_ms == 15 * 60 * 1000
        assert client.overlap_ms == 8 * 1000
        assert client.max_retries == 5

    def test_siliconflow_defaults(self):
        client = SiliconFlowASRClient({"api_key": "sk_test"})
        assert client.max_file_size_mb == 50
        assert client.chunk_duration_ms == 10 * 60 * 1000
        assert client.overlap_ms == 5 * 1000
        assert client.max_retries == 3
        assert client._model == "FunAudioLLM/SenseVoiceSmall"

    def test_siliconflow_custom_threshold(self):
        client = SiliconFlowASRClient({
            "api_key": "sk_test",
            "max_file_size_mb": 80,
        })
        assert client.max_file_size_mb == 80

    def test_single_file_under_limit(self):
        client = GroqASRClient({"api_key": "gsk_test"})
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            with patch.object(client, "_transcribe_single", return_value="text") as single, \
                 patch.object(client, "_transcribe_chunked") as chunked:
                result = client.transcribe(tmp)
                assert result == "text"
                single.assert_called_once()
                chunked.assert_not_called()
        finally:
            os.unlink(tmp)

    def test_bad_extension(self):
        client = GroqASRClient({"api_key": "gsk_test"})
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            tmp = f.name
        try:
            import pytest
            with pytest.raises(ValueError):
                client.transcribe(tmp)
        finally:
            os.unlink(tmp)

    def test_missing_file(self):
        client = GroqASRClient({"api_key": "gsk_test"})
        import pytest
        with pytest.raises(FileNotFoundError):
            client.transcribe("nonexistent.mp3")


class TestFFmpegDetection:
    def test_check_ffmpeg_missing(self):
        client = GroqASRClient({"api_key": "gsk_test"})
        import pytest
        from finminutes.core.asr_client import FFmpegMissingError
        with patch("finminutes.core.asr_client.shutil.which", return_value=None):
            with pytest.raises(FFmpegMissingError):
                client._check_ffmpeg()

    def test_check_ffmpeg_present(self):
        client = GroqASRClient({"api_key": "gsk_test"})
        with patch("finminutes.core.asr_client.shutil.which", return_value="/usr/bin/ffmpeg"):
            client._check_ffmpeg()

    def test_no_ffmpeg_check_when_under_limit(self):
        """音频未超限时完全不触发 ffmpeg 检测。"""
        client = GroqASRClient({"api_key": "gsk_test"})
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            with patch("finminutes.core.asr_client.shutil.which", return_value=None) as which, \
                 patch.object(client, "_transcribe_single", return_value="text") as single:
                result = client.transcribe(tmp)
                assert result == "text"
                single.assert_called_once()
                which.assert_not_called()
        finally:
            os.unlink(tmp)

    def test_chunked_raises_when_ffmpeg_missing(self):
        """音频超限需切分时，缺失 ffmpeg 直接抛 FFmpegMissingError。"""
        client = GroqASRClient({"api_key": "gsk_test"})
        import pytest
        from finminutes.core.asr_client import FFmpegMissingError
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"\x00" * (30 * 1024 * 1024))
            tmp = f.name
        try:
            with patch("finminutes.core.asr_client.shutil.which", return_value=None):
                with pytest.raises(FFmpegMissingError):
                    client.transcribe(tmp)
        finally:
            os.unlink(tmp)


class TestMergeChunks:
    def test_merge_with_overlap(self):
        chunks = [
            {"text": "今天天气很好明天会下雨并且明天会有大风后天"},
            {"text": "并且明天会有大风后天天气晴朗适合出行"},
        ]
        assert ASRClient._merge_chunks(chunks) == "今天天气很好明天会下雨并且明天会有大风后天\n天气晴朗适合出行"

    def test_merge_no_overlap(self):
        chunks = [{"text": "第一段"}, {"text": "第二段"}]
        assert ASRClient._merge_chunks(chunks) == "第一段\n第二段"
