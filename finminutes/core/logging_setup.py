import logging
import re
import sys

_API_KEY_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9]{20,})"),
    re.compile(r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9_\-]{16,}", re.IGNORECASE),
]


class ApiKeyRedactor(logging.Filter):
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled

    def filter(self, record):
        if not self.enabled:
            return True
        msg = record.getMessage()
        for pattern in _API_KEY_PATTERNS:
            msg = pattern.sub(r"\1***", msg)
        record.msg = msg
        record.args = ()
        return True


class TranscriptRedactor(logging.Filter):
    def __init__(self, enabled=True, max_len=100):
        super().__init__()
        self.enabled = enabled
        self.max_len = max_len

    def filter(self, record):
        if not self.enabled:
            return True
        msg = record.getMessage()
        if len(msg) > self.max_len:
            record.msg = msg[:self.max_len] + f"... (truncated {len(msg) - self.max_len} chars)"
            record.args = ()
        return True


def setup_logging(config: dict | None = None):
    if config is None:
        config = {}
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file")

    root = logging.getLogger("finminutes")
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    redact_api = log_cfg.get("redact_api_key", True)
    redact_transcript = log_cfg.get("redact_transcript", True)
    root.addFilter(ApiKeyRedactor(enabled=redact_api))
    root.addFilter(TranscriptRedactor(enabled=redact_transcript))

    return root
