from finminutes.core.asr_client import ASRClient


def test_unwrap_json_response():
    """硅基流动 SenseVoice 返回 JSON 包裹，需解包 text 字段。"""
    assert ASRClient._unwrap_response('{"text": "你好世界"}') == "你好世界"


def test_unwrap_plain_text():
    """Groq 等返回纯文本，原样返回。"""
    assert ASRClient._unwrap_response("纯文本转录内容") == "纯文本转录内容"


def test_unwrap_invalid_json_stays_unchanged():
    assert ASRClient._unwrap_response('{"broken"') == '{"broken"'


def test_unwrap_json_without_text():
    assert ASRClient._unwrap_response('{"foo": "bar"}') == '{"foo": "bar"}'


def test_format_hotwords_empty():
    assert ASRClient._format_hotwords(None) == ""
    assert ASRClient._format_hotwords([]) == ""


def test_format_hotwords_contains_terms():
    prompt = ASRClient._format_hotwords(["热压键合", "流片"])
    assert "热压键合" in prompt
    assert "流片" in prompt


def test_format_hotwords_dedupes_and_skips_blank():
    prompt = ASRClient._format_hotwords(["流片", "流片", "", "  "])
    assert prompt.count("流片") == 1


def test_format_hotwords_respects_char_budget():
    many = [f"术语{i:03d}" for i in range(200)]
    prompt = ASRClient._format_hotwords(many)
    assert 0 < len(prompt) <= ASRClient.HOTWORD_PROMPT_MAX_CHARS


def test_hotword_prompt_disabled_by_default():
    client = object.__new__(ASRClient)
    client.hotword_prompt_enabled = False
    assert client._build_hotword_prompt(["热压键合"]) == ""


def test_hotword_prompt_enabled_returns_text():
    client = object.__new__(ASRClient)
    client.hotword_prompt_enabled = True
    assert "热压键合" in client._build_hotword_prompt(["热压键合"])


def test_groq_enables_hotword_prompt_by_default():
    from finminutes.core.asr_client import GroqASRClient

    assert GroqASRClient.DEFAULT_HOTWORD_PROMPT is True


def test_siliconflow_keeps_hotword_prompt_off():
    """SenseVoice 非 Whisper 系，不支持 prompt 参数，必须默认关闭。"""
    from finminutes.core.asr_client import SiliconFlowASRClient

    assert SiliconFlowASRClient.DEFAULT_HOTWORD_PROMPT is False
