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
