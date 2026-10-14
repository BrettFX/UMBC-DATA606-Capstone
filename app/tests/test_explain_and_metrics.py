from types import SimpleNamespace

import pytest

from app.services import explain, metrics
from app.services.errors import AppError


def test_metrics():
    assert metrics.wer_cer("one two three", "one two three")["wer"] == 0
    assert metrics.wer_cer("one two three", "one too three")["wer"] == pytest.approx(1 / 3)
    ents = [{"label": "RUNWAY", "text": "runway two four"}]
    assert metrics.entity_scores(ents, ents)["f1"] == 1
    assert metrics.entity_scores(ents, [])["recall"] == 0


def test_test_key_validation():
    with pytest.raises(AppError, match="Enter"):
        explain.test_key("", "m")
    with pytest.raises(AppError, match="look like"):
        explain.test_key("abc", "m")


def fake_client(text="", exc=None, stop="end_turn"):
    def create(**kw):
        if exc:
            raise exc
        return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_explain_ok(monkeypatch):
    out = '{"summary":"s","instructions":[{"text":"t","entities":[]}],"corrections_or_repeats":[],"unclear_or_possible_errors":[],"verbosity_note":""}'
    monkeypatch.setattr(explain, "_client", lambda k: fake_client(out))
    assert explain.explain("sk-ant-x", "m", "hello", []).summary == "s"


def test_explain_bad_json_and_auth(monkeypatch):
    import anthropic, httpx2

    monkeypatch.setattr(explain, "_client", lambda k: fake_client("not json"))
    with pytest.raises(AppError, match="unexpected format"):
        explain.explain("sk-ant-x", "m", "hello", [])
    resp = httpx2.Response(401, request=httpx2.Request("POST", "http://x"))
    err = anthropic.AuthenticationError("bad", response=resp, body=None)
    monkeypatch.setattr(explain, "_client", lambda k: fake_client(exc=err))
    with pytest.raises(AppError, match="rejected"):
        explain.test_key("sk-ant-x", "m")


def test_explain_empty_transcript():
    with pytest.raises(AppError):
        explain.explain("sk-ant-x", "m", "  ", [])
