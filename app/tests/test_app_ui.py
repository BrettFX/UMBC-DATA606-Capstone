from pathlib import Path

from streamlit.testing.v1 import AppTest

MAIN = str(Path(__file__).resolve().parents[1] / "main.py")


def test_sample_flow_runs_without_exception():
    at = AppTest.from_file(MAIN, default_timeout=180)
    at.session_state['entered'] = True
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    # the first bundled sample is analyzed automatically
    assert len(at.tabs) >= 6
    assert any(m.label == "Entities" for m in at.metric)
    assert not at.exception


def test_stale_session_objects_are_recomputed():
    """After a code change, objects cached in session state by the old code lack new fields; the app must recompute, not crash."""
    from types import SimpleNamespace

    at = AppTest.from_file(MAIN, default_timeout=180)
    at.session_state["entered"] = True
    at.run()
    assert not at.exception
    # swap in stale objects, as an older version of the dataclasses would have left behind
    at.session_state["audio"] = SimpleNamespace(samples=None, sr=16000, warnings=[], applied_steps=[], segments=[])
    at.session_state["result"] = SimpleNamespace(segments=[], transcript="", entities=[])
    at.session_state["digest"] = at.session_state["auto"] = "from-an-older-version"
    at.run()
    assert not at.exception, at.exception
    assert any(m.label == "Entities" for m in at.metric)


def _readback_expander(at):
    return next(e for e in at.expander if e.label == "Voice read-back (experimental)")


def test_voice_readback_is_shown_but_disabled_when_kokoro_is_missing(monkeypatch):
    from app.services import speech

    monkeypatch.setattr(speech, "is_available", lambda: False)
    at = AppTest.from_file(MAIN, default_timeout=180)
    at.session_state["entered"] = True
    at.run()
    assert not at.exception, at.exception
    exp = _readback_expander(at)  # visible, not hidden
    warning = " ".join(w.value for w in exp.warning)
    assert "unavailable" in warning and "./setup.sh" in warning and "environment" in warning
    assert all(b.disabled for b in exp.button) and exp.button
    assert all(w.disabled for w in exp.selectbox) and all(w.disabled for w in exp.slider)


def test_voice_readback_is_enabled_when_kokoro_is_installed(monkeypatch):
    from app.services import speech

    monkeypatch.setattr(speech, "is_available", lambda: True)
    at = AppTest.from_file(MAIN, default_timeout=180)
    at.session_state["entered"] = True
    at.run()
    exp = _readback_expander(at)
    assert not list(exp.warning)
    assert not any(b.disabled for b in exp.button)
