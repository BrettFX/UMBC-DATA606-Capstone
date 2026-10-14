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
