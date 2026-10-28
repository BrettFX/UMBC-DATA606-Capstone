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


# ---- multi-segment clips: bounded transcript box and segment bands on the charts ----

def _result(n_segments):
    from types import SimpleNamespace as NS
    segs = [NS(start_s=i * 10.0, end_s=(i + 1) * 10.0, text="descend flight level eight zero", entities=[],
               words=[NS(start=i * 10.0 + k * 0.5, end=i * 10.0 + k * 0.5 + 0.4) for k in range(5)]) for i in range(n_segments)]
    return NS(segments=segs, transcript=" ".join(s.text for s in segs))


def test_transcript_box_scrolls_inside_a_height_cap_and_names_segments():
    from app.ui import live_transcript

    html = live_transcript.build_html(_result(8))
    assert "max-height" in html and "overflow-y:auto" in html and "position:sticky" in html
    assert html.count("class='ciq-seg'") == 8 and "<span class='ciq-now'>" in html  # a 'Segment k of n' slot for the playing segment


def test_single_segment_transcript_has_no_segment_labels():
    from app.ui import live_transcript

    html = live_transcript.build_html(_result(1))
    assert "class='ciq-seg'" not in html and "<span class='ciq-now'>" not in html


def test_segment_bands_only_for_multi_segment_audio():
    import numpy as np
    from app.ui import viz

    y = np.random.default_rng(0).uniform(-0.3, 0.3, 16000 * 3).astype("float32")
    assert not viz.waveform_fig(y, 16000).layout.shapes and not viz.waveform_fig(y, 16000, spans=[(0, 3)]).layout.shapes
    spans = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    fig = viz.waveform_fig(y, 16000, t0=0.5, spans=spans)
    xs = sorted(round(s.x0, 2) for s in fig.layout.shapes)
    assert xs == [1.5, 1.5, 2.5]  # shaded band 2 starts at 1.5 and the cuts (dotted lines) at 1.5 and 2.5, shifted by t0
    assert [a.text for a in fig.layout.annotations] == ["S1", "S2", "S3"]
    assert len(viz.mel_fig(y, 16000, spans=spans).layout.annotations) == 3


# ---- Transcript & entities tab for multi-segment clips ----

def _segments(n):
    from types import SimpleNamespace as NS
    text = "lufthansa four three nine zero descend flight level eight zero"
    ents = [{"text": "lufthansa four three nine zero", "label": "CALLSIGN", "span": [0, 30]},
            {"text": "descend", "label": "COMMAND", "span": [31, 38]}]
    return [NS(start_s=i * 10.0, end_s=(i + 1) * 10.0, text=text, entities=ents if i % 2 == 0 else []) for i in range(n)]


def test_multi_segment_clearance_cards_are_per_segment_in_a_scrolling_box():
    from app.ui import clearance_card

    html = clearance_card.segment_cards_html(_segments(6))
    assert "max-height:22rem" in html and "overflow-y:auto" in html
    assert html.count("Segment ") == 6 and html.count("AIRCRAFT") == 3  # only segments with entities get a card
    assert "No aviation entities in this segment." in html


def test_multi_segment_highlighted_transcript_is_one_bounded_document():
    from app.ui import clearance_card

    html = clearance_card.segments_displacy_html(_segments(12))
    assert html.count("class='seg'") == 12 and html.count("<html>") == 1
    assert clearance_card.MAX_MULTI_HEIGHT <= 400  # the iframe height is capped; the content scrolls inside it
