"""ClearanceIQ — academic research prototype: ATC speech → transcript → aviation entities → plain-English explanation.

Run from the repo root:  streamlit run app/main.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app import config  # noqa: E402
from app.services import audio_prep, models, pipeline  # noqa: E402
from app.services.errors import AppError  # noqa: E402
from app.ui import live_transcript, model_card, playhead, results, sidebar, splash  # noqa: E402

st.set_page_config(page_title="ClearanceIQ", page_icon="✈️", layout="wide")

_THEME_CSS = """
<style>
.stApp {background:radial-gradient(ellipse at 75% 0%,#0d2a4d 0%,#06142a 42%,#030c18 100%) fixed}
[data-testid="stHeader"] {background:transparent}
/* Streamlit wraps the keyed container in a layout wrapper that is only as tall as the header, so the wrapper must be the sticky element */
[data-testid="stLayoutWrapper"]:has(> .st-key-ciq_header) {position:sticky; top:0; z-index:1000}
.st-key-ciq_header {margin:0 -1rem 0.6rem; padding:.55rem 1rem .5rem;
  background:linear-gradient(180deg,rgba(4,14,28,.97) 0%,rgba(5,17,34,.93) 100%); backdrop-filter:blur(8px);
  border-bottom:1px solid #15304f; box-shadow:0 8px 24px rgba(0,0,0,.35)}
.st-key-ciq_header [data-testid="stMarkdownContainer"] p {margin:0}
.ciq-brand {display:flex; align-items:baseline; gap:18px; flex-wrap:wrap}
.ciq-brand-name {font-size:1.7rem; font-weight:700; letter-spacing:.03em; color:#eaf5ff; text-shadow:0 0 26px rgba(86,180,233,.35)}
.ciq-brand-tag {font-size:.85rem; opacity:.65}
.stButton > button[kind="primary"] {background:linear-gradient(135deg,#2a77a8,#4a9bc9); border:0; color:#f0f7fc; font-weight:600;
  letter-spacing:.04em; box-shadow:0 0 12px rgba(86,160,210,.22)}
.stButton > button[kind="primary"]:hover {box-shadow:0 0 20px rgba(86,160,210,.4); filter:brightness(1.08)}
[data-testid="stMetric"] {background:rgba(86,180,233,.06); border:1px solid #1d3a5f; border-radius:10px; padding:10px 14px}
[data-testid="stTabs"] [aria-selected="true"] {color:#6aaed4}
[data-testid="stAlert"] {border:1px solid rgba(120,200,255,.25); border-radius:10px}
[data-testid="stAlertContentWarning"] {color:#eadbb8}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {background:rgba(190,150,80,.10) !important}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {background:rgba(190,150,80,.09); border-color:rgba(190,150,80,.35)}
</style>
"""


def _schema_fingerprint() -> str:
    """Changes whenever the cached result classes change shape. Objects kept in session state by an older version of the code
    (e.g. after a hot reload) would otherwise lack new fields and crash the UI; with a new fingerprint they are recomputed."""
    from dataclasses import fields

    classes = (audio_prep.PreparedAudio, audio_prep.Segment, pipeline.Result, pipeline.SegmentResult, models.ModelInfo)
    return hashlib.sha1(repr([[f.name for f in fields(c)] for c in classes]).encode()).hexdigest()[:10]


def header() -> None:
    with st.container(key="ciq_header"):  # pinned to the top while scrolling (see _THEME_CSS)
        st.markdown("<div class='ciq-brand'><span class='ciq-brand-name'>ClearanceIQ</span>"
                    "<span class='ciq-brand-tag'>Radio transmission → transcript → structured clearance</span></div>",
                    unsafe_allow_html=True)
    st.warning("**Academic research prototype.** Not certified or validated for operational use; "
               "output is automatic and may be wrong. Always verify against the audio.", icon="⚠️")


def main() -> None:
    ss = st.session_state
    if not ss.get("entered"):
        splash.render_splash()
        return
    # one element on every run, so later reruns do not shift the widget tree and reset the active tab
    st.html(_THEME_CSS + (splash.ENTER_CSS if ss.pop("just_entered", False) else ""))
    header()
    missing = models.missing_assets()
    if missing:
        st.error("**Setup required.** These model files were not found:\n\n" + "\n".join(f"- `{m}`" for m in missing)
                 + "\n\nRun `./download-models.sh --profile cpu` from the repository root (it falls back to a public Google Drive bundle "
                 "when you have no AWS access), then reload this page.")
        st.stop()

    data, filename, sample = sidebar.input_panel()
    ner_name = sidebar.ner_choice()
    api_key, llm_model = sidebar.ai_panel()
    st.sidebar.divider()
    st.sidebar.caption(config.DISCLAIMER.split(". ")[0] + ".")

    if data is None:
        st.info("Choose a sample clip, upload a recording, or record one in the sidebar to begin.")
        with st.expander("About ClearanceIQ", expanded=True):
            results.about_tab()
        return

    st.audio(data, format="audio/wav" if filename.endswith(".wav") else None)
    live_box = st.container()  # transcript lives right under the player; a fixed container keeps the layout below it stable
    digest = hashlib.sha1(data).hexdigest() + ner_name + _schema_fingerprint()
    ss = st.session_state

    try:
        if ss.get("digest") != digest:
            for k in ("audio", "result"):
                ss.pop(k, None)
            ss.digest = digest
        if "audio" not in ss:
            ss.audio = audio_prep.prepare(data, filename)
        audio = ss.audio
    except AppError as e:
        results.show_error(e)
        return

    for w in audio.warnings:
        st.warning(w)
    if audio.applied_steps:
        st.caption("Processing applied: " + " · ".join(audio.applied_steps))

    run_clicked = st.button("Analyze", type="primary")
    status_box = st.container()  # always present, so the progress widget never shifts the tabs below it
    if run_clicked or ("result" not in ss and sample is not None and ss.get("auto") != digest):
        try:
            with status_box, st.status("Analyzing…", expanded=False) as status:
                ss.result = pipeline.run(audio, ner_name, progress=lambda m: status.update(label=m + "…"))
                status.update(label="Done", state="complete")
            ss.auto = digest
        except AppError as e:
            ss.auto = digest  # remember the failure so the auto-run does not retry on every rerun
            results.show_error(e)
            return
        except Exception as e:  # last resort: never show a traceback to the user
            ss.auto = digest
            st.error(f"**Something went wrong while analyzing this audio.** ({type(e).__name__})")
            return

    if "result" not in ss:
        st.info("Press **Analyze** to transcribe this clip.")
        return
    with live_box:
        live_transcript.render(ss.result)
        playhead.inject()
    tabs = st.tabs(["Audio analysis", "Transcript & entities", "Evidence", "Explain (AI)", "Models", "About"])
    with tabs[0]:
        results.audio_tab(audio, ss.result)
    with tabs[1]:
        results.transcript_tab(ss.result)
    with tabs[2]:
        results.evidence_tab(sample, ss.result)
    with tabs[3]:
        results.explain_tab(ss.result, api_key, llm_model)
    with tabs[4]:
        model_card.models_tab(ner_name)
    with tabs[5]:
        results.about_tab()


main()
