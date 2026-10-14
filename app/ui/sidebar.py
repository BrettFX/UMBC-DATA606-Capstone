"""Sidebar: input source, entity model, and the optional AI assistant (API key gating)."""

from __future__ import annotations

import os

import streamlit as st

from app import config
from app.services import explain
from app.services.errors import AppError
from app.services.samples import list_samples


def input_panel():
    """Returns (data_bytes | None, filename, sample | None)."""
    st.sidebar.header("1 · Audio")
    samples = list_samples()
    modes = (["Sample clip"] if samples else []) + ["Upload file", "Record"]
    mode = st.sidebar.radio("Source", modes, horizontal=True, label_visibility="collapsed")
    if mode == "Sample clip":
        sample = st.sidebar.selectbox("Labeled sample", samples, format_func=lambda s: s.label)
        st.sidebar.caption("Test and validation clips have ground truth. The longest clip (training split) demonstrates splitting at pauses.")
        return sample.read_bytes(), sample.file, sample
    if mode == "Upload file":
        up = st.sidebar.file_uploader("Radio recording", type=list(config.ALLOWED_EXTENSIONS),
                                      help=f"Up to {config.MAX_UPLOAD_MB} MB, {config.MAX_DURATION_S / 60:.0f} min. Mono or stereo.")
        return (up.getvalue(), up.name, None) if up else (None, "", None)
    rec = st.sidebar.audio_input("Record a transmission")
    return (rec.getvalue(), "recording.wav", None) if rec else (None, "", None)


def ner_choice() -> str:
    names = [n for n, p in config.NER_DIRS.items() if (p / "meta.json").exists()] or [config.DEFAULT_NER]
    st.sidebar.header("2 · Entity model")
    choice = st.sidebar.selectbox("Entity extractor", names, label_visibility="collapsed")
    st.sidebar.caption(config.NER_BLURBS.get(choice, "") + " See the **Models** tab for the full model card.")
    return choice


def ai_panel() -> tuple[str | None, str]:
    """Renders the API-key box. Returns (api_key, model_id) when the key has been verified, else (None, model_id)."""
    st.sidebar.header("3 · AI assistant (optional)")
    ss = st.session_state
    ss.setdefault("llm_ready", False)
    ss.setdefault("llm_key_checked", "")
    model_label = st.sidebar.selectbox("Claude model", list(config.LLM_MODELS), index=list(config.LLM_MODELS).index(config.DEFAULT_LLM))
    model = config.LLM_MODELS[model_label]
    key = st.sidebar.text_input("Anthropic API key", type="password", value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                placeholder="sk-ant-...", help="Used only in this browser session; never saved to disk.")
    if ss.llm_ready and (key.strip() != ss.llm_key_checked or ss.get("llm_model_checked") != model):
        ss.llm_ready = False  # key or model changed: verify again
    if st.sidebar.button("Test key", use_container_width=True, disabled=not key.strip()):
        with st.sidebar, st.spinner("Checking key…"):
            try:
                explain.test_key(key, model)
                ss.llm_ready, ss.llm_key_checked, ss.llm_model_checked = True, key.strip(), model
            except AppError as e:
                ss.llm_ready = False
                st.sidebar.error(f"{e}" + (f" {e.hint}" if e.hint else ""))
    if ss.llm_ready:
        st.sidebar.success("Key verified. AI explanations enabled.")
    else:
        st.sidebar.caption("Add and test a key to enable plain-English explanations. Only transcript text is sent to Anthropic, never audio.")
    return (key.strip() if ss.llm_ready else None), model
