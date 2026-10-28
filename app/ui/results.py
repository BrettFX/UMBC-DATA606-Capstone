"""Result tabs: transcript + entities, audio analysis, evidence against ground truth, AI explanation."""

from __future__ import annotations

import html

import streamlit as st

from app import config
from app.services import explain, metrics, speech
from app.services.audio_prep import PreparedAudio
from app.services.errors import AppError
from app.services.pipeline import Result
from app.services.samples import Sample
from app.ui import clearance_card, playhead, viz


def show_error(e: AppError) -> None:
    st.error(f"**{e}**" + (f"  \n{e.hint}" if e.hint else ""))


def transcript_tab(result: Result) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words", len(result.transcript.split()))
    c2.metric("Entities", len(result.entities))
    c3.metric("Transcription time", f"{result.asr_seconds:.1f} s")
    c4.metric("Entity time", f"{result.ner_seconds * 1000:.0f} ms")
    present = {e["label"] for e in result.entities}
    st.subheader("Clearance")
    clearance_card.render_card(result.entities)
    st.subheader("Highlighted transcript")
    multi = len(result.segments) > 1
    for i, seg in enumerate(result.segments, 1):
        if multi:
            st.caption(f"Segment {i} · {seg.start_s:.1f}–{seg.end_s:.1f} s")
        if seg.text:
            clearance_card.render_displacy(seg.text, seg.entities)
        else:
            st.caption("(no speech recognized)")
    clearance_card.legend(present)
    st.text_area("Plain transcript (copy)", result.transcript, height=90)
    readback(result)
    mi = result.info
    with st.expander("Model details"):
        st.markdown(
            f"**Speech recognition:** {mi.asr_name} — {mi.asr_backend}  \n"
            f"**Model files:** `{mi.asr_source}`  \n"
            f"**Entity model:** {mi.ner_name} (`{mi.ner_dir}`)"
            + (f"  \n**Reported test WER:** {mi.asr_test_wer:.1%}" if mi.asr_test_wer is not None else "")
            + (f"  \n**Reported entity dev F1:** {mi.ner_dev_f1:.1%}" if mi.ner_dev_f1 is not None else ""))


def readback(result: Result) -> None:
    """Experimental: speak the transcript back with Kokoro TTS (post-process after ASR).

    Always shown once there is a transcript; when Kokoro is not installed the controls are disabled and a message says how to fix it.
    """
    if not result.transcript:
        return
    available = speech.is_available()
    with st.expander("Voice read-back (experimental)"):
        if not available:
            st.warning(speech.unavailable_message())
        st.caption("Synthesizes the recognized transcript with Kokoro-82M so you can hear what the model understood. "
                   "This is a text-to-speech step applied after recognition; it does not re-process the original audio.")
        c1, c2 = st.columns(2)
        voice = config.TTS_VOICES[c1.selectbox("Voice", list(config.TTS_VOICES), disabled=not available)]
        speed = c2.slider("Speed", 0.7, 1.3, 1.0, 0.05, disabled=not available)
        key = (result.transcript, voice, speed)
        cache = st.session_state.setdefault("readbacks", {})
        if st.button("Read back transcript", disabled=not available) and key not in cache:
            with st.spinner("Synthesizing…"):
                try:
                    cache[key] = speech.synthesize(result.transcript, voice, speed)
                except AppError as e:
                    show_error(e)
                    return
        if available and key in cache:
            st.audio(cache[key], format="audio/wav")


def audio_tab(audio: PreparedAudio, result: Result) -> None:
    st.caption("Press play on the audio player: the line follows playback in both charts. Click a chart to jump to that moment.")
    cfg = {"displayModeBar": False}
    st.plotly_chart(viz.waveform_fig(audio.samples, audio.sr, audio.trim_start_s), use_container_width=True, config=cfg)
    st.plotly_chart(viz.mel_fig(audio.samples, audio.sr, audio.trim_start_s), use_container_width=True, config=cfg)
    wpm = metrics.speech_rate_wpm(result.transcript, audio.duration_s)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{audio.duration_s:.1f} s")
    c2.metric("Speech rate", f"{wpm:.0f} wpm", help=metrics.DEFINITIONS["Speech rate"])
    c3.metric("SNR (est.)", f"{audio.snr_db:.0f} dB" if audio.snr_db is not None else "n/a", help=metrics.DEFINITIONS["SNR (est.)"])
    c4.metric("Original", f"{audio.original_sr / 1000:g} kHz · {audio.original_channels} ch")
    st.caption("Charts show the audio as given to the model (mono, 16 kHz, silence trimmed) on the original recording's timeline, so they line up with the player above.")


def evidence_tab(sample: Sample | None, result: Result) -> None:
    if sample is None:
        st.info("Ground-truth comparison is available for the bundled labeled samples. Select one in the sidebar.")
        return
    m = metrics.wer_cer(sample.ground_truth, result.transcript)
    es = metrics.entity_scores(sample.entities, result.entities) if sample.entities else None
    if sample.note:
        st.caption(sample.note)
    if sample.split != "test":
        st.caption(f"This clip comes from the {sample.split} split; the model may have seen {'it' if sample.split == 'train' else 'similar data'} "
                   "during development, so treat its scores as a demo rather than an evaluation.")
    c = st.columns(5)
    c[0].metric("WER", f"{m['wer']:.1%}", help=metrics.DEFINITIONS["WER"])
    c[1].metric("CER", f"{m['cer']:.1%}", help=metrics.DEFINITIONS["CER"])
    if es:
        c[2].metric("Entity precision", f"{es['precision']:.0%}", help=metrics.DEFINITIONS["Entity precision"])
        c[3].metric("Entity recall", f"{es['recall']:.0%}", help=metrics.DEFINITIONS["Entity recall"])
        c[4].metric("Entity F1", f"{es['f1']:.0%}", help=metrics.DEFINITIONS["Entity F1"])
    else:
        st.caption("No hand-labeled entities exist for this clip, so only transcript scores are shown.")
    st.markdown("**Ground truth**")
    st.write(sample.ground_truth)
    st.markdown("**Word-level differences** (red = in ground truth but missing/changed, green = extra in transcript)")
    parts = []
    for op, w in metrics.word_diff(sample.ground_truth, result.transcript):
        w = html.escape(w)
        parts.append({"=": w, "-": f"<span style='background:#C4755A55;text-decoration:line-through;padding:0 3px;border-radius:3px'>{w}</span>",
                      "+": f"<span style='background:#4E9A7F55;padding:0 3px;border-radius:3px'>{w}</span>"}[op])
    st.markdown(" ".join(parts), unsafe_allow_html=True)
    if sample.entities:
        st.markdown("**Ground-truth entities**")
        clearance_card.render_displacy(sample.ground_truth, sample.entities)
    with st.expander("How these scores are defined"):
        for k in ("WER", "CER", "Entity precision", "Entity recall", "Entity F1"):
            st.markdown(f"**{k}** — {metrics.DEFINITIONS[k]}")
        st.caption("Scores on one clip are anecdotal; see the model details for test-set results.")


def explain_tab(result: Result, api_key: str | None, model: str) -> None:
    st.caption("AI-generated interpretation of an automatic transcript. Verify against the audio and official procedures.")
    if not api_key:
        st.button("Explain this transmission", disabled=True, help="Add and test an Anthropic API key in the sidebar.")
        st.info("This feature is off until you add an Anthropic API key in the sidebar and press **Test key**.")
        return
    key = explain.cache_key(result.transcript, result.entities, model)
    cache = st.session_state.setdefault("explanations", {})
    if st.button("Explain this transmission", type="primary") and key not in cache:
        with st.spinner("Asking Claude…"):
            try:
                cache[key] = explain.explain(api_key, model, result.transcript, result.entities)
            except AppError as e:
                show_error(e)
                return
    ex = cache.get(key)
    if not ex:
        return
    st.markdown(f"#### {ex.summary}")
    if ex.instructions:
        st.markdown("**What was said, in order**")
        for item in ex.instructions:
            st.markdown(f"- {item.get('text', '')}")
    for title, items in (("Corrections or repeats", ex.corrections_or_repeats),
                         ("Unclear or possible recognition errors", ex.unclear_or_possible_errors)):
        if items:
            st.markdown(f"**{title}**")
            for it in items:
                st.markdown(f"- {it}")
    if ex.verbosity_note:
        st.markdown(f"**Wording:** {ex.verbosity_note}")


def about_tab() -> None:
    st.markdown(config.DISCLAIMER)
    st.markdown("""
**How it works.** Audio is converted to mono 16 kHz, trimmed and (if longer than 28 s) split at pauses. A Whisper
medium model fine-tuned on ATC speech transcribes it; a spaCy model tags callsigns, runways, altitudes, headings,
frequencies, squawk codes, instructions, fixes and facilities. Optionally, Claude explains the result in plain English,
and an experimental voice read-back (needs Kokoro; see the panel on the Transcript & entities tab) speaks the transcript back.

**What it is not.** It does not confirm clearances, detect conflicts, or raise safety alerts. Scores quoted here are
research measurements on held-out datasets (ATCOSIM and ATC-ASR), not operational performance figures.

**Privacy.** Audio is processed locally. If you enable the AI assistant, only the transcript text and entity tags are
sent to the Anthropic API using the key you provide; the key is kept in this browser session only.
""")
