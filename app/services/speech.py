"""Optional text-to-speech read-back of the transcript with Kokoro-82M (experimental).

A post-process step after ASR: the recognized text is synthesized to a clean, evenly paced voice so a listener can hear what the
model understood. Kokoro is an optional dependency (`pip install "kokoro>=0.9.4"`, plus `espeak-ng` on the system); when it is
missing the app simply hides the feature.
"""

from __future__ import annotations

import io
import re
import sys
import wave

import numpy as np
import streamlit as st

from app import config
from app.services.errors import AppError


def is_available() -> bool:
    try:
        import kokoro  # noqa: F401
    except ImportError:
        return False
    return True


def unavailable_message() -> str:
    """Markdown shown in the UI when Kokoro cannot be imported: what is wrong and how to fix it."""
    return (
        "**Voice read-back is unavailable.** The Kokoro text-to-speech package could not be imported by the Python "
        f"environment running this app (`{sys.executable}`).\n\n"
        "1. Install the project dependencies by running `./setup.sh`; it installs Kokoro and the `espeak-ng` system package.\n"
        "2. Restart the app from that environment: use `./run-app.sh`, or `conda activate data-science` and then "
        "`streamlit run app/main.py`.")


def prepare_text(transcript: str) -> str:
    """Corpus transcripts are lowercase spoken-form words; collapse whitespace so the synthesizer sees one clean sentence."""
    return re.sub(r"\s+", " ", transcript).strip()


def to_wav_bytes(samples: np.ndarray, sr: int = config.TTS_SAMPLE_RATE) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    from kokoro import KPipeline

    return KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")  # "a" = American English


def synthesize(transcript: str, voice: str = config.DEFAULT_TTS_VOICE, speed: float = 1.0, pipeline=None) -> bytes:
    """Returns a 24 kHz mono WAV of `transcript`. `pipeline` is injectable for tests."""
    text = prepare_text(transcript)
    if not text:
        raise AppError("There is no transcript to read back.")
    try:
        pipe = pipeline or _load_pipeline()
        chunks = [np.asarray(audio, dtype=np.float32) for _, _, audio in pipe(text, voice=voice, speed=speed) if audio is not None]
    except ImportError as exc:
        raise AppError("Voice read-back is not installed.", "Run ./setup.sh to install the project dependencies and restart the app from that environment.") from exc
    except Exception as exc:
        raise AppError("Voice read-back failed.", f"Technical detail: {type(exc).__name__}: {exc}") from exc
    if not chunks:
        raise AppError("The voice model produced no audio for this transcript.")
    return to_wav_bytes(np.concatenate(chunks))
