"""Word timings for a transcript: forced alignment of the *given* text against the audio with Whisper's cross-attention.

Uses the already-loaded CTranslate2 model (`model.align`, the same routine faster-whisper uses for word timestamps), so the
words match the transcript exactly and it only costs one encoder pass over a short window. Times are in seconds from the start
of the segment's audio. Any failure returns an empty list: the app then simply shows the transcript without highlighting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np

from app import config

logger = logging.getLogger(__name__)


@dataclass
class Word:
    text: str
    start: float
    end: float


def _window(duration_s: float) -> int:
    from src.asr.inference import DYNAMIC_WINDOWS, FULL_WINDOW_S, pick_window

    return pick_window(duration_s, DYNAMIC_WINDOWS) or FULL_WINDOW_S


def _voiced_span(y: np.ndarray, sr: int) -> tuple[float, float]:
    iv = librosa.effects.split(y, top_db=35, frame_length=512, hop_length=128)
    if len(iv) == 0:
        return 0.0, len(y) / sr
    return iv[0][0] / sr, iv[-1][1] / sr


def _even(words: list[str], t0: float, t1: float) -> list[Word]:
    """Fallback: spread the words over [t0, t1] in proportion to their length."""
    weights = np.array([max(len(w), 2) for w in words], dtype=float)
    edges = t0 + np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * (t1 - t0)
    return [Word(w, float(a), float(b)) for w, a, b in zip(words, edges[:-1], edges[1:])]


def align_words(asr, samples: np.ndarray, text: str, sr: int = config.SAMPLE_RATE) -> list[Word]:
    words = text.split()
    if not words or len(samples) < sr // 4:
        return []
    onset, offset = _voiced_span(samples, sr)
    try:
        from faster_whisper.audio import pad_or_trim

        model, tok = asr.model, asr._tokenizer
        window = _window(len(samples) / sr)
        feats = model.feature_extractor(samples.astype(np.float32))[:, : window * 100]
        encoded = model.encode(pad_or_trim(feats, length=window * 100))
        tokens = [t for t in tok.encode(" " + " ".join(words)) if t < tok.eot]
        aligned = model.find_alignment(tok, [tokens], encoded, num_frames=min(len(samples) // 160, window * 100))[0]
        aligned = [a for a in aligned if a["word"].strip()]
        if len(aligned) != len(words):
            raise ValueError(f"aligner produced {len(aligned)} words for {len(words)}")
        starts = np.array([a["start"] for a in aligned], dtype=float)
        ends = np.array([a["end"] for a in aligned], dtype=float)
    except Exception as exc:
        logger.warning("word alignment failed (%s); spreading words evenly over the speech", exc)
        return _even(words, onset, offset)
    # The aligner stretches words over silence (the first one from time 0, others across pauses). Cap each word at a plausible
    # length, as Whisper's own word-timestamp heuristic does, so the highlight only lights up while the word is spoken.
    med = float(np.median(ends - starts))
    cap = max(0.9, 3 * med)
    cap_first = max(0.7, 2 * med)
    if ends[0] - starts[0] > cap_first:
        starts[0] = ends[0] - cap_first
    ends = np.minimum(ends, starts + cap)
    starts[0] = max(starts[0], onset - 0.05)
    ends[-1] = min(ends[-1], offset + 0.05)
    ends = np.maximum(ends, starts + 0.04)
    for i in range(1, len(words)):  # monotone
        starts[i] = max(starts[i], starts[i - 1])
        ends[i] = max(ends[i], starts[i])
    return [Word(w, float(a), float(b)) for w, a, b in zip(words, starts, ends)]
