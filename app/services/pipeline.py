"""Prepared audio -> transcript -> entities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app import config
from app.services import models
from app.services.align import Word, align_words
from app.services.audio_prep import PreparedAudio
from app.services.errors import AppError


@dataclass
class SegmentResult:
    start_s: float
    end_s: float
    text: str
    entities: list[dict]       # {text,label,span} with spans relative to `text`
    words: list[Word] = field(default_factory=list)   # times on the original recording's timeline (seconds)


@dataclass
class Result:
    segments: list[SegmentResult]
    asr_seconds: float
    ner_seconds: float
    ner_name: str
    info: models.ModelInfo = field(repr=False, default=None)

    @property
    def transcript(self) -> str:
        return " ".join(s.text for s in self.segments if s.text).strip()

    @property
    def entities(self) -> list[dict]:
        return [e for s in self.segments for e in s.entities]


def run(audio: PreparedAudio, ner_name: str = config.DEFAULT_NER, progress=None) -> Result:
    def tick(msg: str):
        if progress:
            progress(msg)

    tick("Loading models")
    asr, ner = models.load_asr(), models.load_ner(ner_name)
    texts, t0 = [], time.perf_counter()
    for i, seg in enumerate(audio.segments, 1):
        tick(f"Transcribing{f' segment {i}/{len(audio.segments)}' if len(audio.segments) > 1 else ''}")
        try:
            texts.append(asr.transcribe(seg.samples))
        except Exception as exc:
            raise AppError("Speech recognition failed on this audio.", f"Technical detail: {type(exc).__name__}: {exc}") from exc
    asr_s = time.perf_counter() - t0
    tick("Extracting entities")
    t1 = time.perf_counter()
    ents = ner.predict_many(texts) if texts else []
    ner_s = time.perf_counter() - t1
    tick("Aligning words")
    segs = []
    for s, t, e in zip(audio.segments, texts, ents):
        offset = audio.trim_start_s + s.start_s
        words = [Word(w.text, w.start + offset, w.end + offset) for w in align_words(asr, s.samples, t)]
        segs.append(SegmentResult(s.start_s, s.end_s, t, e, words))
    return Result(segs, asr_s, ner_s, ner_name, models.model_info(ner_name))
