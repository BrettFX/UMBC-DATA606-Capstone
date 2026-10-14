"""Cached model loaders and the metadata the UI shows about them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from app import config
from app.services.errors import AppError


@dataclass
class ModelInfo:
    asr_name: str
    asr_backend: str
    asr_source: str
    ner_name: str
    ner_dir: str
    asr_test_wer: float | None
    ner_dev_f1: float | None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def missing_assets() -> list[str]:
    """Human-readable list of files the app needs but cannot find (empty if all is well)."""
    from src.asr.inference import DEFAULT_CT2_DIR, DEFAULT_MODEL_DIR

    out = []
    if not (DEFAULT_CT2_DIR / "model.bin").exists() and not (DEFAULT_MODEL_DIR / "config.json").exists():
        out.append(f"ASR model: {DEFAULT_CT2_DIR.relative_to(config.REPO_ROOT)}")
    if not (config.NER_DIRS[config.DEFAULT_NER] / "meta.json").exists():
        out.append(f"NER model: {config.NER_DIRS[config.DEFAULT_NER].relative_to(config.REPO_ROOT)}")
    return out


@st.cache_resource(show_spinner=False)
def load_asr():
    from src.asr.inference import best_transcriber

    try:
        return best_transcriber(device="cpu", threads=4)
    except FileNotFoundError as exc:
        raise AppError("The speech-recognition model is not installed.",
                       "Run ./download-models.sh --profile cpu, then restart the app.") from exc


@st.cache_resource(show_spinner=False)
def load_ner(name: str = config.DEFAULT_NER):
    from src.ner.inference import SpacyNer

    path = config.NER_DIRS[name]
    if not (path / "meta.json").exists():
        raise AppError(f"The entity-extraction model '{name}' is not installed.",
                       "Run ./download-models.sh --profile cpu, then restart the app.")
    return SpacyNer(path)


def model_info(ner_name: str = config.DEFAULT_NER) -> ModelInfo:
    from src.asr.inference import DEFAULT_CT2_DIR, DEFAULT_MODEL_DIR

    scores = _read_json(DEFAULT_MODEL_DIR.parent / "test_scores.json")
    ner_res = _read_json(config.NER_DIRS[ner_name].parent / "results.json")
    hist = ner_res.get("history") or []
    best = ner_res.get("best_epoch")
    f1 = next((h["dev_f1"] for h in hist if h.get("epoch") == best), None)
    return ModelInfo(asr_name=config.ASR_MODEL_NAME, asr_backend="CTranslate2 int8, CPU",
                     asr_source=DEFAULT_MODEL_DIR.parent.name, ner_name=ner_name,
                     ner_dir=str(config.NER_DIRS[ner_name].parent.name),
                     asr_test_wer=scores.get("wer"), ner_dev_f1=f1)
