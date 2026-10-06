"""Inference-class tests on tiny models (CPU only): the Hugging Face and CTranslate2 backends and the spaCy NER wrapper."""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TINY = "openai/whisper-tiny.en"


def tiny_dir(tmp_path):
    """A local copy of whisper-tiny.en (so the code path matches a model directory), or skip if not cached."""
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.WhisperForConditionalGeneration.from_pretrained(TINY)
        processor = transformers.WhisperProcessor.from_pretrained(TINY)
    except OSError:
        pytest.skip("whisper-tiny.en not cached")
    model.save_pretrained(tmp_path / "tiny")
    processor.save_pretrained(tmp_path / "tiny")
    return tmp_path / "tiny"


def test_hf_transcriber_returns_text_on_cpu_fp32_and_int8(tmp_path):
    from asr.inference import HFTranscriber

    d, audio = tiny_dir(tmp_path), np.zeros(16_000, dtype=np.float32)
    assert isinstance(HFTranscriber(d, device="cpu", dtype="fp32", threads=2).transcribe(audio), str)
    assert isinstance(HFTranscriber(d, device="cpu", int8_dynamic=True, threads=2).transcribe(audio), str)
    with pytest.raises(ValueError):
        HFTranscriber(d, device="cuda", int8_dynamic=True)


def test_ct2_conversion_is_idempotent_and_transcribes(tmp_path):
    pytest.importorskip("ctranslate2")
    pytest.importorskip("faster_whisper")
    from asr.inference import CT2Transcriber, convert_to_ct2

    d = tiny_dir(tmp_path)
    out = convert_to_ct2(d, tmp_path / "ct2", "int8")
    stamp = (out / "model.bin").stat().st_mtime_ns
    assert convert_to_ct2(d, tmp_path / "ct2", "int8") == out and (out / "model.bin").stat().st_mtime_ns == stamp  # no reconversion
    assert isinstance(CT2Transcriber(out, threads=2).transcribe(np.zeros(16_000, dtype=np.float32)), str)


def test_spacy_ner_wrapper_predicts_spans():
    model = ROOT / "models/ner/spacy-balanced/model-best"
    if not model.exists():
        pytest.skip("NER model not downloaded (run ./download-models.sh --only ner)")
    from ner.inference import SpacyNer

    ner = SpacyNer(model)
    ents = ner.predict("lufthansa four one alpha descend flight level one two zero")
    assert ("descend", "COMMAND") in {(e["text"], e["label"]) for e in ents}
    assert all(e["span"][1] > e["span"][0] for e in ents)
    assert len(ner.predict_many(["roger", "contact rhein one three two decimal four"])) == 2
