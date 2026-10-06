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


def test_pick_window_chooses_the_smallest_window_that_fits():
    from asr.inference import DYNAMIC_WINDOWS, pick_window

    assert [pick_window(d, DYNAMIC_WINDOWS) for d in (0.5, 10.0, 10.1, 15.0, 20.5, 30.0, 31.0)] == [10, 10, 15, 15, 30, 30, None]


def test_dynamic_windows_work_for_both_backends_and_reject_overlong_clips(tmp_path):
    pytest.importorskip("faster_whisper")
    from asr.inference import CT2Transcriber, DYNAMIC_WINDOWS, HFTranscriber, convert_to_ct2

    d = tiny_dir(tmp_path)
    hf = HFTranscriber(d, device="cpu", threads=2, windows=DYNAMIC_WINDOWS)
    clips = [np.zeros(16_000 * s, dtype=np.float32) for s in (3, 12, 25)]  # one clip in each of the 10, 15 and 30 s windows
    assert all(isinstance(hf.transcribe(c), str) for c in clips)
    with pytest.raises(ValueError):
        hf.transcribe(np.zeros(16_000 * 35, dtype=np.float32))
    ct2 = CT2Transcriber(convert_to_ct2(d, tmp_path / "ct2", "int8"), threads=2, windows=DYNAMIC_WINDOWS)
    assert all(isinstance(ct2.transcribe(c), str) for c in clips)
    assert isinstance(ct2.transcribe(np.zeros(16_000 * 35, dtype=np.float32)), str)  # longer than any window: library fallback


def test_best_transcriber_uses_ctranslate2_int8_with_dynamic_windows_on_cpu(tmp_path):
    pytest.importorskip("faster_whisper")
    from asr.inference import CT2Transcriber, DYNAMIC_WINDOWS, best_transcriber

    asr = best_transcriber(tiny_dir(tmp_path), device="cpu", threads=2)
    assert isinstance(asr, CT2Transcriber) and asr.windows == DYNAMIC_WINDOWS
    assert (tmp_path / "ct2-int8" / "model.bin").exists()  # the converted copy is created next to the model
    assert isinstance(asr.transcribe(np.zeros(16_000, dtype=np.float32)), str)


def test_asr_inference_imports_without_torch_or_transformers():
    """A CPU device must be able to import the inference module without the heavy training dependencies."""
    import subprocess

    code = ("import sys; sys.path.insert(0, 'src'); import asr.inference; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules and 'datasets' not in sys.modules, sorted(m for m in sys.modules if m in ('torch', 'transformers', 'datasets'))")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-500:]


def test_lazy_asr_package_still_exposes_the_training_api():
    import asr

    assert callable(asr.train) and callable(asr.evaluate) and asr.TrainConfig.__name__ == "TrainConfig" and callable(asr.load_split)


def test_best_transcriber_on_cpu_needs_only_the_ct2_directory_and_explains_when_it_is_missing(tmp_path):
    pytest.importorskip("faster_whisper")
    from asr.inference import CT2Transcriber, best_transcriber, convert_to_ct2

    d = tiny_dir(tmp_path)
    ct2 = convert_to_ct2(d, tmp_path / "somewhere" / "ct2", "int8")
    assert (ct2 / "conversion.json").exists()  # provenance of the conversion
    asr = best_transcriber(tmp_path / "no-hf-model-here", ct2_dir=ct2, device="cpu", threads=2)  # a device with no Hugging Face model
    assert isinstance(asr, CT2Transcriber)
    with pytest.raises(FileNotFoundError, match="download-models.sh --profile cpu"):
        best_transcriber(tmp_path / "no-hf-model-here", ct2_dir=tmp_path / "missing", device="cpu")
