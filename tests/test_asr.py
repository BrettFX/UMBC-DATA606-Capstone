"""CPU-only tests for the ASR pipeline pieces (text normalization, dataset slicing, collation)."""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asr import WhisperCollator, expand_digits_to_atc_words, load_split, normalize_for_scoring  # noqa: E402

DATASET = ROOT / "data" / "processed" / "combined_dataset"


def test_digits_expand_digit_by_digit():
    assert expand_digits_to_atc_words("descend flight level 660") == "descend flight level six six zero"
    assert expand_digits_to_atc_words("contact 118.7") == "contact one one eight decimal seven"
    assert expand_digits_to_atc_words("already spelled out") == "already spelled out"


def test_normalize_for_scoring_is_case_and_punctuation_insensitive():
    assert normalize_for_scoring("Descend, Flight Level 90!") == normalize_for_scoring("descend flight level nine zero")
    assert normalize_for_scoring("") == ""


@pytest.mark.skipif(not DATASET.exists(), reason="processed dataset not available")
def test_load_split_stratifies_by_source_and_is_reproducible():
    a = load_split(DATASET, "validation", limit=40, seed=1)
    b = load_split(DATASET, "validation", limit=40, seed=1)
    assert a["utterance_id"] == b["utterance_id"] and len(a) == 40
    counts = {s: a["dataset_source"].count(s) for s in set(a["dataset_source"])}
    assert sorted(counts.values()) == [20, 20]  # equal share per source
    assert set(a.column_names) == {"audio", "text", "utterance_id", "dataset_source"}


def test_collator_masks_padding_and_drops_leading_bos():
    transformers = pytest.importorskip("transformers")
    try:
        processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-small.en")
    except OSError:
        pytest.skip("whisper-small.en processor not cached")
    collate = WhisperCollator(processor, decoder_start_token_id=50257)
    rows = [{"audio": {"array": np.zeros(16_000, dtype=np.float32)}, "text": "descend flight level one two zero"},
            {"audio": {"array": np.zeros(8_000, dtype=np.float32)}, "text": "roger"}]
    batch = collate(rows)
    assert tuple(batch["input_features"].shape) == (2, 80, 3000)  # always a fixed 30 s window
    labels = batch["labels"]
    assert labels.shape[0] == 2 and (labels[1] == -100).any() and not (labels[0] == -100).all()
    assert (labels[:, 0] != 50257).all()  # the decoder start token is supplied by the model, not the labels


def test_collator_emits_features_in_the_requested_dtype():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-small.en")
    except OSError:
        pytest.skip("whisper-small.en processor not cached")
    rows = [{"audio": {"array": np.zeros(16_000, dtype=np.float32)}, "text": "roger"}]
    assert WhisperCollator(processor, 50257)(rows)["input_features"].dtype == torch.float32
    # a frozen fp16 LoRA base needs fp16 inputs outside autocast (generation during validation)
    assert WhisperCollator(processor, 50257, feature_dtype=torch.float16)(rows)["input_features"].dtype == torch.float16
