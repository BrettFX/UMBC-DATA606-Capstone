"""ASR fine-tuning, evaluation and inference for ATC speech (Whisper).

Public API (resolved lazily, so importing `asr.inference` on a CPU device needs neither torch nor transformers):
    - `normalize_for_scoring` / `expand_digits_to_atc_words`: scoring-time text normalization (`asr.text`).
    - `load_split` / `WhisperCollator`: dataset slicing and on-the-fly feature extraction (`asr.data`).
    - `TrainConfig` / `train` / `evaluate`: the fine-tuning run and test-set scoring (`asr.train`).
    - `asr.inference`: the transcription backends and `best_transcriber` (imported explicitly).
"""

import importlib

_EXPORTS = {"WhisperCollator": "data", "load_split": "data", "expand_digits_to_atc_words": "text", "normalize_for_scoring": "text",
            "TrainConfig": "train", "evaluate": "train", "train": "train"}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'asr' has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    for export, owner in _EXPORTS.items():  # bind every export of that module now, so `asr.train` stays the function
        if owner == module_name:
            globals()[export] = getattr(module, export)
    return globals()[name]
