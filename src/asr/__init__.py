"""ASR fine-tuning and evaluation for ATC speech (Whisper): data loading, collation, training, scoring.

Public API:
    - `normalize_for_scoring`: the scoring-time text normalization shared by every ASR comparison.
    - `load_split` / `WhisperCollator`: dataset slicing and on-the-fly feature extraction.
    - `TrainConfig` / `train` / `evaluate`: the fine-tuning run and test-set scoring (see `train.py`).
"""

from .data import WhisperCollator, load_split
from .text import expand_digits_to_atc_words, normalize_for_scoring
from .train import TrainConfig, evaluate, train

__all__ = ["WhisperCollator", "load_split", "expand_digits_to_atc_words", "normalize_for_scoring",
           "TrainConfig", "evaluate", "train"]
