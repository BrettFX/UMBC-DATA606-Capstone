"""Dataset slicing and batch collation for Whisper fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_from_disk

KEEP_COLUMNS = ["audio", "text", "utterance_id", "dataset_source"]


def load_split(dataset_dir: Path, split: str, *, limit: int | None = None, seed: int = 42) -> Dataset:
    """One split of the cached combined dataset, minus rows whose audio failed to decode.

    `limit` takes a seeded random sample stratified by `dataset_source` (equal share per source, so the
    smaller real-audio source is not drowned out by the simulated one); None keeps every row.
    """
    ds = load_from_disk(str(dataset_dir))[split]
    ok = [i for i, err in enumerate(ds["audio_decode_error"]) if not err]
    ds = ds.select(ok)
    if limit is not None and limit < len(ds):
        sources = np.array(ds["dataset_source"])
        rng = np.random.default_rng(seed)
        names = sorted(set(sources))
        picks = []
        for k, name in enumerate(names):
            idx = np.flatnonzero(sources == name)
            quota = limit // len(names) + (1 if k < limit % len(names) else 0)
            picks.extend(rng.choice(idx, size=min(quota, len(idx)), replace=False).tolist())
        ds = ds.select(sorted(picks))
    return ds.select_columns(KEEP_COLUMNS)


@dataclass
class WhisperCollator:
    """Turns raw `{audio, text}` rows into model inputs on the fly: log-mel features from the waveform and
    tokenized labels padded with -100 so the loss ignores padding. Computing features here, in the
    DataLoader workers, avoids precomputing an 80x3000 float32 array (~1 MB) per clip for the whole corpus.
    The decoder's own start token supplies BOS, so a leading BOS in the labels is dropped.
    """

    processor: object
    decoder_start_token_id: int
    feature_dtype: torch.dtype = torch.float32  # match the model's weight dtype (fp16 for a frozen LoRA base)

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        feats = self.processor.feature_extractor(
            [r["audio"]["array"] for r in rows], sampling_rate=16_000, return_tensors="pt").input_features
        labels = self.processor.tokenizer.pad(
            [{"input_ids": self.processor.tokenizer(r["text"]).input_ids} for r in rows], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.decoder_start_token_id).all():
            ids = ids[:, 1:]
        return {"input_features": feats.to(self.feature_dtype), "labels": ids}
