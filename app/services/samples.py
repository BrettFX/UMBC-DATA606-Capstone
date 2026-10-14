"""Bundled labeled sample clips (exported by scripts/build_app_samples.py)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from app import config


@dataclass
class Sample:
    id: str
    file: str
    source: str
    duration_s: float
    ground_truth: str
    entities: list[dict]
    split: str = "test"
    note: str = ""

    @property
    def label(self) -> str:
        src = {"atcosim": "ATCOSIM (simulated, clean)", "atc-asr-dataset": "ATC-ASR (real radio)"}.get(self.source, self.source)
        tag = {"test": "test", "validation": "validation", "train": "train"}.get(self.split, self.split)
        long = " · LONGEST" if self.duration_s > 30 else ""
        return f"{src} · {self.duration_s:.1f} s · {tag} #{self.id.split('-')[-1]}{long}"

    def read_bytes(self) -> bytes:
        return (config.SAMPLES_DIR / self.file).read_bytes()


def list_samples() -> list[Sample]:
    path = config.SAMPLES_DIR / "samples.json"
    if not path.exists():
        return []
    return [Sample(**r) for r in json.loads(path.read_text())]
