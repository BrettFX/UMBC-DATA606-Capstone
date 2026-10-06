#!/usr/bin/env python3
"""Experiment: does a shorter Whisper encoder window keep accuracy, and how much faster is it?

Whisper always encodes a 30 s window, so a 3 s clip wastes ~90% of the encoder's work. This feeds each clip into a
shortened window (10 s, 8 s, ...) with no further training and compares WER with the native 30 s window on exactly
the same clips (only clips that fit the shorter window are used). Runs on the GPU for speed of experimentation;
the CPU latency of a promising window is then measured by scripts/benchmark_inference.py.

Usage:
    python scripts/experiment_short_window.py [--model medium] [--n 300]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from asr import load_split, normalize_for_scoring  # noqa: E402
from asr.inference import HFTranscriber  # noqa: E402

MODELS = {"medium": REPO_ROOT / "models/whisper-medium-en-atc-finetuned-full-lora/final",
          "small": REPO_ROOT / "models/whisper-small-en-atc-finetuned-full/final"}


def wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    return float(jiwer.wer([normalize_for_scoring(r) for r in refs], [normalize_for_scoring(h) for h in hyps]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="medium")
    ap.add_argument("--n", type=int, default=300, help="Test utterances (stratified by source).")
    ap.add_argument("--windows", type=int, nargs="+", default=[20, 15, 10, 8, 6, 4])
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "res" / "benchmarks" / "experiments")
    args = ap.parse_args()

    ds = load_split(REPO_ROOT / "data/processed/combined_dataset", "test", limit=args.n, seed=42)
    audio = [np.asarray(r["audio"]["array"], dtype=np.float32) for r in ds]
    refs, ids, sources = ds["text"], ds["utterance_id"], ds["dataset_source"]
    dur = np.array([len(a) / 16_000 for a in audio])
    print(f"{args.model}: {len(audio)} utterances, durations mean {dur.mean():.1f}s, p90 {np.percentile(dur, 90):.1f}s, max {dur.max():.1f}s", flush=True)

    def run(window: int, keep: np.ndarray) -> tuple[list[str], float]:
        asr = HFTranscriber(MODELS[args.model], device="cuda", dtype="fp16", windows=(window,))
        for i in np.flatnonzero(keep)[:2]:
            asr.transcribe(audio[i])
        hyps, t0 = [], time.perf_counter()
        for i in np.flatnonzero(keep):
            hyps.append(asr.transcribe(audio[i]))
        return hyps, (time.perf_counter() - t0) / keep.sum()

    rows = []
    ref_hyps, _ = run(30, np.ones(len(audio), bool))
    ref_hyps = dict(zip(range(len(audio)), ref_hyps))
    for w in args.windows:
        keep = dur <= w
        idx = np.flatnonzero(keep)
        hyps, sec = run(w, keep)
        base = [ref_hyps[i] for i in idx]
        sub_refs = [refs[i] for i in idx]
        row = {"window_s": w, "clips": int(keep.sum()), "share_of_test": float(keep.mean()), "wer_30s": wer(sub_refs, base), "wer_window": wer(sub_refs, hyps),
               "gpu_s_per_clip": sec, "bad_clips": int(sum(wer([r], [h]) > 0.5 for r, h in zip(sub_refs, hyps))),
               "bad_clips_30s": int(sum(wer([r], [h]) > 0.5 for r, h in zip(sub_refs, base)))}
        rows.append(row)
        print(f"window {w:>2}s: {row['clips']} clips ({100 * row['share_of_test']:.0f}% of test) | WER 30s window {100 * row['wer_30s']:.2f}% -> {w}s window "
              f"{100 * row['wer_window']:.2f}% | clips with WER>50%: {row['bad_clips_30s']} -> {row['bad_clips']} | GPU {sec:.3f}s/clip", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"short_window_{args.model}_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"model": args.model, "n": args.n, "utterance_ids": ids, "rows": rows}, indent=2))
    print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
