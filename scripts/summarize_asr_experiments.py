#!/usr/bin/env python3
"""Summarize every ASR experiment scored on the full test split, with bootstrap confidence intervals.

Reads the per-row result CSVs in <data-dir>/processed/asr_results/ (finetuned_*_full*.csv and
comparison_*_full_test.csv), pairs each fine-tuned run with its training metadata under models/ when present,
and writes experiments_summary.csv and experiments_summary.md next to the CSVs. Corpus WER is edit-distance
weighted; confidence intervals resample utterances, and the paired comparison against a reference model
resamples the same utterances for both models, so it tests whether two models really differ.

Usage:
    python scripts/summarize_asr_experiments.py
    python scripts/summarize_asr_experiments.py --reference "jacktol/whisper-medium.en-fine-tuned-for-ATC"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from asr import normalize_for_scoring  # noqa: E402 -- import after sys.path setup

N_BOOT = 2000


def edit_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-utterance word errors (S+D+I) and reference length, so corpus WER is sum(errors) / sum(ref words)."""
    errs, words = [], []
    for ref, hyp in zip(df["reference"].map(normalize_for_scoring), df["hypothesis"].map(normalize_for_scoring)):
        if not ref:
            errs.append(len(hyp.split()))
            words.append(0)
            continue
        out = jiwer.process_words(ref, hyp)
        errs.append(out.substitutions + out.deletions + out.insertions)
        words.append(len(ref.split()))
    return pd.DataFrame({"utterance_id": df["utterance_id"].values, "source": df["dataset_source"].values,
                         "errors": errs, "words": words})


def wer(c: pd.DataFrame) -> float:
    return c["errors"].sum() / max(c["words"].sum(), 1)


def boot(c: pd.DataFrame, rng, other: pd.DataFrame | None = None) -> tuple[float, float]:
    """95% CI of WER (or of the WER difference c - other) by resampling utterances."""
    idx = rng.integers(0, len(c), size=(N_BOOT, len(c)))
    e, w = c["errors"].values[idx].sum(1), c["words"].values[idx].sum(1)
    stat = e / np.maximum(w, 1)
    if other is not None:  # same resampled utterances for both models (rows are aligned by utterance_id)
        stat = stat - other["errors"].values[idx].sum(1) / np.maximum(other["words"].values[idx].sum(1), 1)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def train_info(csv: Path, models_dir: Path) -> dict:
    """Training metadata for a finetuned_<model>_full<tag>.csv result, or {} for an off-the-shelf comparison."""
    stem = csv.stem.removeprefix("finetuned_")
    model, _, tag = stem.partition("_full")
    folder = models_dir / f"{model}-atc-finetuned-full{tag}"
    metrics = folder / "train_metrics.json"
    if not metrics.exists():
        return {}
    m = json.loads(metrics.read_text())
    cfg = json.loads((folder / "train_config.json").read_text())
    val = [h["eval_wer"] for h in m["log_history"] if "eval_wer" in h]
    return {"epochs_trained": round(m["train"]["epoch"], 1), "train_minutes": round(m["train"]["train_runtime"] / 60, 1),
            "best_val_wer": min(val) if val else None, "lora": bool(cfg.get("lora")), "learning_rate": cfg["learning_rate"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--reference", default=None,
                   help="model_version to compare every other model against (default: the first off-the-shelf comparison).")
    args = p.parse_args(argv)
    results = args.data_dir / "processed" / "asr_results"
    files = sorted([*results.glob("finetuned_*_full*.csv"), *results.glob("comparison_*_full_test.csv")])
    counts, info = {}, {}
    for f in files:
        df = pd.read_csv(f)
        label = df["model_version"].iloc[0]
        counts[label], info[label] = edit_counts(df).set_index("utterance_id"), train_info(f, REPO_ROOT / "models")
    ref_label = args.reference or next((m for m in counts if not info[m]), next(iter(counts)))
    rng = np.random.default_rng(0)
    rows = []
    for label, c in counts.items():
        for source in ["ALL", *sorted(c["source"].unique())]:
            sub = c if source == "ALL" else c[c["source"] == source]
            lo, hi = boot(sub, rng)
            row = {"model": label, "source": source, "n": len(sub), "wer": wer(sub), "wer_lo": lo, "wer_hi": hi}
            if label != ref_label:
                base = counts[ref_label].reindex(sub.index)
                dlo, dhi = boot(sub, rng, other=base)
                row.update(diff_vs_ref=wer(sub) - wer(base), diff_lo=dlo, diff_hi=dhi,
                           verdict="better" if dhi < 0 else "worse" if dlo > 0 else "no clear difference")
            rows.append({**row, **(info[label] if source == "ALL" else {})})
    out = pd.DataFrame(rows)
    out.to_csv(results / "experiments_summary.csv", index=False)

    lines = [f"ASR experiments on the full test split (reference model for comparisons: {ref_label})", ""]
    for source in ["ALL", *sorted(set(out["source"]) - {"ALL"})]:
        lines += [f"### {source}", "", "| model | n | WER (95% CI) | vs reference (95% CI) |", "|---|---|---|---|"]
        for r in out[out["source"] == source].sort_values("wer").itertuples():
            vs = ("reference" if r.model == ref_label else
                  f"{100 * r.diff_vs_ref:+.1f} pts ({100 * r.diff_lo:+.1f}, {100 * r.diff_hi:+.1f}): {r.verdict}")
            lines.append(f"| {r.model} | {r.n} | {100 * r.wer:.1f}% ({100 * r.wer_lo:.1f}-{100 * r.wer_hi:.1f}) | {vs} |")
        lines.append("")
    meta = out[(out["source"] == "ALL") & out["train_minutes"].notna()]
    if len(meta):
        lines += ["### Training", "", "| model | LoRA | learning rate | epochs trained | train minutes | best validation WER |", "|---|---|---|---|---|---|"]
        lines += [f"| {r.model} | {'yes' if r.lora else 'no'} | {r.learning_rate:g} | {r.epochs_trained} | {r.train_minutes} | {100 * r.best_val_wer:.2f}% |"
                  for r in meta.itertuples()]
    text = "\n".join(lines)
    (results / "experiments_summary.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
