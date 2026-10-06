#!/usr/bin/env python3
"""Paired accuracy analysis of a benchmark run: is each configuration's WER really different from a reference's?

Reads the per-utterance transcripts saved in a run under res/benchmarks/runs/, looks up the reference text by
utterance id, and writes res/benchmarks/experiments/accuracy_<run-id>.md with WER, bootstrap confidence intervals and
paired differences against the reference configuration, overall and for real versus simulated audio.

Usage:
    python scripts/analyze_benchmark_accuracy.py --tag accuracy-300
    python scripts/analyze_benchmark_accuracy.py --run res/benchmarks/runs/<file>.json --reference medium/hf-fp16/gpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarking import paired_wer_table, render_paired  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=None, help="A run JSON (default: the newest run with --tag).")
    ap.add_argument("--tag", default="accuracy-300")
    ap.add_argument("--reference", default="medium/hf-fp16/gpu", help="Configuration the others are compared against.")
    args = ap.parse_args()
    runs = REPO_ROOT / "res/benchmarks/runs"
    path = args.run or sorted(runs.glob(f"*_{args.tag}.json"))[-1]
    run = json.loads(path.read_text())

    from datasets import load_from_disk

    test = load_from_disk(str(REPO_ROOT / "data/processed/combined_dataset"))["test"].select_columns(["utterance_id", "text", "dataset_source"])
    by_id = {u: (t, s) for u, t, s in zip(test["utterance_id"], test["text"], test["dataset_source"])}
    ids = run["settings"]["sample_ids"]
    refs, sources = [by_id[i][0] for i in ids], [by_id[i][1] for i in ids]
    hyps = {c: r["hypotheses"] for c, r in run["results"].items() if r.get("hypotheses")}
    if args.reference not in hyps:
        print(f"reference {args.reference} not in this run; available: {sorted(hyps)}", file=sys.stderr)
        return 2

    parts = [f"# Accuracy of benchmark configurations: run {run['run_id']} ({run['tag']})", "",
             f"{len(ids)} test utterances (seed {run['settings']['seed']}), compared with {args.reference}. Confidence intervals resample "
             "utterances; differences are paired (the same utterances for both configurations).", ""]
    for name, sel in [("All utterances", None), *[(f"{s} only", s) for s in sorted(set(sources))]]:
        keep = [i for i, s in enumerate(sources) if sel is None or s == sel]
        table = paired_wer_table([refs[i] for i in keep], {c: [h[i] for i in keep] for c, h in hyps.items()}, args.reference)
        parts.append(render_paired(table, args.reference, f"{name} (n={len(keep)})"))
    text = "\n".join(parts)
    out = REPO_ROOT / "res/benchmarks/experiments" / f"accuracy_{run['run_id']}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text + f"saved {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
