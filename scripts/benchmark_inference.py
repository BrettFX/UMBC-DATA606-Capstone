#!/usr/bin/env python3
"""Benchmark inference latency (and accuracy) of the ASR and NER models across backends, devices and threads.

Every configuration runs in its own process on the same fixed utterances, at batch size 1 (how the app uses it),
and is logged with hardware and version info so later improvements can be compared against a baseline:
    res/benchmarks/runs/<run-id>_<tag>.json   full results of one run
    res/benchmarks/history.csv                one row per configuration per run (all runs)
    res/benchmarks/latest.md                  report of the latest run, with changes against the baseline run
    res/benchmarks/BASELINE.md                the report of the run tagged "baseline"

Config names are <model>/<backend>/<device>, e.g. medium/ct2-int8/cpu-4t or medium/hf-fp16/gpu.
    backends: hf-fp32, hf-fp16, hf-int8dyn (torch dynamic int8, CPU), ct2-int8 (CTranslate2/faster-whisper);
    a "-dynwin" suffix (e.g. ct2-int8-dynwin) encodes each clip in the smallest of 10/15/20/30 s windows that fits it

Usage:
    python scripts/benchmark_inference.py --suite quick --tag smoke
    python scripts/benchmark_inference.py --suite full --tag baseline        # the reference measurement
    python scripts/benchmark_inference.py --configs medium/ct2-int8/cpu-4t --tag my-change
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MODELS = {"medium": REPO_ROOT / "models/whisper-medium-en-atc-finetuned-full-lora/final",
          "small": REPO_ROOT / "models/whisper-small-en-atc-finetuned-full/final"}
CT2_DIRS = {name: path.parent / "ct2-int8" for name, path in MODELS.items()}  # converted copy next to each model
NER_MODEL = REPO_ROOT / "models/ner/spacy-balanced/model-best"
NER_TEXTS = REPO_ROOT / "data/processed/ner_dataset/curated/heldout.jsonl"
DATASET_DIR = REPO_ROOT / "data/processed/combined_dataset"

SUITES = {
    "quick": ["medium/hf-fp16/gpu", "medium/ct2-int8/cpu-4t", "ner/spacy/cpu"],
    "windows": ["medium/hf-fp16-dynwin/gpu",
                "medium/hf-fp32-dynwin/cpu-4t", "medium/hf-int8dyn-dynwin/cpu-4t",
                "medium/ct2-int8-dynwin/cpu-2t", "medium/ct2-int8-dynwin/cpu-4t", "medium/ct2-int8-dynwin/cpu-8t",
                "small/hf-fp32-dynwin/cpu-4t", "small/ct2-int8-dynwin/cpu-2t", "small/ct2-int8-dynwin/cpu-4t", "small/ct2-int8-dynwin/cpu-8t"],
    "full": ["medium/hf-fp16/gpu",
             "medium/hf-fp32/cpu-4t", "medium/hf-fp32/cpu-8t",
             "medium/hf-int8dyn/cpu-4t", "medium/hf-int8dyn/cpu-8t",
             "medium/ct2-int8/cpu-2t", "medium/ct2-int8/cpu-4t", "medium/ct2-int8/cpu-8t",
             "small/hf-fp32/cpu-4t",
             "small/ct2-int8/cpu-2t", "small/ct2-int8/cpu-4t", "small/ct2-int8/cpu-8t",
             "medium/hf-fp16-dynwin/gpu", "medium/hf-fp32-dynwin/cpu-4t", "medium/hf-int8dyn-dynwin/cpu-4t",
             "medium/ct2-int8-dynwin/cpu-2t", "medium/ct2-int8-dynwin/cpu-4t", "medium/ct2-int8-dynwin/cpu-8t",
             "small/hf-fp32-dynwin/cpu-4t", "small/ct2-int8-dynwin/cpu-2t", "small/ct2-int8-dynwin/cpu-4t", "small/ct2-int8-dynwin/cpu-8t",
             "ner/spacy/cpu"],
}


def parse_config(key: str) -> dict:
    model, backend, device = key.split("/")
    return {"model": model, "backend": backend, "gpu": device == "gpu",
            "threads": int(device.split("-")[1].rstrip("t")) if device.startswith("cpu-") else None}


# ----------------------------------------------------------------------------------------------- worker

def worker(key: str, samples_path: Path, warmup: int) -> dict:
    """Time one configuration in this (fresh) process and return its result dict."""
    import numpy as np

    with open(samples_path, "rb") as f:
        samples = pickle.load(f)
    result: dict = {"config": key}

    if key.startswith("ner/"):
        from ner.inference import SpacyNer

        texts = [json.loads(line)["text"] for line in open(NER_TEXTS)]
        t0 = time.perf_counter()
        ner = SpacyNer(NER_MODEL)
        result["load_s"] = time.perf_counter() - t0
        for t in texts[:warmup]:
            ner.predict(t)
        lat = []
        for t in texts:
            t0 = time.perf_counter()
            ner.predict(t)
            lat.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ner.predict_many(texts)
        batch_s = time.perf_counter() - t0
        lat = np.asarray(lat)
        result.update(n=len(lat), median_s=float(np.median(lat)), p95_s=float(np.percentile(lat, 95)), mean_s=float(lat.mean()),
                      throughput_utt_per_s=len(texts) / batch_s)
    else:
        from asr.inference import DYNAMIC_WINDOWS, FULL_WINDOW_S

        cfg = parse_config(key)
        windows = DYNAMIC_WINDOWS if "dynwin" in cfg["backend"] else (FULL_WINDOW_S,)
        t0 = time.perf_counter()
        if cfg["backend"].startswith("hf-"):
            from asr.inference import HFTranscriber

            asr = HFTranscriber(MODELS[cfg["model"]], device="cuda" if cfg["gpu"] else "cpu", dtype="fp16" if "fp16" in cfg["backend"] else "fp32",
                                int8_dynamic="int8dyn" in cfg["backend"], threads=cfg["threads"], windows=windows)
        else:
            from asr.inference import CT2Transcriber

            asr = CT2Transcriber(CT2_DIRS[cfg["model"]], device="cuda" if cfg["gpu"] else "cpu", compute_type="int8", threads=cfg["threads"] or 0,
                                 windows=windows)
        result["load_s"] = time.perf_counter() - t0
        audio = samples["audio"]
        t0 = time.perf_counter()
        for a in audio[:warmup]:  # the first calls are slower (allocation, kernel selection); report them separately
            asr.transcribe(a)
        result["warmup_s"] = (time.perf_counter() - t0) / max(warmup, 1)
        lat, hyps = [], []
        for a in audio:
            t0 = time.perf_counter()
            hyps.append(asr.transcribe(a))
            lat.append(time.perf_counter() - t0)
        from benchmarking import aggregate

        result.update(aggregate(lat, samples["durations"], hyps, samples["refs"]))
        result["hypotheses"] = hyps
    result["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if "gpu" in key:
        import torch

        result["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    return result


# ------------------------------------------------------------------------------------------ orchestrator

def build_samples(n: int, seed: int, path: Path) -> dict:
    """The fixed utterances every ASR configuration is timed on (stratified by source, seeded)."""
    import numpy as np

    from asr import load_split

    ds = load_split(DATASET_DIR, "test", limit=n, seed=seed)
    audio = [np.asarray(r["audio"]["array"], dtype=np.float32) for r in ds]
    samples = {"ids": ds["utterance_id"], "refs": ds["text"], "audio": audio, "durations": [len(a) / 16_000 for a in audio]}
    path.write_bytes(pickle.dumps(samples))
    return samples


def run_config(key: str, samples_path: Path, warmup: int) -> dict:
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONWARNINGS="ignore", TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_PROGRESS_BARS="1")
    threads = None if key.startswith("ner/") else parse_config(key)["threads"]
    if threads:  # a CPU configuration: hide the GPU and cap the math libraries' threads
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    elif key.startswith("ner/"):
        env.update(CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, __file__, "--worker", key, "--samples", str(samples_path), "--warmup", str(warmup)],
                          capture_output=True, text=True, env=env, timeout=3 * 3600)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("BENCH_RESULT "):
            return json.loads(line[len("BENCH_RESULT "):])
    return {"config": key, "error": (proc.stderr or proc.stdout)[-400:].replace("\n", " | ")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=sorted(SUITES), default=None)
    ap.add_argument("--configs", nargs="+", default=None, help="Explicit config names (overrides --suite).")
    ap.add_argument("--n", type=int, default=24, help="Utterances per ASR configuration (half real, half simulated audio).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=2, help="Untimed calls before measuring.")
    ap.add_argument("--tag", default="adhoc", help='Label for this run; the run tagged "baseline" is what later runs are compared to.')
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "res" / "benchmarks")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--samples", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:  # internal: benchmark a single configuration and print its result
        print("BENCH_RESULT " + json.dumps(worker(args.worker, args.samples, args.warmup)))
        return 0

    from benchmarking import append_history, baseline_rows, render_report, save_run, system_info

    configs = args.configs or SUITES[args.suite or "quick"]
    needed = {parse_config(c)["model"] for c in configs if "/ct2-" in c}
    if needed:
        from asr.inference import convert_to_ct2

        for m in sorted(needed):
            convert_to_ct2(MODELS[m], CT2_DIRS[m], "int8")  # a no-op when already converted

    info = system_info(REPO_ROOT)
    run_id, stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"), datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory() as tmp:
        samples_path = Path(tmp) / "samples.pkl"
        samples = build_samples(args.n, args.seed, samples_path)
        print(f"run {run_id} ({args.tag}): {len(configs)} configs on {len(samples['ids'])} utterances, "
              f"{sum(samples['durations']):.0f}s of audio ({sum(samples['durations']) / len(samples['durations']):.1f}s avg)", flush=True)
        results = {}
        for i, key in enumerate(configs, 1):
            t0 = time.perf_counter()
            results[key] = run_config(key, samples_path, args.warmup)
            r = results[key]
            print(f"[{i}/{len(configs)}] {key}: " + (f"ERROR {r['error'][:150]}" if r.get("error") else
                  f"median {r['median_s']:.3f}s p95 {r['p95_s']:.3f}s" + (f" WER {100 * r['wer']:.1f}%" if "wer" in r else "")) +
                  f"  ({time.perf_counter() - t0:.0f}s)", flush=True)

    history = args.out_dir / "history.csv"
    baseline = baseline_rows(history) if args.tag != "baseline" else None
    report = render_report(run_id, args.tag, info, results, baseline)
    settings = {"n": args.n, "seed": args.seed, "warmup": args.warmup, "sample_ids": samples["ids"], "audio_seconds": sum(samples["durations"])}
    path = save_run(args.out_dir, run_id, args.tag, stamp, info, settings, results, report)
    append_history(history, run_id, stamp, args.tag, info, {k: {kk: vv for kk, vv in v.items() if kk != "hypotheses"} for k, v in results.items()})
    print("\n" + report + f"saved {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
