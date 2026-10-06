"""Latency-benchmark bookkeeping: aggregate per-utterance timings, log runs, and compare against a baseline.

A benchmark run times every configuration (model x backend x device x threads) on the same utterances, in its own
process, and records latency, real-time factor, memory and WER together, so a change that is faster but less
accurate shows up immediately. Each run is saved in full under res/benchmarks/runs/, one row per configuration is
appended to res/benchmarks/history.csv, and a markdown report compares the run with the run tagged "baseline".
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

HISTORY_FIELDS = ["run_id", "timestamp_utc", "tag", "git_commit", "git_dirty", "config", "n", "median_s", "p95_s", "mean_s",
                  "rtf", "wer", "cer", "load_s", "warmup_s", "peak_rss_mb", "gpu_peak_mb", "error"]


def aggregate(latencies: list[float], durations: list[float], hyps: list[str], refs: list[str]) -> dict:
    """Summary statistics for one configuration. `rtf` is total processing time over total audio time
    (below 1.0 means faster than real time); WER/CER are corpus-level on normalized text."""
    import jiwer

    from asr.text import normalize_for_scoring

    lat = np.asarray(latencies, dtype=float)
    r, h = [normalize_for_scoring(x) for x in refs], [normalize_for_scoring(x) for x in hyps]
    return {"n": len(lat), "median_s": float(np.median(lat)), "p95_s": float(np.percentile(lat, 95)), "mean_s": float(lat.mean()),
            "max_s": float(lat.max()), "rtf": float(lat.sum() / sum(durations)), "audio_s": float(sum(durations)),
            "wer": float(jiwer.wer(r, h)), "cer": float(jiwer.cer(r, h))}


def system_info(repo_root: Path) -> dict:
    """Hardware, OS and library versions, so a result can be interpreted (or reproduced) later."""
    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor())
    mem_kb = next((int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal")), 0)
    versions = {}
    for mod in ("torch", "transformers", "ctranslate2", "faster_whisper", "spacy", "numpy"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = None
    return {"cpu": cpu, "logical_cores": __import__("os").cpu_count(), "ram_gb": round(mem_kb / 1e6, 1),
            "gpu": run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader") or None,
            "os": platform.platform(), "wsl2": "microsoft" in Path("/proc/version").read_text().lower(), "python": platform.python_version(),
            "versions": versions, "git_commit": run("git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"),
            "git_dirty": bool(run("git", "-C", str(repo_root), "status", "--porcelain"))}


def append_history(path: Path, run_id: str, timestamp: str, tag: str, info: dict, results: dict[str, dict]) -> None:
    """Add one row per configuration to the cumulative history CSV."""
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new:
            w.writeheader()
        for config, r in results.items():
            w.writerow({"run_id": run_id, "timestamp_utc": timestamp, "tag": tag, "git_commit": info["git_commit"],
                        "git_dirty": info["git_dirty"], "config": config, **{k: r.get(k) for k in HISTORY_FIELDS if k in r}})


def baseline_rows(path: Path, tag: str = "baseline") -> dict[str, dict]:
    """The rows of the most recent run tagged `tag`, keyed by configuration (empty if there is none)."""
    if not path.exists():
        return {}
    rows = [r for r in csv.DictReader(open(path)) if r["tag"] == tag]
    if not rows:
        return {}
    last = rows[-1]["run_id"]
    return {r["config"]: r for r in rows if r["run_id"] == last}


def _f(x, spec: str, none: str = "-") -> str:
    return none if x in (None, "") else format(float(x), spec)


def _secs(x) -> str:
    """Seconds with enough precision for sub-10 ms values (NER) and two decimals otherwise."""
    return format(float(x), ".3f" if float(x) < 0.1 else ".2f")


def render_report(run_id: str, tag: str, info: dict, results: dict[str, dict], baseline: dict[str, dict] | None = None,
                  reference: str = "medium/hf-fp32/cpu-4t") -> str:
    """Markdown table of one run, with speedup against `reference` and changes against the baseline run."""
    ref = results.get(reference, {}).get("median_s")
    lines = [f"# Inference benchmark: run {run_id} ({tag})", "",
             f"{info['cpu']} ({info['logical_cores']} logical cores, {info['ram_gb']} GB RAM) | GPU: {info['gpu'] or 'none'} | "
             f"{'WSL2, ' if info['wsl2'] else ''}python {info['python']} | torch {info['versions']['torch']}, "
             f"ctranslate2 {info['versions']['ctranslate2']} | commit {info['git_commit']}{' (dirty)' if info['git_dirty'] else ''}", "",
             "Latency is wall time per utterance at batch size 1 (feature extraction + decoding), excluding model load and "
             "two warm-up calls. RTF = processing time / audio time (< 1 is faster than real time). The hybrid CPU "
             "schedules threads across performance and efficiency cores, so thread-count rows are approximate.", "",
             "| config | median s | p95 s | RTF | speedup | WER | load s | warm-up s | peak RAM MB | GPU MB |"
             + (" vs baseline (median, WER) |" if baseline else ""), "|" + "---|" * (10 + bool(baseline))]
    for cfg, r in results.items():
        if r.get("error"):
            lines.append(f"| {cfg} | ERROR: {str(r['error'])[:80]} |" + " |" * (8 + bool(baseline)))
            continue
        speed = f"{ref / r['median_s']:.1f}x" if ref and r.get("median_s") and cfg != reference and cfg.startswith(reference.split('/')[0]) else "-"
        wer = "-" if r.get("wer") is None else f"{100 * r['wer']:.1f}%"
        row = (f"| {cfg} | {_secs(r['median_s'])} | {_secs(r['p95_s'])} | {_f(r.get('rtf'), '.2f')} | {speed} | "
               f"{wer} | {_f(r.get('load_s'), '.1f')} | {_f(r.get('warmup_s'), '.1f')} | "
               f"{_f(r.get('peak_rss_mb'), '.0f')} | {_f(r.get('gpu_peak_mb'), '.0f')} |")
        if baseline:
            twin = cfg if cfg in baseline else cfg.replace("-dynwin", "")  # a dynamic-window config is compared with its full-window twin
            b = baseline.get(twin)
            row += (f" {100 * (r['median_s'] / float(b['median_s']) - 1):+.0f}% latency, {100 * (r['wer'] - float(b['wer'])):+.1f} pts WER"
                    f"{'' if twin == cfg else ' (vs ' + twin + ')'} |"
                    if b and b.get("median_s") and r.get("wer") is not None and b.get("wer") not in (None, "") else " new |")
        lines.append(row)
    return "\n".join(lines) + "\n"


def save_run(out_dir: Path, run_id: str, tag: str, timestamp: str, info: dict, settings: dict, results: dict[str, dict], report: str) -> Path:
    runs = out_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{run_id}_{tag}.json"
    path.write_text(json.dumps({"run_id": run_id, "timestamp_utc": timestamp, "tag": tag, "system": info, "settings": settings,
                                "results": results}, indent=2))
    (out_dir / "latest.md").write_text(report)
    if tag == "baseline":
        (out_dir / "BASELINE.md").write_text(report)
    return path


def paired_wer_table(refs: list[str], hyps: dict[str, list[str]], reference: str, n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """Per-configuration WER with a bootstrap CI, plus the paired difference against `reference` on the same utterances.

    Resampling utterances (the same ones for both models) tests whether a configuration really differs from the
    reference, which a pair of single WER numbers on a few hundred utterances cannot. Also counts utterances whose
    transcript is identical to the reference's and utterances with WER above 50% (gross failures such as hallucination).
    """
    import jiwer

    from asr.text import normalize_for_scoring

    ref = [normalize_for_scoring(r) for r in refs]
    words = np.array([len(r.split()) for r in ref])
    errors, texts = {}, {}
    for cfg, hs in hyps.items():
        texts[cfg] = [normalize_for_scoring(h) for h in hs]
        errors[cfg] = np.array([len(h.split()) if not r else (lambda o: o.substitutions + o.deletions + o.insertions)(jiwer.process_words(r, h))
                                for r, h in zip(ref, texts[cfg])])
    idx = np.random.default_rng(seed).integers(0, len(ref), size=(n_boot, len(ref)))
    w_boot = words[idx].sum(1)
    pct = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    rows = []
    for cfg, e in errors.items():
        lo, hi = pct(e[idx].sum(1) / w_boot)
        row = {"config": cfg, "n": len(ref), "wer": float(e.sum() / words.sum()), "wer_lo": lo, "wer_hi": hi,
               "bad_clips": int(((e / np.maximum(words, 1)) > 0.5).sum()), "identical_to_ref": float(np.mean([a == b for a, b in zip(texts[cfg], texts[reference])]))}
        if cfg != reference:
            dlo, dhi = pct((e[idx].sum(1) - errors[reference][idx].sum(1)) / w_boot)
            row.update(diff=float((e.sum() - errors[reference].sum()) / words.sum()), diff_lo=dlo, diff_hi=dhi,
                       verdict="better" if dhi < 0 else "worse" if dlo > 0 else "no clear difference")
        rows.append(row)
    return rows


def render_paired(rows: list[dict], reference: str, title: str) -> str:
    lines = [f"### {title}", "", f"| config | WER (95% CI) | vs {reference} (95% CI) | identical transcripts | clips with WER>50% |", "|---|---|---|---|---|"]
    for r in rows:
        vs = "reference" if r["config"] == reference else f"{100 * r['diff']:+.2f} pts ({100 * r['diff_lo']:+.2f}, {100 * r['diff_hi']:+.2f}): {r['verdict']}"
        lines.append(f"| {r['config']} | {100 * r['wer']:.2f}% ({100 * r['wer_lo']:.2f}-{100 * r['wer_hi']:.2f}) | {vs} | {100 * r['identical_to_ref']:.0f}% | {r['bad_clips']} |")
    return "\n".join(lines) + "\n"
