#!/usr/bin/env python3
"""CLI entry point for the LLM NER annotation stage (full corpus, resumable).

Annotates every trainable utterance in <data-dir>/processed/utterances.parquet with a vLLM-served
Qwen3.5 model (schema-guided JSON, validate-and-retry) and appends results chunk by chunk to
<data-dir>/processed/ner_dataset/annotations/annotations_<signature>.jsonl. Interrupt it at any
time; re-running resumes from the cache. The signature changes whenever the model, prompt, flags
or NER module source change, so labels from an older setup are never silently reused.

vLLM pins its own torch, so this stage runs in a separate environment from the notebooks:
    ~/venvs/vllm/bin/python scripts/run_ner_annotate.py --dry-run
    ~/venvs/vllm/bin/python scripts/run_ner_annotate.py --limit 100      # smoke test
    ~/venvs/vllm/bin/python scripts/run_ner_annotate.py                  # full run (9B, ~7 h on the dev GPU)
    ~/venvs/vllm/bin/python scripts/run_ner_annotate.py --preset 4b      # fast, lower accuracy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402 -- import after sys.path setup
from ner import AnnotationConfig, VllmBackend, build_pool, load_cache, run_annotation  # noqa: E402
from ner.prompt import PROMPTS  # noqa: E402

logger = logging.getLogger(__name__)

# Hardware presets for the 8 GB WSL2 dev GPU (see data/ner_gold/README.md and the repo notes for why
# each flag is needed). `secs_per_utt` is the measured rate, used only for the dry-run estimate.
OFFLOAD = ("gate_up_proj,down_proj,qkv_proj,o_proj,in_proj_qkvz,in_proj_qkv,in_proj_z,out_proj,embed_tokens")
PRESETS = {
    "9b": dict(model="RedHatAI/Qwen3.5-9B-quantized.w4a16", cpu_offload_gb=8.0, offload_params=OFFLOAD,
               enforce_eager=True, max_num_seqs=32, secs_per_utt=2.44),
    "4b": dict(model="RedHatAI/Qwen3.5-4B-quantized.w4a16", cpu_offload_gb=0.0, offload_params="",
               enforce_eager=False, max_num_seqs=32, secs_per_utt=0.28),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                   help="Root data directory (default: <repo_root>/data).")
    p.add_argument("--preset", choices=sorted(PRESETS), default="9b", help="Model + GPU settings (default: 9b).")
    p.add_argument("--model", default=None, help="Override the preset's model id.")
    p.add_argument("--prompt", type=int, choices=sorted(PROMPTS), default=3,
                   help="System-prompt version (default: 3, the latest guidelines).")
    p.add_argument("--chunk-size", type=int, default=256,
                   help="Utterances per checkpoint; at most one chunk of work is lost on a crash (default: 256).")
    p.add_argument("--limit", type=int, default=None,
                   help="Annotate at most this many not-yet-cached utterances (smoke test / time-boxed run).")
    p.add_argument("--seed", type=int, default=42, help="Seed for the processing order (default: 42).")
    p.add_argument("--no-postprocess", action="store_true", help="Skip the span normalizer.")
    p.add_argument("--no-check-missed", action="store_true", help="Skip the missed-command-verb retry check.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report pool size, cache state, signature and a time estimate; load no model.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    preset = dict(PRESETS[args.preset])
    secs_per_utt = preset.pop("secs_per_utt")
    if args.model:
        preset["model"] = args.model

    config = AnnotationConfig(model=preset["model"], prompt_version=args.prompt,
                              check_missed=not args.no_check_missed, postprocess=not args.no_postprocess)
    utterances = pd.read_parquet(args.data_dir / "processed" / "utterances.parquet")
    pool = build_pool(utterances)
    out_dir = args.data_dir / "processed" / "ner_dataset" / "annotations"
    cached = load_cache(out_dir / f"annotations_{config.signature()}.jsonl")
    todo = len(pool) - len(cached.keys() & set(pool["utterance_id"]))
    if args.limit is not None:
        todo = min(todo, args.limit)

    logger.info("signature %s | model %s | prompt v%d", config.signature(), config.model, config.prompt_version)
    logger.info("pool %d utterances | cached %d | to annotate %d (~%.1f h at %.2f s/utt)",
                len(pool), len(cached), todo, todo * secs_per_utt / 3600, secs_per_utt)
    if args.dry_run:
        return 0

    backend = VllmBackend(
        preset["model"], cpu_offload_gb=preset["cpu_offload_gb"], enforce_eager=preset["enforce_eager"],
        offload_params=set(filter(None, preset["offload_params"].split(","))), max_num_seqs=preset["max_num_seqs"])
    path = run_annotation(backend, pool, out_dir, config, chunk_size=args.chunk_size, limit=args.limit, seed=args.seed)
    logger.info("annotations: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
