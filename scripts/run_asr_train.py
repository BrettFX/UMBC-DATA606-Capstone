#!/usr/bin/env python3
"""CLI entry point for fine-tuning Whisper on the full ATC training split and scoring it on the test split.

Runs in the project's conda environment (`data-science`). Training needs most of an 8 GB GPU, so it will
not start while another job (e.g. the NER annotation run) is using it.

Usage:
    python scripts/run_asr_train.py --dry-run
    python scripts/run_asr_train.py --train-limit 200 --epochs 1 --test-limit 50     # smoke test
    python scripts/run_asr_train.py                                                  # full run
    python scripts/run_asr_train.py --eval-only                                      # re-score a saved model
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from asr import TrainConfig, evaluate, load_split, train  # noqa: E402 -- import after sys.path setup

logger = logging.getLogger(__name__)
MIN_FREE_MIB = 5500  # whisper-small.en fp16 mixed-precision training peaks near this on the dev GPU
SECS_PER_SAMPLE_EPOCH = 0.185  # measured: 1,000 samples x 3 epochs in 556 s on the dev GPU


def free_gpu_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out.splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="Root data directory.")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Where per-row test results are written (default: <data-dir>/processed/asr_results).")
    p.add_argument("--model", default="openai/whisper-small.en",
                   help="Checkpoint to fine-tune. whisper-small.en fits 8 GB; medium.en does not (full fine-tune).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: models/<model>-atc-finetuned-full under the repo.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=4, help="Per-device batch size.")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--train-limit", type=int, default=None, help="Train on a stratified sample of this size.")
    p.add_argument("--val-samples", type=int, default=500, help="Validation rows scored each epoch.")
    p.add_argument("--test-limit", type=int, default=None, help="Score a sample of the test split (default: all).")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers for audio decoding/features.")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints and start over.")
    p.add_argument("--eval-only", action="store_true", help="Score the saved model; do not train.")
    p.add_argument("--skip-eval", action="store_true", help="Train but skip the final test-set scoring.")
    p.add_argument("--dry-run", action="store_true", help="Report sizes and a time estimate; touch no GPU.")
    p.add_argument("--force-gpu", action="store_true", help="Skip the free-GPU-memory check.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    safe = args.model.split("/")[-1].replace(".", "-")
    config = TrainConfig(
        model=args.model, dataset_dir=args.data_dir / "processed" / "combined_dataset",
        output_dir=args.output_dir or REPO_ROOT / "models" / f"{safe}-atc-finetuned-full",
        epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, grad_accum=args.grad_accum,
        warmup_steps=args.warmup_steps, train_limit=args.train_limit, val_samples=args.val_samples,
        workers=args.workers, resume=not args.no_resume)

    n_train = len(load_split(config.dataset_dir, "train", limit=config.train_limit))
    steps = -(-n_train // (config.batch_size * config.grad_accum)) * config.epochs
    logger.info("model %s | %d train rows | %d optimizer steps | ~%.1f h of training",
                config.model, n_train, steps, n_train * config.epochs * SECS_PER_SAMPLE_EPOCH / 3600)
    if args.dry_run:
        return 0

    free = free_gpu_mib()
    if not args.force_gpu and (free is None or free < MIN_FREE_MIB):
        logger.error("only %s MiB of GPU memory is free (need ~%d). Another job (the NER annotation run?) is probably "
                     "using the GPU; wait for it to finish, or pass --force-gpu.", free, MIN_FREE_MIB)
        return 2

    model_dir = config.output_dir / "final"
    if not args.eval_only:
        model_dir = train(config)
    if not args.skip_eval:
        out_csv = (args.results_dir or args.data_dir / "processed" / "asr_results") / f"finetuned_{safe}_full.csv"
        scores = evaluate(model_dir, config.dataset_dir, "test", out_csv, limit=args.test_limit,
                          label=f"{args.model}-finetuned-atc-full")
        (config.output_dir / "test_scores.json").write_text(json.dumps(scores, indent=2))
        logger.info("test WER %.3f CER %.3f %s", scores["wer"], scores["cer"],
                    {k: round(v["wer"], 3) for k, v in scores["by_source"].items()})
        logger.info("per-row results: %s", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
