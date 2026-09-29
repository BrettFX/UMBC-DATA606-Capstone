#!/usr/bin/env python3
"""CLI entry point for running the ATC data ingestion pipeline without a notebook.

Running this instead of executing notebooks/comprehensive_eda.ipynb end to
end avoids that notebook's much larger memory footprint (EDA plots, audio
playback widgets, word clouds, etc. all held in the same kernel alongside
the pipeline's own data) -- useful on machines with less RAM than the full
EDA notebook needs.

Usage:
    python scripts/run_ingest.py
    python scripts/run_ingest.py --force --num-proc 4
    python scripts/run_ingest.py --purge-raw
    python scripts/run_ingest.py --data-dir /path/to/data --purge-raw
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from preprocessing import DataIngestPipeline  # noqa: E402 -- import after sys.path setup

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Root data directory (default: <repo_root>/data, regardless of "
        "the current working directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute from scratch even if cached processed output already "
        "exists under <data-dir>/processed/.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Worker processes for the metadata-derivation step "
        "(default: single-process).",
    )
    parser.add_argument(
        "--purge-raw",
        action="store_true",
        help="After a successful run, delete each source's raw Hugging Face "
        "download cache under <data-dir>/raw/ to reclaim disk space. Only "
        "removes sources this run actually used; a later run without "
        "cached output re-downloads them.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    pipeline = DataIngestPipeline(data_dir=str(args.data_dir), num_proc=args.num_proc)
    combined_dataset, utterance_df = pipeline.run(force=args.force)

    n_splits = len(combined_dataset)
    print(f"\nIngestion complete: {len(utterance_df)} utterances across {n_splits} splits.")
    if pipeline.validation_report_ is not None:
        print("\nRow counts by original split / dataset source:")
        print(pipeline.validation_report_.to_string(index=False))
    if pipeline.cleaning_log_ is not None and len(pipeline.cleaning_log_):
        print("\nData-quality cleaning log:")
        print(pipeline.cleaning_log_.to_string(index=False))

    if args.purge_raw:
        pipeline.purge_raw()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
