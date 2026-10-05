#!/usr/bin/env python3
"""CLI entry point for uploading trained-model artifacts to S3. A dry run unless --yes is passed.

Usage:
    python scripts/upload_artifacts.py --task ner                 # dry run: lists what would be uploaded and where
    python scripts/upload_artifacts.py --task ner --yes           # upload
    python scripts/upload_artifacts.py --task asr --folder asr --yes

Files go to s3://<bucket>/<base>/<folder>/<run-id>/..., with a fresh run id per upload so an earlier model is
never replaced. AWS credentials come from the standard chain (environment, ~/.aws).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import artifacts  # noqa: E402 -- import after sys.path setup

logger = logging.getLogger(__name__)
DEFAULT_BASE = "s3://endurasoft-dev-ml-ops/ml-tasks"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["asr", "ner"], required=True)
    p.add_argument("--base", default=DEFAULT_BASE, help=f"S3 prefix that holds the task folders (default: {DEFAULT_BASE}).")
    p.add_argument("--folder", default=None, help="Task folder under --base (default: the task name).")
    p.add_argument("--variants", nargs="+", default=None, help="Only these model directories (e.g. spacy bert-balanced).")
    p.add_argument("--run-id", default=None, help="Folder for this upload (default: a UTC timestamp).")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into a prefix that already has objects.")
    p.add_argument("--yes", action="store_true", help="Actually upload. Without it nothing is sent.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    items = artifacts.collect(args.task, REPO_ROOT, args.variants)
    if not items:
        logger.error("no %s artifacts found; train a model first", args.task)
        return 2
    run_id = args.run_id or artifacts.new_run_id()
    dest = f"{args.base.rstrip('/')}/{args.folder or args.task}/{run_id}"
    total = sum(i.size for i in items)
    for i in items:
        logger.info("%10.1f MB  %s", i.size / 1e6, i.rel_key)
    logger.info("%d files, %.1f MB  ->  %s/", len(items), total / 1e6, dest)
    if not args.yes:
        logger.info("dry run: nothing uploaded. Re-run with --yes to upload.")
        return 0

    import boto3

    manifest = artifacts.build_manifest(args.task, run_id, items, REPO_ROOT)
    n = artifacts.upload(items, manifest, dest, boto3.client("s3"), overwrite=args.overwrite)
    logger.info("uploaded %d objects to %s/", n, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
