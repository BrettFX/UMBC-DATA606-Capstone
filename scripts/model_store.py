#!/usr/bin/env python3
"""Upload, promote, download and list trained models in S3, so a new device can be set up in one command.

Models live at s3://<bucket>/ml-tasks/<task>/<model-name>/<run-id>/ with a `latest/` copy of the run that
inference should use, e.g. ml-tasks/asr/lora-whisper-medium-en/latest/. `upload` and `promote` are dry runs
unless --yes is passed; `download` is safe by default (it never overwrites a differing local file without
--force). AWS credentials come from the standard chain (environment, ~/.aws).

Usage:
    python scripts/model_store.py list
    python scripts/model_store.py upload --task ner --model-name spacy-balanced --variants spacy-balanced --set-latest --yes
    python scripts/model_store.py promote --task asr --model-name lora-whisper-medium-en --run-id 20261006-112237 --yes
    python scripts/model_store.py download --task asr --model-name lora-whisper-medium-en     # the `latest` run
    python scripts/model_store.py download --task ner --model-name spacy-balanced --run-id 20261005-231314
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=artifacts.DEFAULT_BASE, help=f"S3 prefix holding the task folders (default: {artifacts.DEFAULT_BASE}).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, run_id_help):
        sp.add_argument("--task", choices=["asr", "ner"], required=True)
        sp.add_argument("--model-name", required=True, help="Label for the model in S3 (e.g. lora-whisper-medium-en).")
        sp.add_argument("--run-id", default=None, help=run_id_help)

    up = sub.add_parser("upload", help="Upload local model artifacts as a new run.")
    common(up, "Folder for this upload (default: a UTC timestamp).")
    up.add_argument("--variants", nargs="+", default=None, help="Only these local model directories (e.g. spacy-balanced).")
    up.add_argument("--set-latest", action="store_true", help="Also make this run the model's `latest/`.")
    up.add_argument("--overwrite", action="store_true", help="Allow writing into a run prefix that already has objects.")
    up.add_argument("--yes", action="store_true", help="Actually upload. Without it nothing is sent.")

    pr = sub.add_parser("promote", help="Make an existing run the model's `latest/` (replaces its contents).")
    common(pr, "The run to promote (required).")
    pr.add_argument("--yes", action="store_true", help="Actually copy. Without it nothing changes.")

    dl = sub.add_parser("download", help="Download a run into the repo's local layout and verify checksums.")
    common(dl, "Run to download (default: latest).")
    dl.add_argument("--dest-root", type=Path, default=REPO_ROOT, help="Repo root to restore into (default: this repo).")
    dl.add_argument("--force", action="store_true", help="Overwrite local files that differ from the download.")
    dl.add_argument("--dry-run", action="store_true", help="Show what would be downloaded; write nothing.")

    ls = sub.add_parser("list", help="List the models, their runs and what `latest` points to.")
    ls.add_argument("--task", choices=["asr", "ner"], default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    import boto3
    client = boto3.client("s3")
    bucket, base_prefix = artifacts.split_s3_uri(args.base)

    if args.command == "list":
        for task in ([args.task] if args.task else ["asr", "ner"]):
            for name in artifacts.common_prefixes(client, bucket, f"{base_prefix}/{task}/"):
                runs = artifacts.common_prefixes(client, bucket, f"{base_prefix}/{task}/{name}/")
                latest = ""
                if "latest" in runs:
                    import json
                    body = client.get_object(Bucket=bucket, Key=f"{base_prefix}/{task}/{name}/latest/manifest.json")["Body"].read()
                    latest = f" -> run {json.loads(body).get('run_id')}"
                logger.info("%s/%s: runs %s%s", task, name, sorted(r for r in runs if r != "latest"), latest and f" | latest{latest}")
        return 0

    prefix = artifacts.model_prefix(args.base, args.task, args.model_name)

    if args.command == "upload":
        items = artifacts.collect(args.task, REPO_ROOT, args.variants)
        if not items:
            logger.error("no %s artifacts found; train a model first", args.task)
            return 2
        run_id = args.run_id or artifacts.new_run_id()
        dest = f"{prefix}/{run_id}"
        for i in items:
            logger.info("%10.1f MB  %s", i.size / 1e6, i.rel_key)
        logger.info("%d files, %.1f MB  ->  %s/%s", len(items), sum(i.size for i in items) / 1e6, dest,
                    f"  (and latest/)" if args.set_latest else "")
        if not args.yes:
            logger.info("dry run: nothing uploaded. Re-run with --yes to upload.")
            return 0
        manifest = artifacts.build_manifest(args.task, run_id, items, REPO_ROOT, args.model_name)
        n = artifacts.upload(items, manifest, dest, client, overwrite=args.overwrite)
        logger.info("uploaded %d objects to %s/", n, dest)
        if args.set_latest:
            m = artifacts.copy_prefix(client, bucket, f"{base_prefix}/{args.task}/{args.model_name}/{run_id}",
                                      f"{base_prefix}/{args.task}/{args.model_name}/latest", replace=True)
            logger.info("latest/ now holds run %s (%d objects)", run_id, m)
        return 0

    if args.command == "promote":
        if not args.run_id:
            logger.error("--run-id is required for promote")
            return 2
        src_key, dst_key = (f"{base_prefix}/{args.task}/{args.model_name}/{args.run_id}",
                            f"{base_prefix}/{args.task}/{args.model_name}/latest")
        n = len(artifacts.list_keys(client, bucket, src_key + "/"))
        logger.info("promote %s/%s (%d objects) -> %s/ (replacing its contents)", prefix, args.run_id, n, f"{prefix}/latest")
        if not args.yes:
            logger.info("dry run: nothing changed. Re-run with --yes to promote.")
            return 0
        logger.info("latest/ now holds run %s (%d objects)", args.run_id, artifacts.copy_prefix(client, bucket, src_key, dst_key, replace=True))
        return 0

    plan = artifacts.download(client, f"{prefix}/{args.run_id or 'latest'}", args.task, args.dest_root, force=args.force, dry_run=args.dry_run)
    for p in plan:
        logger.info("%-9s %8.1f MB  %s", p["status"], p["size"] / 1e6, p["path"].relative_to(args.dest_root))
    new = sum(p["status"] != "unchanged" for p in plan)
    logger.info("%s %d of %d files (%.1f MB) into %s", "would download" if args.dry_run else "downloaded", new, len(plan),
                sum(p["size"] for p in plan if p["status"] != "unchanged") / 1e6, args.dest_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
