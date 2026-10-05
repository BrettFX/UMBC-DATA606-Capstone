#!/usr/bin/env python3
"""CLI entry point for the NER curation stage: LLM annotations -> training data and a review queue.

Reads the annotation cache written by scripts/run_ner_annotate.py, routes doubtful labels to a review
queue, removes any text shared with the human-verified evaluation sets, and exports spaCy DocBins and
BERT IOB2 JSONL under <data-dir>/processed/ner_dataset/curated/. No GPU needed.

Usage:
    python scripts/run_ner_curate.py
    python scripts/run_ner_curate.py --annotations path/to/annotations_<signature>.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ner import curate, load_cache, write_exports  # noqa: E402 -- import after sys.path setup

logger = logging.getLogger(__name__)
HUMAN_SETS = {"gold": "gold_expanded.jsonl", "heldout": "heldout_gold.jsonl"}  # export name -> file in data/ner_gold/


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="Root data directory.")
    p.add_argument("--annotations", type=Path, default=None,
                   help="Annotation cache to curate (default: the newest annotations_*.jsonl).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    ann_dir = args.data_dir / "processed" / "ner_dataset" / "annotations"
    path = args.annotations or max(ann_dir.glob("annotations_*.jsonl"), key=lambda p: p.stat().st_mtime)
    annotations = list(load_cache(path).values())
    human = {name: [json.loads(line) for line in open(args.data_dir / "ner_gold" / file)]
             for name, file in HUMAN_SETS.items()}
    out_dir = args.data_dir / "processed" / "ner_dataset" / "curated"
    logger.info("curating %d annotations from %s", len(annotations), path.name)
    manifest = write_exports(curate(annotations, human), out_dir)
    manifest["source_annotations"] = path.name
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("utterances per split: %s", manifest["utterances"])
    logger.info("dropped: %s | review queue: %d", manifest["dropped"], manifest["review_queue"])
    logger.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
