#!/usr/bin/env python3
"""CLI entry point for training the NER student models on curated LLM labels.

Run scripts/run_ner_annotate.py and scripts/run_ner_curate.py first. Each model is scored on the LLM-labeled
dev and test splits and on the two human-verified sets (gold = tuning set, heldout = final test), with the
same span-level metric used for the LLM annotators.

Usage:
    python scripts/run_ner_train.py --model spacy
    python scripts/run_ner_train.py --model spacy --balance        # oversample utterances with rare labels
    python scripts/run_ner_train.py --model bert                   # GPU
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="Root data directory.")
    p.add_argument("--model", choices=["spacy", "bert"], default="spacy")
    p.add_argument("--balance", action="store_true", help="Oversample utterances containing under-represented labels.")
    p.add_argument("--balance-target", type=int, default=1500, help="Labels with fewer spans than this are boosted.")
    p.add_argument("--balance-cap", type=int, default=4, help="Maximum repeat factor for any utterance.")
    p.add_argument("--epochs", type=int, default=None, help="Default: 40 (spacy, with early stopping) / 4 (bert).")
    p.add_argument("--bert-model", default="bert-base-uncased", help="Checkpoint for --model bert.")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: models/ner/<model>[-balanced].")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    curated = args.data_dir / "processed" / "ner_dataset" / "curated"
    if not (curated / "train.jsonl").exists():
        logger.error("no curated data in %s; run scripts/run_ner_curate.py first", curated)
        return 2
    name = args.model + ("-balanced" if args.balance else "")
    out_dir = args.out_dir or REPO_ROOT / "models" / "ner" / name

    if args.model == "spacy":
        from ner.train_spacy import train_spacy
        results = train_spacy(curated, out_dir, epochs=args.epochs or 40, balance=args.balance,
                              balance_target=args.balance_target, balance_cap=args.balance_cap, seed=args.seed)
    else:
        from ner.train_bert import train_bert
        results = train_bert(curated, out_dir, model_name=args.bert_model, epochs=args.epochs or 4, balance=args.balance,
                             balance_target=args.balance_target, balance_cap=args.balance_cap, seed=args.seed)

    for split, s in results["scores"].items():
        logger.info("%-9s n=%4d  F1 %.3f (P %.3f R %.3f)  macro-F1 %.3f  exact %d/%d", split, s["n"], s["f1"],
                    s["precision"], s["recall"], s["macro_f1"], s["exact"], s["n"])
    logger.info("inference: %.0f utterances/s | model saved to %s", results["utterances_per_s"], out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
