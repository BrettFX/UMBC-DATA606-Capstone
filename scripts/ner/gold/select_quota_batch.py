"""Add validation/test utterances until every label has at least --quota spans in the gold pool.

    python scripts/ner/gold/select_quota_batch.py [--quota 40]

Counts spans already in gold.jsonl plus the pre-labels of candidates_prelabeled.jsonl, estimates
how many spans each remaining utterance would add with the rule-based candidates, and greedily
picks utterances for the labels furthest below quota. Writes data/ner_gold/work/candidates_batch2.jsonl.
These rows are selected for their labels, so they are biased: use the random batch for any
natural-distribution metric and this batch only for per-label (balanced-slice) metrics.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from ner import NER_LABELS, is_trainable_utterance, rule_candidates  # noqa: E402

GOLD = REPO / "data" / "ner_gold"
WORK = GOLD / "work"  # intermediates live here; final gold/silver/held-out files stay in GOLD
WAYPOINT_CUE = re.compile(r"\b(?:direct(?: to)?|proceed|set course|via)\b")


def estimate(text: str) -> Counter:
    c = Counter(label for label, _ in rule_candidates(text))
    c["WAYPOINT"] += len(WAYPOINT_CUE.findall(text))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quota", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    args = ap.parse_args()

    read = lambda d, p: [json.loads(line) for line in open(d / p)]
    gold, cands = read(GOLD, "gold.jsonl"), read(WORK, "candidates_prelabeled.jsonl")
    have = Counter(e["label"] for r in gold for e in r["entities"]) + Counter(e["label"] for r in cands for e in r["prelabels"])
    taken = {r["utterance_id"] for r in gold} | {r["utterance_id"] for r in cands}

    df = pd.read_parquet(REPO / "data" / "processed" / "utterances.parquet")
    pool = df[df["dataset_split"].isin(["validation", "test"]) & ~df["utterance_id"].isin(taken)]
    pool = pool[pool["transcript_normalized"].map(is_trainable_utterance) & (pool["word_count"] <= 40)]
    pool = pool.drop_duplicates("transcript_normalized").sample(frac=1, random_state=args.seed).reset_index(drop=True)
    est = [estimate(t) for t in pool["transcript_normalized"]]

    print("before:", dict(sorted(have.items(), key=lambda kv: kv[1])))
    picked, used = [], set()
    for label in sorted(NER_LABELS, key=lambda l: have[l]):  # rarest first
        for i in sorted(range(len(pool)), key=lambda i: -est[i][label]):  # richest in this label first
            if have[label] >= args.quota:
                break
            if i in used or est[i][label] == 0:
                continue
            used.add(i)
            picked.append((i, f"quota:{label}"))
            have.update(est[i])  # an utterance also adds its other labels
    out = pool.loc[[i for i, _ in picked]].assign(selection_reason=[r for _, r in picked])
    path = WORK / "candidates_batch2.jsonl"
    if path.exists() and not args.overwrite:
        sys.exit(f"{path} already exists; it is the provenance of a labeling round. Pass --overwrite to replace it.")
    with open(path, "w") as f:
        for r in out.itertuples():
            f.write(json.dumps({"utterance_id": r.utterance_id, "text": r.transcript_normalized, "dataset_split": r.dataset_split,
                                "dataset_source": r.dataset_source, "selection_reason": r.selection_reason, "blind": False}) + "\n")
    print(f"wrote {len(out)} utterances to {path.relative_to(REPO)}")
    print("estimated after:", dict(sorted(have.items(), key=lambda kv: kv[1])))
    print(out["selection_reason"].value_counts().to_string())
    print(out.groupby(["dataset_split", "dataset_source"]).size().to_string())


if __name__ == "__main__":
    main()
