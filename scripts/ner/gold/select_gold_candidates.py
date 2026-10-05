"""Pick new utterances to hand-label for the NER gold set.

    python scripts/ner/gold/select_gold_candidates.py [--n-random 100] [--n-per-cue 12]

Writes data/ner_gold/work/candidates.jsonl. Gold is drawn only from the validation and test splits,
so training labels stay separate from the labels used to evaluate. Each row carries:
  selection_reason  "random" rows are an unbiased sample (use only these for an unbiased accuracy
                    estimate); "targeted:<LABEL>" rows oversample rare labels and bias any metric.
  blind             True for a random ~25% of rows: the labeling tool hides the model's pre-labels
                    for these, so the agreement between blind and pre-labeled rows measures how much
                    seeing the pre-labels anchors the reviewer.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from ner import is_trainable_utterance  # noqa: E402

CUES = {"SQUAWK": r"\bsquawk\b", "WAYPOINT": r"\b(?:direct(?: to)?|proceed|set course|via)\b",
        "RUNWAY": r"\brunway\b", "HEADING": r"\bheading\b"}
GOLD = REPO / "data" / "ner_gold"
WORK = GOLD / "work"  # intermediates live here; final gold/silver/held-out files stay in GOLD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--n-per-cue", type=int, default=12)
    ap.add_argument("--blind-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="candidates.jsonl", help="output file name inside data/ner_gold/work/")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    ap.add_argument("--exclude-all-labeled", action="store_true",
                    help="also exclude every utterance_id found in any other data/ner_gold/*.jsonl and work/*.jsonl (use for a held-out set)")
    args = ap.parse_args()

    df = pd.read_parquet(REPO / "data" / "processed" / "utterances.parquet")
    files = ([p for d in (GOLD, WORK) for p in d.glob("*.jsonl") if p.name != args.out] if args.exclude_all_labeled
             else [GOLD / "gold.jsonl"])
    rows = [json.loads(line) for p in files for line in open(p)]
    have = {r["utterance_id"] for r in rows}
    have_texts = {r["text"] for r in rows if "text" in r}  # identical wording under another id is also a leak
    pool = df[df["dataset_split"].isin(["validation", "test"]) & ~df["utterance_id"].isin(have)
              & ~df["transcript_normalized"].isin(have_texts)]
    pool = pool[pool["transcript_normalized"].map(is_trainable_utterance) & (pool["word_count"] <= 40)]
    pool = pool.drop_duplicates("transcript_normalized").reset_index(drop=True)
    print(f"{len(pool)} eligible utterances (validation+test, deduplicated, not already gold)")

    parts = []
    for label, pat in CUES.items():  # targeted first so the random draw doesn't duplicate them
        cand = pool[pool["transcript_normalized"].str.contains(pat)]
        parts.append(cand.sample(n=min(args.n_per_cue, len(cand)), random_state=args.seed).assign(selection_reason=f"targeted:{label}"))
    rest = pool[~pool["utterance_id"].isin(pd.concat(parts)["utterance_id"])]
    per_source = args.n_random // 2  # equal real/simulated audio: the real source is the harder, more relevant one
    for _, g in rest.groupby("dataset_source"):
        parts.append(g.sample(n=min(per_source, len(g)), random_state=args.seed).assign(selection_reason="random"))
    out = pd.concat(parts).drop_duplicates("utterance_id").sample(frac=1, random_state=args.seed).reset_index(drop=True)

    rng = random.Random(args.seed)
    out["blind"] = [rng.random() < args.blind_frac for _ in range(len(out))]
    WORK.mkdir(exist_ok=True)
    path = WORK / args.out
    if path.exists() and not args.overwrite:
        sys.exit(f"{path} already exists; it is the provenance of a labeling round. Pass --overwrite to replace it.")
    with open(path, "w") as f:
        for r in out.itertuples():
            f.write(json.dumps({"utterance_id": r.utterance_id, "text": r.transcript_normalized, "dataset_split": r.dataset_split,
                                "dataset_source": r.dataset_source, "selection_reason": r.selection_reason, "blind": bool(r.blind)}) + "\n")
    print(f"wrote {len(out)} candidates to {path.relative_to(REPO)}")
    print(out.groupby(["selection_reason"]).size().to_string())
    print(out.groupby(["dataset_split", "dataset_source"]).size().to_string())
    print(f"blind: {int(out['blind'].sum())}")


if __name__ == "__main__":
    main()
