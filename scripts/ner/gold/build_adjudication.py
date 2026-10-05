"""Build the human audit file from two annotators' labels (Claude's, and the 9B model's pre-labels).

    python scripts/ner/gold/build_adjudication.py [--n-agreed 20]

A disagreement is *explained* when every differing span is covered by a decided guideline (see
RULES below) in which the 9B departs from the guideline; those are resolved automatically in favor
of the guideline-following labels and never shown. The remaining disagreements become
"adjudicate" rows (options A/B randomly assigned so the reviewer can't tell which annotator is
which), and a random sample of agreed utterances become blind "audit" rows (labels hidden) that
estimate how often agreement hides an error.

Writes data/ner_gold/work/audit_queue.jsonl (load in scripts/ner/gold/gold_labeler.html) and
data/ner_gold/work/audit_resolved_auto.jsonl (the auto-resolved + agreed-unaudited labels); file names and
inputs are options, so the same script serves later rounds (see --claude/--prelabels/--queue/--no-auto-rules).
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from ner import entity_spans, with_spans  # noqa: E402

GOLD = REPO / "data" / "ner_gold"
WORK = GOLD / "work"  # intermediates live here; final gold/silver/held-out files stay in GOLD
GENERIC = {"continue", "fly", "line up", "lining up", "set course", "confirm", "continue approach", "continue climb",
           "reduce speed", "speed up", "call", "wait", "report", "maintain", "identified", "descend", "climb", "climbing",
           "descending", "disregard", "standing by", "stand by", "turn left", "turn right", "right turn", "turning right"}


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def rule(side, span, text, other):
    """The decided guideline that explains a span only one annotator has (None = unexplained).

    Decisions (project owner): bare 'direct' and 'proceed direct' are COMMAND ('proceed direct' one span);
    generic motion/readback verbs are COMMAND; a 'cleared ...' clearance is one COMMAND span of any length;
    partial callsigns count; digit-only shorthand frequencies count; 'radar contact' is not an entity.
    """
    s, e, label = span
    x = text[s:e]
    if side == "claude":
        if label == "COMMAND" and x in ("direct", "proceed direct"): return "direct"
        if label == "COMMAND" and x in GENERIC: return "generic verbs"
        if label == "COMMAND" and x.startswith("cleared"): return "cleared span"
        if label == "FREQUENCY" and "decimal" not in x and "point" not in x: return "shorthand frequency"
        if label == "CALLSIGN" and (len(x.split()) <= 1 or any(overlaps(span, o) for o in other)): return "callsign boundary"
    else:
        if label == "COMMAND" and x == "contact" and "radar contact" in text: return "radar contact"
        if label == "FACILITY" and x in ("radar", "ils approach") and ("radar contact" in text or "ils approach" in text): return "radar contact"
        if label == "COMMAND" and x == "cleared": return "cleared span"
        if label == "COMMAND" and x == "proceed": return "direct"
        if label == "CALLSIGN" and any(overlaps(span, o) for o in other): return "callsign boundary"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-agreed", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--claude", default="claude_labels.jsonl", help="second annotator's labels (file in data/ner_gold/work/)")
    ap.add_argument("--prelabels", nargs="+", default=["candidates_prelabeled.jsonl", "candidates_batch2_prelabeled.jsonl"],
                    help="model pre-label files in data/ner_gold/work/")
    ap.add_argument("--queue", default="audit_queue.jsonl", help="output queue file name")
    ap.add_argument("--resolved", default="audit_resolved_auto.jsonl", help="output file for auto-resolved/agreed labels")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing queue (it holds reviewer-facing A/B assignments)")
    ap.add_argument("--no-auto-rules", action="store_true",
                    help="show every disagreement to the reviewer instead of auto-resolving rule-explained ones (use for a held-out set)")
    args = ap.parse_args()
    read = lambda p: [json.loads(line) for line in open(WORK / p)]
    claude = {r["utterance_id"]: r for r in read(args.claude)}
    model = {r["utterance_id"]: r for f in args.prelabels for r in read(f)}
    assert set(claude) == set(model)
    if (WORK / args.queue).exists() and not args.overwrite:
        sys.exit(f"{WORK / args.queue} already exists; pass --overwrite to replace it.")
    rng = random.Random(args.seed)
    spans = lambda t, e: set(entity_spans(t, e))

    adjudicate, agreed, auto = [], [], []
    for u, c in claude.items():
        t = c["text"]
        C, N = spans(t, c["entities"]), spans(t, model[u]["prelabels"])
        if C == N:
            agreed.append(u)
        elif not args.no_auto_rules and all([rule("claude", a, t, N - C) for a in C - N] + [rule("9b", b, t, C - N) for b in N - C]):
            auto.append(u)
        else:
            adjudicate.append(u)

    base = lambda u: {k: claude[u][k] for k in ("utterance_id", "text", "dataset_split", "dataset_source", "selection_reason")}
    queue = []
    for u in adjudicate:
        a, b = ("claude", "9b") if rng.random() < 0.5 else ("9b", "claude")
        opts = {"claude": with_spans(claude[u]["text"], claude[u]["entities"]),
                "9b": with_spans(claude[u]["text"], model[u]["prelabels"])}
        queue.append({**base(u), "kind": "adjudicate", "optA": opts[a], "optB": opts[b], "srcA": a, "srcB": b})
    audit = rng.sample(agreed, min(args.n_agreed, len(agreed)))
    for u in audit:
        queue.append({**base(u), "kind": "audit", "blind": True, "reference": with_spans(claude[u]["text"], claude[u]["entities"])})
    rng.shuffle(queue)
    with open(WORK / args.queue, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in queue)

    with open(WORK / args.resolved, "w") as f:
        for u in agreed + auto:
            f.write(json.dumps({**base(u), "entities": with_spans(claude[u]["text"], claude[u]["entities"]),
                                "resolution": "agreed" if u in set(agreed) else "rule-explained (guideline-following labels)",
                                "audited": u in set(audit)}) + "\n")
    print(f"{len(claude)} utterances: {len(agreed)} agreed, {len(auto)} auto-resolved by decided rules, "
          f"{len(adjudicate)} need adjudication")
    print(f"audit queue: {len(adjudicate)} adjudicate + {len(audit)} blind audits = {len(queue)} items")


if __name__ == "__main__":
    main()
