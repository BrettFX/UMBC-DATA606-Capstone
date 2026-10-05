"""Train a spaCy NER model on curated LLM labels and score it on dev and the human-verified sets."""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import compounding, minibatch

from .evaluation import evaluate_sets, time_predict
from .schema import NER_LABELS

logger = logging.getLogger(__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def oversample(rows: list[dict], target: int = 1500, cap: int = 4) -> list[dict]:
    """Repeat utterances that contain under-represented labels.

    A label with fewer than `target` spans is boosted by `target / count` (at most `cap`); an utterance is
    repeated by the largest boost among its labels. Common labels are left alone, so this shifts the
    balance toward rare labels without discarding data.
    """
    counts = Counter(e["label"] for r in rows for e in r["entities"])
    boost = {label: min(cap, math.ceil(target / n)) if n < target else 1 for label, n in counts.items()}
    out = []
    for r in rows:
        out.extend([r] * max([boost[e["label"]] for e in r["entities"]], default=1))
    return out


def _examples(nlp, rows: list[dict]) -> list[Example]:
    return [Example.from_dict(nlp.make_doc(r["text"]), {"entities": [(e["span"][0], e["span"][1], e["label"]) for e in r["entities"]]})
            for r in rows]


def _predict(nlp):
    def predict(texts: list[str]):
        return [[(e.start_char, e.end_char, e.label_) for e in doc.ents] for doc in nlp.pipe(texts, batch_size=128)]
    return predict


def train_spacy(curated_dir: Path, out_dir: Path, *, epochs: int = 40, patience: int = 6, dropout: float = 0.2,
                balance: bool = False, balance_target: int = 1500, balance_cap: int = 4, seed: int = 42) -> dict:
    """Train, keep the epoch with the best dev F1, save it to `out_dir` and return the evaluation results."""
    spacy.util.fix_random_seed(seed)
    random.seed(seed)
    sets = {p.stem: read_rows(p) for p in curated_dir.glob("*.jsonl") if p.stem != "review_queue"}
    train_rows = oversample(sets["train"], balance_target, balance_cap) if balance else sets["train"]
    logger.info("train %d utterances%s | dev %d", len(train_rows), f" (oversampled from {len(sets['train'])})" if balance else "",
                len(sets["dev"]))

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for label in NER_LABELS:
        ner.add_label(label)
    train_ex = _examples(nlp, train_rows)
    dev_ex = _examples(nlp, sets["dev"])
    optimizer = nlp.initialize(lambda: train_ex)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        random.shuffle(train_ex)
        losses: dict = {}
        for batch in minibatch(train_ex, size=compounding(4.0, 32.0, 1.001)):
            nlp.update(batch, sgd=optimizer, drop=dropout, losses=losses)
        dev_f1 = float(nlp.evaluate(dev_ex)["ents_f"])  # spaCy returns numpy floats, which JSON cannot encode
        history.append({"epoch": epoch, "loss": round(float(losses["ner"]), 1), "dev_f1": round(dev_f1, 4)})
        logger.info("epoch %2d loss %8.1f dev F1 %.4f", epoch, losses["ner"], dev_f1)
        if dev_f1 > best_f1:
            best_f1, best_epoch = dev_f1, epoch
            nlp.to_disk(out_dir / "model-best")
        elif epoch - best_epoch >= patience:
            logger.info("no dev improvement for %d epochs; stopping", patience)
            break

    best = spacy.load(out_dir / "model-best")
    predict = _predict(best)
    results = {"model": "spacy-blank-en-cnn", "best_epoch": best_epoch, "balance": balance, "train_utterances": len(train_rows),
               "history": history, "scores": evaluate_sets(predict, {k: v for k, v in sets.items() if k != "train"}),
               "utterances_per_s": time_predict(predict, [r["text"] for r in sets["dev"]] * 3)}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results
