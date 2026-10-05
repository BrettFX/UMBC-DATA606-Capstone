"""Fine-tune a BERT token classifier on curated LLM labels and score it on dev and the human-verified sets."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)

from .curate import BERT_LABELS
from .evaluation import evaluate_sets, time_predict
from .train_spacy import oversample, read_rows

logger = logging.getLogger(__name__)
LABEL2ID = {label: i for i, label in enumerate(BERT_LABELS)}


class _Words(torch.utils.data.Dataset):
    """Whitespace-tokenized utterances with the label on each word's first subword (-100 elsewhere)."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int = 128):
        self.items = []
        for r in rows:
            enc = tokenizer(r["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
            labels, prev = [], None
            for w in enc.word_ids():
                labels.append(-100 if w is None or w == prev else LABEL2ID[r["ner_tags"][w]])
                prev = w
            self.items.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def tags_to_spans(text: str, tags: list[str]) -> list[tuple[int, int, str]]:
    """Character spans from per-word IOB2 tags (an I- tag with no open span of its label starts a new span)."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    spans, cur = [], None
    for (s, e), tag in zip(words, tags):
        if tag == "O":
            cur = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or cur is None or cur[2] != label:
            cur = [s, e, label]
            spans.append(cur)
        else:
            cur[1] = e
    return [tuple(s) for s in spans]


def _predict(model, tokenizer, batch_size: int = 64):
    @torch.no_grad()
    def predict(texts: list[str]):
        model.eval()
        out = []
        for lo in range(0, len(texts), batch_size):
            words = [t.split() for t in texts[lo:lo + batch_size]]
            enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=128, padding=True, return_tensors="pt")
            pred = model(**{k: v.to(model.device) for k, v in enc.items()}).logits.argmax(-1).cpu()
            for b, ws in enumerate(words):
                tags, prev = [], None
                for pos, w in enumerate(enc.word_ids(b)):
                    if w is not None and w != prev:
                        tags.append(BERT_LABELS[int(pred[b, pos])])
                    prev = w
                tags += ["O"] * (len(ws) - len(tags))  # words lost to truncation
                out.append(tags_to_spans(texts[lo + b], tags))
        return out
    return predict


def train_bert(curated_dir: Path, out_dir: Path, *, model_name: str = "bert-base-uncased", epochs: int = 4,
               learning_rate: float = 5e-5, batch_size: int = 32, balance: bool = False, balance_target: int = 1500,
               balance_cap: int = 4, seed: int = 42) -> dict:
    """Fine-tune, keep the epoch with the lowest dev loss, save it and return the evaluation results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets = {p.stem: read_rows(p) for p in curated_dir.glob("*.jsonl") if p.stem != "review_queue"}
    train_rows = oversample(sets["train"], balance_target, balance_cap) if balance else sets["train"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True) if "roberta" in model_name else AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(BERT_LABELS), id2label=dict(enumerate(BERT_LABELS)), label2id=LABEL2ID).to(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), learning_rate=learning_rate, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2, num_train_epochs=epochs, weight_decay=0.01, warmup_steps=100,
        fp16=device == "cuda", eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, report_to=[], seed=seed, logging_steps=50)
    trainer = Trainer(model=model, args=args, train_dataset=_Words(train_rows, tokenizer), eval_dataset=_Words(sets["dev"], tokenizer),
                      data_collator=DataCollatorForTokenClassification(tokenizer), processing_class=tokenizer)
    logger.info("fine-tuning %s on %d utterances (%d epochs)", model_name, len(train_rows), epochs)
    trainer.train()
    trainer.save_model(str(out_dir / "model-best"))
    tokenizer.save_pretrained(str(out_dir / "model-best"))

    predict = _predict(model, tokenizer)
    results = {"model": model_name, "balance": balance, "train_utterances": len(train_rows),
               "log_history": trainer.state.log_history,
               "scores": evaluate_sets(predict, {k: v for k, v in sets.items() if k != "train"}),
               "utterances_per_s": time_predict(predict, [r["text"] for r in sets["dev"]])}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return results
