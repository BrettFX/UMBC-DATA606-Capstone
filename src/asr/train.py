"""Whisper fine-tuning run and test-set evaluation.

`train` fine-tunes a Whisper checkpoint on the training split (resuming from the newest checkpoint if
one exists), selecting the best epoch by validation WER; `evaluate` transcribes a split and scores it
with the same normalization as every other ASR comparison in the project.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import jiwer
import pandas as pd
import torch
from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor)

from .data import WhisperCollator, load_split
from .text import normalize_for_scoring

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    model: str = "openai/whisper-small.en"
    dataset_dir: Path = Path("data/processed/combined_dataset")
    output_dir: Path = Path("models/whisper-small-en-atc-finetuned-full")
    epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 4  # per device; with grad_accum 8 the effective batch is 32
    grad_accum: int = 8
    warmup_steps: int = 100
    train_limit: int | None = None  # None = the whole train split
    val_samples: int = 500  # validation subset scored each epoch (full validation generation is slow)
    seed: int = 42
    workers: int = 4  # DataLoader workers: audio decoding and feature extraction run there
    max_new_tokens: int = 128  # bounds runaway repetition; the longest utterances need ~100 tokens
    resume: bool = True
    early_stopping_patience: int | None = None  # stop after this many epochs without a validation-WER gain
    max_steps: int = -1  # >0 caps optimizer steps (a quick speed/memory benchmark); -1 trains for `epochs`
    lora: bool = False  # train low-rank adapters on a frozen fp16 base instead of every weight
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = False  # trades speed for activation memory; off unless a model needs it


@contextmanager
def _trusted_local_checkpoint_load():
    """Lift transformers' torch<2.6 `torch.load` guard while resuming from our own checkpoint.

    transformers >=5 refuses `torch.load` on torch<2.6 (CVE-2025-32434, about untrusted files), and torch
    2.5's restricted loader cannot read the numpy RNG state in a Trainer checkpoint. The repo pins torch
    2.5.1, and resuming reads optimizer/scheduler/RNG state that this same process wrote into its own local
    `output_dir`, so the files are trusted: for this one call the guard is skipped and `torch.load` runs
    unrestricted. Upgrading torch to >=2.6 would make this unnecessary.
    """
    import transformers.trainer as trainer_module

    original_check, original_load = trainer_module.check_torch_load_is_safe, torch.load
    trainer_module.check_torch_load_is_safe = lambda: None
    torch.load = lambda *a, **k: original_load(*a, **{**k, "weights_only": False})
    try:
        yield
    finally:
        trainer_module.check_torch_load_is_safe, torch.load = original_check, original_load


def _metrics_fn(processor):
    def compute_metrics(pred):
        ids = pred.label_ids.copy()
        ids[ids == -100] = processor.tokenizer.pad_token_id
        hyp = [normalize_for_scoring(t) for t in processor.batch_decode(pred.predictions, skip_special_tokens=True)]
        ref = [normalize_for_scoring(t) for t in processor.batch_decode(ids, skip_special_tokens=True)]
        return {"wer": jiwer.wer(ref, hyp)}
    return compute_metrics


def train(config: TrainConfig) -> Path:
    """Fine-tune and return the directory of the saved best model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(config.model)
    # Full fine-tuning keeps fp32 master weights (the Trainer handles fp16 mixed precision); small.en's weights +
    # grads + Adam (~4 GB) fit an 8 GB card. LoRA instead freezes an fp16 base and trains small fp32 adapters, which
    # is how a larger model fits the same card.
    model = WhisperForConditionalGeneration.from_pretrained(
        config.model, dtype=torch.float16 if config.lora and device == "cuda" else torch.float32).to(device)
    if config.lora:
        from peft import LoraConfig, get_peft_model

        model.config.use_cache = False  # incompatible with training; generation re-enables it per call
        model = get_peft_model(model, LoraConfig(
            r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))  # adapters stay fp32
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("LoRA rank %d: %.1fM trainable of %.1fM parameters", config.lora_rank, trainable / 1e6,
                    sum(p.numel() for p in model.parameters()) / 1e6)
    train_ds = load_split(config.dataset_dir, "train", limit=config.train_limit, seed=config.seed)
    val_ds = load_split(config.dataset_dir, "validation", limit=config.val_samples, seed=config.seed)
    logger.info("train %d rows, validation %d rows, effective batch %d", len(train_ds), len(val_ds),
                config.batch_size * config.grad_accum)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir), learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum, warmup_steps=config.warmup_steps,
        num_train_epochs=config.epochs, max_steps=config.max_steps, fp16=device == "cuda", eval_strategy="epoch",
        save_strategy="epoch", gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=config.max_new_tokens, logging_steps=25,
        remove_unused_columns=False,  # the collator reads `audio` and `text`, which the model signature lacks
        dataloader_num_workers=config.workers, report_to=[], seed=config.seed)
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=WhisperCollator(processor, model.config.decoder_start_token_id,
                                      feature_dtype=torch.float16 if config.lora and device == "cuda" else torch.float32),
        compute_metrics=_metrics_fn(processor), processing_class=processor,
        callbacks=[EarlyStoppingCallback(config.early_stopping_patience)] if config.early_stopping_patience else None)

    resume = config.resume and any(config.output_dir.glob("checkpoint-*"))
    logger.info("starting%s", " (resuming from the newest checkpoint)" if resume else "")
    if resume:
        with _trusted_local_checkpoint_load():
            result = trainer.train(resume_from_checkpoint=True)
    else:
        result = trainer.train()

    final = config.output_dir / "final"
    if config.lora:
        trainer.save_model(str(config.output_dir / "final-adapter"))  # the adapters alone (tens of MB)
        merged = trainer.model.merge_and_unload()  # a plain fp16 Whisper that evaluate() and S3 consumers can load
        merged.config.use_cache = True
        merged.save_pretrained(str(final))
    else:
        trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    (config.output_dir / "train_config.json").write_text(json.dumps(asdict(config), default=str, indent=2))
    (config.output_dir / "train_metrics.json").write_text(json.dumps(
        {"train": result.metrics, "log_history": trainer.state.log_history,
         "best_checkpoint": trainer.state.best_model_checkpoint}, indent=2))
    return final


@torch.no_grad()
def transcribe(model, processor, ds, *, batch_size: int = 8, max_new_tokens: int = 128) -> list[str]:
    """Raw (unnormalized) hypotheses for every row of `ds`, in order."""
    model.eval()
    hyps = []
    for lo in range(0, len(ds), batch_size):
        audio = [r["array"] for r in ds[lo:lo + batch_size]["audio"]]
        feats = processor(audio, sampling_rate=16_000, return_tensors="pt").input_features
        ids = model.generate(feats.to(model.device, dtype=model.dtype), max_new_tokens=max_new_tokens)
        hyps.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return hyps


def evaluate(model_dir: Path, dataset_dir: Path, split: str, out_csv: Path, *, limit: int | None = None,
             batch_size: int = 8, max_new_tokens: int = 128, label: str | None = None) -> dict:
    """Transcribe `split` with the model in `model_dir`, save per-row results to `out_csv` and return the
    corpus-level WER/CER overall and per dataset source."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    ds = load_split(dataset_dir, split, limit=limit)
    t0 = time.perf_counter()
    hyps = transcribe(model, processor, ds, batch_size=batch_size, max_new_tokens=max_new_tokens)
    df = pd.DataFrame({"utterance_id": ds["utterance_id"], "dataset_source": ds["dataset_source"],
                       "reference": ds["text"], "hypothesis": hyps, "model_version": label or str(model_dir)})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    ref, hyp = df["reference"].map(normalize_for_scoring), df["hypothesis"].map(normalize_for_scoring)
    scores = {"n": len(df), "seconds": round(time.perf_counter() - t0, 1),
              "wer": jiwer.wer(list(ref), list(hyp)), "cer": jiwer.cer(list(ref), list(hyp)), "by_source": {}}
    for source, g in df.groupby("dataset_source"):
        r, h = list(ref[g.index]), list(hyp[g.index])
        scores["by_source"][source] = {"n": len(g), "wer": jiwer.wer(r, h), "cer": jiwer.cer(r, h)}
    return scores
