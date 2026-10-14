"""Facts for the model cards: read from each model's results/score files, plus the static description of how it was built."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app import config

NER_LABELS_ORDER = ["CALLSIGN", "COMMAND", "FACILITY", "ALTITUDE", "FREQUENCY", "WAYPOINT", "RUNWAY", "HEADING", "SQUAWK"]

NER_EVAL_SETS = {
    "dev": ("Dev (LLM-labeled)", "Validation utterances labeled by an LLM annotator and curated; used to pick the best training epoch."),
    "test_llm": ("Test (LLM-labeled)", "Large held-out test split with the same LLM-assisted labels as training. Largest set, but labels are not human-verified."),
    "gold": ("Gold (hand-labeled)", "95 utterances labeled and checked by hand. Small, so scores move a lot with a few errors."),
    "heldout": ("Held-out (adjudicated)", "100 utterances, randomly drawn and labeled by humans with disagreements adjudicated. Never used in training or tuning."),
}


@dataclass
class ModelCard:
    key: str
    title: str
    subtitle: str
    task: str
    library: str
    base: str
    language: str = "English (ATC phraseology)"
    tags: list[str] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)         # shown in the header grid
    summary: str = ""
    intended_use: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    training: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, dict] = field(default_factory=dict)      # eval set -> {n, precision, recall, f1, per_label_f1, ...}
    metric_note: str = ""


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _dir_mb(path: Path) -> str:
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return "n/a"
    return f"{total / 1e9:.1f} GB" if total > 1e9 else f"{total / 1e6:.1f} MB"


def ner_card(name: str) -> ModelCard:
    path = config.NER_DIRS[name]
    res = _json(path.parent / "results.json")
    balanced = bool(res.get("balance"))
    n_train = res.get("train_utterances")
    base_train = 7942
    card = ModelCard(
        key=name, title=name, subtitle="Aviation entity extraction (NER) for ATC transcripts",
        task="Token classification (NER)", library="spaCy 3.8", base="blank English pipeline, CNN transition-based parser",
        tags=["ner", "spacy", "atc", "aviation", "cpu-friendly", "balanced" if balanced else "standard"],
        facts={"Parameters": "CNN (tok2vec + NER)", "Size": _dir_mb(path), "Entity labels": "9",
               "Speed (CPU)": f"{res['utterances_per_s']:,.0f} utt/s" if res.get("utterances_per_s") else "n/a",
               "Best epoch": str(res.get("best_epoch", "n/a")), "Training utterances": f"{n_train:,}" if n_train else "n/a"},
        summary=("Tags nine kinds of items in a transcribed air traffic control transmission: callsigns, instructions, facilities, "
                 "altitudes/flight levels, frequencies, waypoints, runways, headings and squawk codes. "
                 + ("This variant repeats utterances containing under-represented labels during training so that rare labels "
                    "(headings, squawks, runways, waypoints) get more learning signal without discarding any data."
                    if balanced else
                    "This variant is trained on the curated data exactly as it is, so the label mix matches what controllers say most often.")),
        intended_use=["Highlighting and structuring entities in automatic transcripts inside ClearanceIQ.",
                      "Research on ATC language understanding.",
                      "Input to the plain-English explanation step (the entity tags are given to the LLM as context)."],
        limitations=["Works on text only: transcription errors from the speech model pass straight through (e.g. a garbled callsign).",
                     "Labels are English and follow the ATCOSIM and ATC-ASR phraseology; other languages and regional variants are not covered.",
                     "Training labels came from an LLM annotator and were curated, so they contain some noise; the hand-labeled sets are small.",
                     "Rare labels (heading, squawk) have few examples in every evaluation set; their scores are noisy.",
                     "Not validated for operational use."],
        training={"Method": "spaCy NER, Adam, dropout 0.2, early stopping on dev F1 (patience 6, max 40 epochs)",
                  "Data": "ATCOSIM (simulated) and ATC-ASR (real radio) transcripts with curated LLM span labels",
                  "Class balance": (f"Oversampled: utterances with labels under 1,500 spans repeated up to 4× ({base_train:,} → {n_train:,})"
                                    if balanced and n_train else "None: original label distribution"),
                  "Seed": "42"},
        metrics=res.get("scores", {}),
        metric_note="Span-level exact match: a prediction counts only if start, end and label all equal the reference.")
    return card


def asr_card() -> ModelCard:
    from src.asr.inference import DEFAULT_CT2_DIR, DEFAULT_MODEL_DIR

    root = DEFAULT_MODEL_DIR.parent
    scores, cfg = _json(root / "test_scores.json"), _json(root / "train_config.json")
    return ModelCard(
        key="asr", title=config.ASR_MODEL_NAME, subtitle="Speech recognition for ATC radio",
        task="Automatic speech recognition", library="CTranslate2 (faster-whisper), int8", base="openai/whisper-medium.en",
        tags=["asr", "whisper", "lora", "atc", "aviation", "ctranslate2", "int8"],
        facts={"Parameters": "≈769 M (medium.en)", "Inference size": _dir_mb(DEFAULT_CT2_DIR), "Precision": "int8 (CPU)",
               "Test WER": f"{scores['wer']:.1%}" if scores.get("wer") is not None else "n/a",
               "Test CER": f"{scores['cer']:.1%}" if scores.get("cer") is not None else "n/a",
               "Speed (CPU)": "≈ 0.7 s / utterance, 4 threads"},
        summary=("Whisper medium (English) fine-tuned with LoRA on air traffic control speech, then converted to CTranslate2 int8 so it "
                 "runs on an ordinary CPU. Digits and callsigns are transcribed as spoken words (\"flight level three one zero\")."),
        intended_use=["Transcribing short ATC radio transmissions (a few seconds each) for ClearanceIQ.",
                      "Research on domain-adapted speech recognition."],
        limitations=["Trained on ATCOSIM (simulated, clean) and ATC-ASR (real radio); accuracy on other accents, radios or noise levels is unknown.",
                     "Clearly worse on real radio than on simulated speech (see per-source scores).",
                     "Window is 30 s: longer recordings are split at pauses by the app, which can cut across a transmission.",
                     "May drop or repeat words in noisy audio; always check against the audio.",
                     "Not validated for operational use."],
        training={"Base model": cfg.get("model", "openai/whisper-medium.en"),
                  "Method": f"LoRA rank {cfg.get('lora_rank')}, alpha {cfg.get('lora_alpha')}, dropout {cfg.get('lora_dropout')}, merged for export",
                  "Schedule": f"{cfg.get('epochs')} epochs, lr {cfg.get('learning_rate')}, batch {cfg.get('batch_size')} × {cfg.get('grad_accum')} accumulation",
                  "Data": "ATCOSIM + ATC-ASR train splits (≈ 14 k utterances)", "Seed": str(cfg.get("seed", 42))},
        metrics={"test": {"n": scores.get("n"), "wer": scores.get("wer"), "cer": scores.get("cer"), "by_source": scores.get("by_source", {})}},
        metric_note="WER/CER on the combined held-out test split, after lowercasing and removing punctuation.")


def all_cards() -> dict[str, ModelCard]:
    cards = {"Speech recognition": asr_card()}
    for name in config.NER_DIRS:
        if (config.NER_DIRS[name] / "meta.json").exists():
            cards[name] = ner_card(name)
    return cards
