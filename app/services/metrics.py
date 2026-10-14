"""Scores shown in the UI, each with a plain-language definition."""

from __future__ import annotations

from collections import Counter

DEFINITIONS = {
    "WER": "Word error rate: (substitutions + deletions + insertions) ÷ words in the reference transcript, after "
           "lowercasing and removing punctuation. Lower is better; 0% means identical text.",
    "CER": "Character error rate: the same calculation on characters instead of words. Lower is better.",
    "Entity precision": "Of the entities the model found, the share that also appear in the ground truth "
                        "(same label and same text).",
    "Entity recall": "Of the entities in the ground truth, the share the model found (same label and same text).",
    "Entity F1": "Harmonic mean of entity precision and recall. 100% means the entities match the ground truth exactly.",
    "Speech rate": "Words in the transcript ÷ speech duration, in words per minute.",
    "SNR (est.)": "Rough signal-to-noise estimate: loudest 10% of frames versus quietest 10%, in dB. Higher is cleaner audio.",
}


def _norm(text: str) -> str:
    from src.asr.text import normalize_for_scoring

    return normalize_for_scoring(text)


def wer_cer(reference: str, hypothesis: str) -> dict[str, float]:
    import jiwer

    ref, hyp = _norm(reference), _norm(hypothesis)
    if not ref:
        return {"wer": float("nan"), "cer": float("nan")}
    return {"wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp)}


def word_diff(reference: str, hypothesis: str) -> list[tuple[str, str]]:
    """[(op, word)] with op in {'=', '-', '+'} — reference words missing from / extra in the hypothesis."""
    import difflib

    ref, hyp = _norm(reference).split(), _norm(hypothesis).split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes():
        if tag == "equal":
            out += [("=", w) for w in ref[i1:i2]]
        else:
            out += [("-", w) for w in ref[i1:i2]] + [("+", w) for w in hyp[j1:j2]]
    return out


def entity_scores(gold: list[dict], pred: list[dict]) -> dict[str, float]:
    """Multiset match on (label, normalized text); independent of character offsets, which differ between
    the ground-truth text and the ASR transcript."""
    g = Counter((e["label"], " ".join(e["text"].lower().split())) for e in gold)
    p = Counter((e["label"], " ".join(e["text"].lower().split())) for e in pred)
    tp = sum((g & p).values())
    prec = tp / sum(p.values()) if p else 0.0
    rec = tp / sum(g.values()) if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "n_pred": sum(p.values()), "n_gold": sum(g.values())}


def speech_rate_wpm(text: str, duration_s: float) -> float:
    return len(text.split()) / duration_s * 60 if duration_s > 0 else 0.0
