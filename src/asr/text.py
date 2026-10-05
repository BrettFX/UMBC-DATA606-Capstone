"""Scoring-time text normalization, moved from the machine learning notebook so every ASR comparison
(baseline, off-the-shelf fine-tune, our fine-tune) is scored identically."""

import re

from preprocessing.text_features import normalize_transcript

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
_NUMERIC_RUN_RE = re.compile(r"\d+(?:[.\-]\d+)*")


def expand_digits_to_atc_words(text: str) -> str:
    """Expand every digit run into individual spoken digit words, matching ATC phraseology (digit by digit,
    never grouped: '660' -> 'six six zero'). '.' -> 'decimal', '-' -> 'dash'. A no-op on text with no digits."""
    def _expand(match: re.Match) -> str:
        words = []
        for ch in match.group(0):
            words.append("decimal" if ch == "." else "dash" if ch == "-" else _DIGIT_WORDS[ch])
        return " ".join(words)

    return _NUMERIC_RUN_RE.sub(_expand, text)


def normalize_for_scoring(text: str) -> str:
    """Digit-word expansion, then casing/punctuation/whitespace normalization. Never mutates the stored raw
    hypothesis: it is applied at scoring time only."""
    return normalize_transcript(expand_digits_to_atc_words(text or ""))
