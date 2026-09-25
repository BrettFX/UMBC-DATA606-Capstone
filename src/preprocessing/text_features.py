"""Transcript-derived text features and normalization.

`normalize_transcript` is deliberately minimal: both source corpora already
spell out digits as words and use the ITU phonetic alphabet natively (e.g.
ATCOSIM: "swissair six six zero romeo"; ATC-ASR-Dataset: "SIERRA DELTA
MIKE"), so the real cross-source inconsistency is casing (plus incidental
whitespace/punctuation), not digit/phonetic representation -- there's
nothing meaningful to collapse or rewrite beyond that.

`count_numeric_tokens`, `count_command_verbs`, and
`count_callsign_like_sequences` are lightweight regex/vocabulary proxies
standing in for the deferred, ground-truth NER work -- not real entity
extraction. Treat their outputs as rough, comparative signals (e.g. "source
X has more numeric-heavy transcripts than source Y"), not exact counts.
"""

from __future__ import annotations

import re
from typing import Dict

from .vocab import (
    ATC_COMMAND_VERBS,
    ATC_NUMERAL_WORDS,
    ICAO_PHONETIC_ALPHABET,
    KNOWN_CALLSIGN_WORDS,
)

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]")

_ANCHOR_VOCAB = ICAO_PHONETIC_ALPHABET | KNOWN_CALLSIGN_WORDS
_CALLSIGN_VOCAB = _ANCHOR_VOCAB | ATC_NUMERAL_WORDS


def normalize_transcript(text: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace.

    Non-destructive of meaning: digit words and phonetic-alphabet words are
    left exactly as published (see module docstring for why there's nothing
    else worth doing here).
    """
    text = text or ""
    text = _PUNCTUATION_RE.sub(" ", text.lower())
    return _WHITESPACE_RE.sub(" ", text).strip()


def count_numeric_tokens(normalized_text: str) -> int:
    """Count of tokens naming a digit/numeral concept (`vocab.ATC_NUMERAL_WORDS`)."""
    return sum(1 for token in normalized_text.split() if token in ATC_NUMERAL_WORDS)


def count_command_verbs(normalized_text: str) -> int:
    """Count of tokens matching the fixed ATC command-verb proxy vocabulary."""
    return sum(1 for token in normalized_text.split() if token in ATC_COMMAND_VERBS)


def count_callsign_like_sequences(normalized_text: str) -> int:
    """Count runs of 2+ consecutive tokens drawn from {phonetic alphabet,
    known airline words, numeral words}, requiring at least one non-numeral
    *anchor* token (phonetic-alphabet or airline word) per run.

    Without the anchor requirement, a pure numeric readback (e.g. a
    frequency "one three four five two") would be 5 consecutive
    numeral-vocabulary tokens with no callsign actually present -- the
    anchor is what signals "this looks like a callsign" rather than "this
    is a string of digits."
    """
    tokens = normalized_text.split()
    count = 0
    run_length = 0
    run_has_anchor = False
    for token in tokens + [None]:  # sentinel flushes the final run
        if token in _CALLSIGN_VOCAB:
            run_length += 1
            run_has_anchor = run_has_anchor or token in _ANCHOR_VOCAB
        else:
            if run_length >= 2 and run_has_anchor:
                count += 1
            run_length = 0
            run_has_anchor = False
    return count


def compute_transcript_features(text: str) -> Dict[str, object]:
    """Derive every transcript-level feature from one row's raw `text`.

    Returns `transcript_normalized` alongside the numeric proxies so
    downstream duplicate detection (`quality.py`) can key off the same
    normalized value computed here rather than recomputing it.
    """
    normalized = normalize_transcript(text)
    return {
        "transcript_normalized": normalized,
        "numeric_token_count": count_numeric_tokens(normalized),
        "command_verb_count": count_command_verbs(normalized),
        "callsign_like_count": count_callsign_like_sequences(normalized),
    }
