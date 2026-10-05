"""Rule-based entity candidates (regex + vocabulary), used as hints and routing cues."""

import importlib.util
import re
from pathlib import Path

# vocab.py is pure data. Load it by path so this module doesn't pull in the
# `preprocessing` package __init__ (librosa, datasets, ...), which the lightweight
# vLLM annotation environment doesn't have.
_spec = importlib.util.spec_from_file_location(
    "_atc_vocab", Path(__file__).resolve().parents[1] / "preprocessing" / "vocab.py")
_vocab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vocab)
ATC_COMMAND_VERBS, KNOWN_CALLSIGN_WORDS = _vocab.ATC_COMMAND_VERBS, _vocab.KNOWN_CALLSIGN_WORDS

from .schema import DIGIT_WORDS, PHONETIC_WORDS

_N = "(?:" + "|".join(sorted(DIGIT_WORDS, key=len, reverse=True)) + ")"
RULE_PATTERNS = {
    "FREQUENCY": re.compile(rf"\b{_N}(?: {_N}){{1,3}} (?:decimal|point)(?: {_N}){{1,3}}\b"),
    "RUNWAY": re.compile(rf"\brunway {_N}(?: {_N})?(?: left| right| center)?\b"),
    "HEADING": re.compile(rf"\bheading (?:of )?(?:{_N}(?: {_N}){{2}}|{_N} hundred|three hundred)\b"),
    "ALTITUDE": re.compile(rf"\b(?:flight level|level)(?: {_N}){{2,3}}\b|\b(?:{_N} )+(?:thousand|hundred)(?: feet)?\b"),
    "SQUAWK": re.compile(rf"\bsquawk(?: {_N}){{4}}\b"),
}
FACILITY_WORDS = frozenset({
    "radar", "tower", "approach", "control", "ground", "information", "apron",
    "zurich", "rhein", "milan", "geneva", "praha", "bratislava", "ruzyne",
})
_CALLSIGN_TOKEN = DIGIT_WORDS | PHONETIC_WORDS


def _overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def rule_candidates(text: str) -> list[tuple[str, str]]:
    """Candidate (label, text) pairs from regex/vocabulary rules, in text order."""
    found = [(m.start(), m.end(), label)
             for label, pat in RULE_PATTERNS.items() for m in pat.finditer(text)]
    toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    i = 0
    while i < len(toks):  # callsign: a known call-word + following digit/letter run, or a long bare run
        word, s, _ = toks[i]
        j = i + 1 if word in KNOWN_CALLSIGN_WORDS else i
        while j < len(toks) and toks[j][0] in _CALLSIGN_TOKEN:
            j += 1
        run = j - i
        if j > i and ((word in KNOWN_CALLSIGN_WORDS and run >= 2) or (word not in KNOWN_CALLSIGN_WORDS and run >= 3)):
            span = (s, toks[j - 1][2], "CALLSIGN")
            if not any(_overlap(span, f) for f in found):
                found.append(span)
        i = max(j, i + 1)
    for w, s, e in toks:
        label = "COMMAND" if w in ATC_COMMAND_VERBS else "FACILITY" if w in FACILITY_WORDS else None
        if label and not any(_overlap((s, e), f) for f in found):
            found.append((s, e, label))
    return [(label, text[s:e]) for s, e, label in sorted(found)]


assert ("FREQUENCY", "one three three decimal four") in rule_candidates("contact zurich one three three decimal four")
assert ("FACILITY", "zurich") in rule_candidates("contact zurich one three three decimal four")
