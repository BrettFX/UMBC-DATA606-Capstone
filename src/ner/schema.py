"""Label schema, annotation constraints and character-span alignment for ATC NER."""

import re
from typing import Literal

from pydantic import BaseModel, field_validator

NerLabel = Literal[
    "CALLSIGN", "RUNWAY", "ALTITUDE", "HEADING", "FREQUENCY",
    "SQUAWK", "COMMAND", "WAYPOINT", "FACILITY",
]

NER_LABELS = {
    "CALLSIGN": "aircraft callsign: airline/telephony word plus spoken digits/letters, or a registration spelled phonetically (e.g. 'lufthansa four one alpha', 'delta india kilo alpha hotel')",
    "RUNWAY": "runway designator including side if given (e.g. 'runway two four left')",
    "ALTITUDE": "altitude or flight level with its numbers, without the verb (e.g. 'flight level three one zero', 'four thousand feet')",
    "HEADING": "compass heading with its digits (e.g. 'heading two seven zero')",
    "FREQUENCY": "radio frequency digits only, starting at the first digit word (e.g. 'one three three decimal four'); the station name is a separate FACILITY",
    "SQUAWK": "transponder code (e.g. 'squawk four two one zero')",
    "COMMAND": "controller instruction verb (phrase) only, never its numbers or destination (e.g. 'descend', 'contact', 'turn left', 'cleared to land', 'set course')",
    "WAYPOINT": "named fix, waypoint, navaid or place used as a navigation target (e.g. 'vemut', 'karlsruhe')",
    "FACILITY": "ATC facility or station name, without any digits (e.g. 'praha', 'rhein', 'zurich radar', 'tower')",
}
assert set(NER_LABELS) == set(NerLabel.__args__)

DIGIT_WORDS = frozenset({"zero", "one", "two", "three", "tree", "four", "five", "fife", "six",
                         "seven", "eight", "nine", "niner", "triple", "double"})
NUMBER_WORDS = DIGIT_WORDS | {"hundred", "thousand"}
PHONETIC_WORDS = frozenset({
    "alpha", "alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
    "juliet", "juliett", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
    "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu",
})

GREETING_PHRASES = (
    "good afternoon", "good morning", "good evening", "good day", "good night",
    "good bye", "goodbye", "bye bye", "bye", "hello", "guten tag", "tschuss",
    "servus", "bonjour", "adios", "adieu", "ciao", "thank you", "thanks",
    "roger", "okay", "ok", "ah", "uh", "oh", "yes", "sir",
)
_GREETING_RE = re.compile(
    r"(?<!\S)(?:" + "|".join(sorted(map(re.escape, GREETING_PHRASES), key=len, reverse=True)) + r")(?!\S)"
)


def strip_greetings(text: str) -> str:
    """`text` with greeting/farewell/filler phrases removed, whitespace collapsed."""
    return " ".join(_GREETING_RE.sub(" ", text).split())


def is_trainable_utterance(text: str) -> bool:
    """False for single-word utterances and ones that are only greeting/filler."""
    return len(text.split()) >= 2 and bool(strip_greetings(text))


class EntitySpan(BaseModel):
    """One entity the LLM wants to record. `text` must be copied verbatim from the utterance."""
    text: str
    label: NerLabel

    @field_validator("text")
    @classmethod
    def _normalize_text(cls, v: str) -> str:
        v = " ".join(v.lower().split())
        if not v:
            raise ValueError("entity text must not be empty")
        return v


class AnnotatedUtterance(BaseModel):
    utterance_id: str
    text: str
    entities: list[EntitySpan]
    n_turns: int
    n_rejected: int
    ok: bool
    latency_s: float


ALTITUDE_UNITS = frozenset({"level", "feet", "foot", "thousand", "hundred", "altitude", "metres", "meters"})
PROCEDURE_WORDS = frozenset({"ils", "localizer", "approach", "runway", "glideslope", "departure", "arrival"})


def check_constraints(ent: EntitySpan, text: str | None = None) -> None:
    """Raise ValueError (with a fix-it message for the LLM) if `ent` breaks a label rule.

    `text` (the full utterance) enables context rules, e.g. a number after 'speed'."""
    toks = ent.text.split()
    has_num = any(t in NUMBER_WORDS for t in toks)
    if "knots" in toks or "knot" in toks:
        raise ValueError("speeds ('... knots') are not entities; do not label them")
    if not strip_greetings(ent.text):
        raise ValueError(f"'{ent.text}' is a greeting/farewell/filler word, not an entity; do not label it")
    if ent.label == "CALLSIGN":
        if toks[0] in {"and", "roger", "ok", "okay", "hello", "oh", "yes", "request", "radar"}:
            raise ValueError(f"callsign must not start with '{toks[0]}'; start at the airline name or first letter/digit")
        if all(t in DIGIT_WORDS for t in toks):
            raise ValueError("a callsign needs an airline name or phonetic letters, not only digits")
    elif ent.label == "RUNWAY":
        if "runway" not in toks or not has_num:
            raise ValueError("RUNWAY must contain the word 'runway' and its number")
    elif ent.label == "HEADING":
        if "heading" not in toks or not has_num:
            raise ValueError("HEADING must contain the word 'heading' and its number")
    elif ent.label == "SQUAWK":
        if "squawk" not in toks or not has_num:
            raise ValueError("SQUAWK must contain the word 'squawk' and its code")
    elif ent.label == "ALTITUDE":
        if not has_num or not ALTITUDE_UNITS & set(toks):
            raise ValueError("ALTITUDE must contain its number and 'level', 'feet', 'thousand' or 'hundred' (e.g. 'flight level three one zero')")
        if toks[0] in {"climb", "descend", "maintain", "climbing", "descending", "to", "at", "on"}:
            raise ValueError(f"ALTITUDE must not start with '{toks[0]}'; the verb is a separate COMMAND")
    elif ent.label == "FREQUENCY":
        if toks[0] not in NUMBER_WORDS:
            raise ValueError("FREQUENCY must start at the first digit word; label the station name FACILITY and drop words like 'on'")
        if "decimal" not in toks and sum(t in DIGIT_WORDS for t in toks) < 4:
            raise ValueError("FREQUENCY needs 'decimal' or at least four digits; a short number is not a frequency")
        if text and re.search(rf"(?<!\S)(?:speed|mach) {re.escape(ent.text)}(?!\S)", text):
            raise ValueError("that number follows 'speed', so it is a speed, not a frequency")
    elif ent.label == "COMMAND":
        if has_num or (len(toks) > 4 and toks[0] != "cleared"):  # a full clearance ("cleared for vor dme approach") may be longer
            raise ValueError("COMMAND is the instruction verb phrase only (max 4 words, no numbers); label numbers and places separately")
    elif ent.label in {"WAYPOINT", "FACILITY"}:
        if has_num:
            raise ValueError(f"{ent.label} must not include digits; a frequency or code is its own entity")
        if ent.label == "WAYPOINT" and PROCEDURE_WORDS & set(toks):
            raise ValueError("ILS/localizer/approach/runway words describe a procedure, not a waypoint")


def align_entities(text: str, entities: list[EntitySpan]):
    """Map entities onto `text` as (start, end, label) character spans.

    Each entity takes the first not-yet-used whole-word occurrence, left to
    right. Returns (aligned spans sorted by start, [(entity, reason), ...])."""
    aligned, rejected = [], []
    for ent in entities:
        pattern = re.compile(rf"(?<!\S){re.escape(ent.text)}(?!\S)")
        matches = list(pattern.finditer(text))
        free = [m for m in matches if not any(m.start() < e and s < m.end() for s, e, _ in aligned)]
        if not matches:
            rejected.append((ent, f"'{ent.text}' does not appear verbatim (on word boundaries) in the utterance"))
        elif not free:
            rejected.append((ent, f"'{ent.text}' only occurs inside text already labeled by another entity"))
        else:
            aligned.append((free[0].start(), free[0].end(), ent.label))
    return sorted(aligned), rejected


def entity_spans(text: str, entities: list[dict]) -> list[tuple[int, int, str]]:
    """(start, end, label) character spans for entity dicts, end exclusive.

    Uses each entity's explicit `"span": [start, end]` when every entity has one
    (checked against `text`); otherwise falls back to `align_entities`."""
    if entities and all("span" in e for e in entities):
        spans = []
        for e in entities:
            start, end = e["span"]
            if not (0 <= start < end <= len(text)) or text[start:end] != e["text"]:
                raise ValueError(f"span {e['span']} does not match {e['text']!r} in {text!r}")
            spans.append((start, end, e["label"]))
        return sorted(spans)
    return align_entities(text, [EntitySpan(**e) for e in entities])[0]


def with_spans(text: str, entities: list[dict]) -> list[dict]:
    """Entity dicts in reading order, each carrying its explicit `span` offsets."""
    return [{"text": text[s:e], "label": l, "span": [s, e]} for s, e, l in entity_spans(text, entities)]
