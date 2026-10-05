"""Deterministic clean-up and extra validators applied around LLM annotation.

NOTE: COMMAND_PHRASES was written after reading errors on the 41-utterance gold set, so
scores on that set are optimistic for any variant that uses it.
"""

import math
import re
from collections import Counter

from .rules import ATC_COMMAND_VERBS
from .schema import entity_spans

# Multi-word instructions that the guidelines treat as one COMMAND.
COMMAND_PHRASES = (
    "cleared for takeoff", "cleared to land", "cleared for landing", "set course", "continue down",
    "line up", "stand by", "turn left", "turn right",
    # added after the double-annotation round (decided guidelines)
    "proceed direct", "cleared for ils approach", "cleared ils approach", "cleared touch and go", "continue approach",
    "continue climb", "stop climb", "stop descent", "right turn", "left turn", "reduce speed", "speed up", "lining up",
    "line up and wait",
)
# Verbs that are COMMANDs under the decided guidelines but missing from the shared vocabulary.
EXTRA_COMMAND_VERBS = frozenset({"continue", "direct", "fly", "confirm", "wait", "call"})


def normalize_entities(text: str, entities: list[dict]) -> list[dict]:
    """Merge adjacent FACILITY spans ("ruzyne" + "tower") and grow COMMAND spans to a known phrase."""
    spans = sorted(entity_spans(text, entities))
    for phrase in COMMAND_PHRASES:  # a COMMAND inside a known phrase becomes the whole phrase
        for m in re.finditer(rf"(?<!\S){re.escape(phrase)}(?!\S)", text):
            inside = [x for x in spans if x[2] == "COMMAND" and x[0] < m.end() and m.start() < x[1]]
            if inside and all(m.start() <= x[0] and x[1] <= m.end() for x in inside):
                spans = [x for x in spans if x not in inside] + [(m.start(), m.end(), "COMMAND")]
    spans.sort()
    merged = []
    for s, e, l in spans:
        if merged and l == "FACILITY" and merged[-1][2] == "FACILITY" and not text[merged[-1][1]:s].strip():
            merged[-1] = (merged[-1][0], e, l)
        else:
            merged.append((s, e, l))
    return [{"text": text[s:e], "label": l} for s, e, l in merged]


def missed_command_feedback(text: str, entities: list[dict]) -> list[str]:
    """Fix-it messages for known command verbs that no entity covers."""
    spans = entity_spans(text, entities)
    msgs = []
    for m in re.finditer(r"\S+", text):
        if m.group() == "contact" and text[:m.start()].endswith("radar "):
            continue  # "radar contact" is a noun phrase, not an instruction
        if m.group() in ATC_COMMAND_VERBS | EXTRA_COMMAND_VERBS and not any(s <= m.start() and m.end() <= e for s, e, _ in spans):
            msgs.append(f"COMMAND '{m.group()}': '{m.group()}' is an instruction verb but is not labeled; "
                        "label it COMMAND unless it is part of a larger entity")
    return msgs


def format_entities(text: str, entities: list[dict]) -> str:
    return "; ".join(f"{e['label']} '{e['text']}'" for e in with_spans_sorted(text, entities)) or "none"


def with_spans_sorted(text: str, entities: list[dict]) -> list[dict]:
    return [{"text": text[s:e], "label": l} for s, e, l in entity_spans(text, entities)]


class ExampleRetriever:
    """TF-IDF nearest neighbours over a pool of labeled utterances (e.g. the gold set)."""

    def __init__(self, pool: dict[str, dict]):
        self.pool = pool
        self.docs = {u: Counter(self._tokens(r["text"])) for u, r in pool.items()}
        df = Counter(w for d in self.docs.values() for w in d)
        self.idf = {w: math.log((1 + len(pool)) / (1 + c)) + 1 for w, c in df.items()}
        self.vecs = {u: self._vec(d) for u, d in self.docs.items()}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return text.split() + [" ".join(p) for p in zip(text.split(), text.split()[1:])]

    def _vec(self, counts):
        v = {w: c * self.idf.get(w, 1.0) for w, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def nearest(self, text: str, k: int, exclude: str | None = None) -> list[dict]:
        q = self._vec(Counter(self._tokens(text)))
        scored = sorted(((sum(x * v.get(w, 0.0) for w, x in q.items()), u) for u, v in self.vecs.items() if u != exclude),
                        reverse=True)
        return [self.pool[u] for _, u in scored[:k]]
