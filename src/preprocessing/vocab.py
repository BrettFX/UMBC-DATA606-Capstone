"""Vocabularies backing the transcript-derived feature proxies in
`text_features.py`. Kept as pure data (no logic) so they're easy to review
or extend independently.

These are all *proxies* standing in for the deferred, ground-truth NER work
(see the project checklist) -- non-exhaustive by design, not a validated
aviation-phraseology reference. Treat anything derived from them as a rough,
comparative signal, not an exact count.
"""

from __future__ import annotations

# ATC uses "niner" (not "nine") to avoid confusion with German "nein" over
# noisy radio, and ICAO-standard "tree" for "three"; both spellings and the
# standard forms are included since transcripts may use either.
ATC_NUMERAL_WORDS = frozenset(
    {
        "zero", "one", "two", "three", "tree", "four", "five", "six",
        "seven", "eight", "nine", "niner",
        "hundred", "thousand", "decimal", "point",
    }
)

# ITU/ICAO phonetic alphabet -- used natively in both source transcripts for
# callsign letters (e.g. "SIERRA DELTA MIKE").
ITU_PHONETIC_ALPHABET = frozenset(
    {
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliet", "juliett", "kilo", "lima", "mike",
        "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
        "uniform", "victor", "whiskey", "xray", "yankee", "zulu",
    }
)

# Non-exhaustive sample of airline/operator call-words observed in this
# project's corpora (ATCOSIM, ATC-ASR-Dataset) plus a few common ICAO
# telephony designators -- a proxy anchor for callsign-like detection, not a
# validated/complete aviation callsign registry.
KNOWN_CALLSIGN_WORDS = frozenset(
    {
        "swissair", "lufthansa", "alitalia", "luxair", "csa", "shamrock",
        "speedbird", "klm", "airfrance", "britishairways", "twinstar",
        "hansa", "sabena", "belair", "crossair",
    }
)

# A minimal, fixed proxy for ATC command verbs -- explicitly a placeholder
# for the deferred, real NER work, not meant to be expanded further here.
ATC_COMMAND_VERBS = frozenset(
    {
        "climb", "descend", "maintain", "contact", "cleared", "turn",
        "hold", "report", "proceed", "expect", "squawk", "standby",
        "identified", "resume", "established", "cross", "taxi", "depart",
        "land",
    }
)
