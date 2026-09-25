"""Vocabularies backing the transcript-derived feature proxies in
`text_features.py`. Kept as pure data (no logic) so they're easy to review
or extend independently.

These are all *proxies* standing in for the deferred, ground-truth NER work
This is non-exhaustive by design.
"""

from __future__ import annotations

# Source: https://www.faa.gov/air_traffic/publications/atpubs/fs_html/chap11_section_1.html#$paragraph11-1-5
ATC_NUMERAL_WORDS = frozenset(
    {
        "zero", "one", "two", "three", "tree", "four", "five", "six",
        "seven", "eight", "nine", "niner",
    }
)

# Source: https://www.faa.gov/air_traffic/publications/atpubs/fs_html/chap11_section_1.html#$paragraph11-1-5
ICAO_PHONETIC_ALPHABET = frozenset(
    {
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliet", "juliett", "kilo", "lima", "mike",
        "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
        "uniform", "victor", "whiskey", "xray", "yankee", "zulu",
    }
)

# Non-exhaustive sample of airline/operator call-words observed in ATCOSIM and ATC-ASR-Dataset plus a few common ICAO
# telephony designators.
KNOWN_CALLSIGN_WORDS = frozenset(
    {
        "swissair", "lufthansa", "alitalia", "luxair", "csa", "shamrock",
        "speedbird", "klm", "airfrance", "britishairways", "twinstar",
        "hansa", "sabena", "belair", "crossair",
    }
)

# A minimal, fixed proxy for ATC command verbs
# NOTE: This is explicitly a placeholder for the deferred NER work and not meant to be expanded further here.
ATC_COMMAND_VERBS = frozenset(
    {
        "climb", "descend", "maintain", "contact", "cleared", "turn",
        "hold", "report", "proceed", "expect", "squawk", "standby",
        "identified", "resume", "established", "cross", "taxi", "depart",
        "land",
    }
)
