"""Entity visualizations: displaCy highlighted transcript, legend, and the structured clearance card."""

from __future__ import annotations

import html

import streamlit as st
import streamlit.components.v1 as components
from spacy import displacy

from app import config

# order and grouping of the clearance card
CARD_ROWS = ["COMMAND", "ALTITUDE", "HEADING", "RUNWAY", "FREQUENCY", "SQUAWK", "WAYPOINT", "FACILITY"]


def _clean(entities: list[dict], text: str) -> list[dict]:
    """Sorted, non-overlapping entities with valid spans."""
    out, last = [], 0
    for e in sorted(entities, key=lambda e: (e["span"][0], -e["span"][1])):
        s, t = e["span"]
        if 0 <= s < t <= len(text) and s >= last:
            out.append(e)
            last = t
    return out


_DISPLACY_CSS = ("<style>body{margin:0;font-family:'Source Sans Pro',sans-serif;color:#e6f1fb}"
                 ".entities{line-height:2.6!important;font-size:1.05rem}mark{color:%s!important;font-weight:600}"
                 ".seg{font-size:.72rem;letter-spacing:.08em;opacity:.6;margin:10px 0 4px}.seg:first-child{margin-top:0}</style>" % config.INK)
MAX_MULTI_HEIGHT = 340   # px: a multi-segment transcript scrolls inside this instead of growing with the segment count


def _displacy_body(text: str, entities: list[dict]) -> str:
    ents = [{"start": e["span"][0], "end": e["span"][1], "label": e["label"]} for e in _clean(entities, text)]
    return displacy.render([{"text": text, "ents": ents, "title": None}], style="ent", manual=True, page=False,
                           options={"ents": list(config.LABEL_COLORS), "colors": config.LABEL_COLORS})


def _wrap(inner: str) -> str:
    return (f"<html><head>{_DISPLACY_CSS}</head><body><div style='background:#0c2342;border:1px solid #1d3a5f;border-radius:10px;"
            f"padding:12px 16px'>{inner}</div></body></html>")


def displacy_html(text: str, entities: list[dict]) -> str:
    return _wrap(_displacy_body(text, entities))


def render_displacy(text: str, entities: list[dict]) -> None:
    n_lines = max(1, len(text) // 55 + 1)
    components.html(displacy_html(text, entities), height=70 + 54 * n_lines, scrolling=True)


def segments_displacy_html(segments) -> str:
    """All segments' highlighted transcripts in one document, each under a 'Segment k' label (one iframe, not one per segment)."""
    parts = []
    for i, seg in enumerate(segments, 1):
        parts.append(f"<div class='seg'>Segment {i} · {seg.start_s:.1f}–{seg.end_s:.1f} s</div>")
        parts.append(_displacy_body(seg.text, seg.entities) if seg.text else "<div style='opacity:.6'>(no speech recognized)</div>")
    return _wrap("".join(parts))


def render_segments_displacy(segments) -> None:
    """Highlighted transcript for a multi-segment clip: one box of bounded height that scrolls, so it cannot push the rest of the tab away."""
    lines = sum(max(1, len(s.text) // 55 + 1) for s in segments)
    natural = 40 * len(segments) + 54 * lines + 30
    components.html(segments_displacy_html(segments), height=min(natural, MAX_MULTI_HEIGHT), scrolling=True)


def legend(present: set[str] | None = None) -> None:
    from src.ner.schema import NER_LABELS

    chips = []
    for label, color in config.LABEL_COLORS.items():
        absent = present is not None and label not in present  # not in this transcript: outline only, still readable
        paint = f"background:transparent;color:{color};border:1px solid {color}" if absent else f"background:{color};color:{config.INK};border:1px solid {color}"
        tip = html.escape(NER_LABELS.get(label, ""), quote=True)
        chips.append(f"<span title=\"{tip}\" style='{paint};border-radius:12px;padding:2px 10px;"
                     f"margin:2px 4px 2px 0;font-size:.8rem;font-weight:600;display:inline-block'>"
                     f"{config.LABEL_TITLES[label]}</span>")
    st.markdown("".join(chips), unsafe_allow_html=True)


def _chip(text: str, label: str) -> str:
    return (f"<span style='background:{config.LABEL_COLORS[label]};color:{config.INK};border-radius:6px;padding:3px 9px;"
            f"margin:2px 6px 2px 0;font-weight:600;display:inline-block'>{html.escape(text)}</span>")


def card_html(entities: list[dict]) -> str:
    """Structured clearance layout: callsign headline, then one row per kind of item."""
    calls = [e["text"] for e in entities if e["label"] == "CALLSIGN"]
    head = " · ".join(dict.fromkeys(t.upper() for t in calls)) or "Callsign not detected"
    rows = []
    for label in CARD_ROWS:
        items = [e["text"] for e in entities if e["label"] == label]
        if items:
            chips = "".join(_chip(t, label) for t in items)
            rows.append(f"<tr><td style='padding:6px 14px 6px 0;opacity:.75;white-space:nowrap;vertical-align:top;border:0'>"
                        f"{config.LABEL_TITLES[label]}</td><td style='border:0'>{chips}</td></tr>")
    return ("<div style='border:1px solid rgba(128,128,128,.4);border-radius:10px;padding:14px 18px'>"
            "<div style='font-size:.75rem;letter-spacing:.08em;opacity:.7'>AIRCRAFT</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:{config.LABEL_COLORS['CALLSIGN']}'>{html.escape(head)}</div>"
            "<table style='border-collapse:collapse;margin-top:6px;border:0'>" + "".join(rows) + "</table></div>")


def render_card(entities: list[dict]) -> None:
    if not entities:
        st.info("No aviation entities were found in this transcript.")
        return
    st.markdown(card_html(entities), unsafe_allow_html=True)


def segment_cards_html(segments) -> str:
    """One clearance card per segment inside a scrolling box. A multi-segment clip holds several separate transmissions, so merging
    them into one card would join unrelated callsigns into a single headline."""
    blocks = []
    for i, seg in enumerate(segments, 1):
        label = f"<div style='font-size:.72rem;letter-spacing:.08em;opacity:.6;margin:12px 0 4px'>Segment {i} · {seg.start_s:.1f}–{seg.end_s:.1f} s</div>"
        body = card_html(seg.entities) if seg.entities else "<div style='opacity:.6;font-size:.9rem'>No aviation entities in this segment.</div>"
        blocks.append(label + body)
    return f"<div style='max-height:22rem;overflow-y:auto;padding-right:6px;scrollbar-width:thin'>{''.join(blocks)}</div>"


def render_segment_cards(segments) -> None:
    st.markdown(segment_cards_html(segments), unsafe_allow_html=True)
