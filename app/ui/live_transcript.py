"""Always-visible transcript under the audio player. Each word is a span with its start/end time; playhead.py lights up the
word being spoken and lets you click a word to jump there. Entities are underlined in their label colour."""

from __future__ import annotations

import html

import streamlit as st

from app import config
from app.services.pipeline import Result

_CSS = """
<style>
.ciq-live {position:relative; max-height:14rem; overflow-y:auto; border:1px solid #1d3a5f; border-radius:12px; padding:14px 18px 12px;
  background:rgba(8,26,50,.72); margin:6px 0 4px; line-height:2.25; font-size:1.18rem; scrollbar-width:thin}
.ciq-live-head {position:sticky; top:-14px; z-index:2; display:flex; justify-content:space-between; align-items:baseline; gap:12px;
  margin:-14px -18px 2px; padding:14px 18px 4px; background:#0b2140; font-size:.72rem; letter-spacing:.1em; text-transform:uppercase}
.ciq-live-head span {opacity:.65}
.ciq-live-head .ciq-now {opacity:1; color:#9fd0ee}
.ciq-seg {font-size:.72rem; letter-spacing:.08em; opacity:.55; margin:6px 0 0}
.ciq-w {display:inline-block; padding:0 6px; margin:0 1px; border-radius:7px; cursor:pointer; opacity:.92;
  border-bottom:2px solid transparent; transition:background .16s ease, color .16s ease, opacity .25s ease, box-shadow .2s ease, transform .16s ease}
.ciq-w:hover {background:rgba(106,174,212,.14)}
.ciq-w.done {opacity:.5}
.ciq-w.on {opacity:1; color:#fff; background:rgba(106,174,212,.30); box-shadow:0 0 0 1px rgba(150,205,235,.55), 0 0 16px rgba(106,174,212,.55);
  transform:translateY(-2px) scale(1.07); animation:ciqPop .22s ease-out}
@keyframes ciqPop {0% {transform:translateY(0) scale(1)} 60% {transform:translateY(-3px) scale(1.12)} 100% {transform:translateY(-2px) scale(1.07)}}
.ciq-live.nowords .ciq-w {cursor:default}
</style>
"""


def _word_label(text: str, entities: list[dict]) -> dict[int, str]:
    """Index of each whitespace-separated word -> label of the entity covering it."""
    out, pos = {}, 0
    for i, w in enumerate(text.split()):
        start = text.index(w, pos)
        end = pos = start + len(w)
        for e in entities:
            if e["span"][0] <= start and end <= e["span"][1]:
                out[i] = e["label"]
                break
    return out


def build_html(result: Result) -> str:
    """The transcript box as HTML. It scrolls inside a fixed maximum height, so a long multi-segment clip never pushes the charts
    and tabs below it out of reach; `playhead.py` scrolls the box to follow the spoken word and shows which segment is playing."""
    parts = []
    timed = any(s.words for s in result.segments)
    multi = len(result.segments) > 1
    for n, seg in enumerate(result.segments, 1):
        if not seg.text:
            continue
        if multi:
            parts.append(f"<div class='ciq-seg'>Segment {n}</div>")
        labels = _word_label(seg.text, seg.entities)
        words = seg.words or []
        for i, w in enumerate(seg.text.split()):
            style = f"border-bottom-color:{config.LABEL_COLORS[labels[i]]};" if i in labels else ""
            timing = f" data-s='{words[i].start:.3f}' data-e='{words[i].end:.3f}'" if i < len(words) and len(words) == len(seg.text.split()) else ""
            parts.append(f"<span class='ciq-w' style='{style}'{timing}>{html.escape(w)}</span>")
    body = " ".join(parts) or "<span style='opacity:.6'>No speech recognized.</span>"
    hint = "click a word to jump · underline = entity" if timed else "underline = entity"
    now = "<span class='ciq-now'></span>" if multi else ""
    return (_CSS + f"<div class='ciq-live{'' if timed else ' nowords'}'><div class='ciq-live-head'><span>Transcript</span>{now}"
            f"<span>{hint}</span></div>{body}</div>")


def render(result: Result) -> None:
    st.html(build_html(result))
