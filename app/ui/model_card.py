"""Hugging Face-style model cards: header with tags and key facts, then Overview / Metrics / Training tabs, plus a comparison."""

from __future__ import annotations

import html

import streamlit as st

from app import config
from app.services import model_cards as mc

_CSS = """
<style>
.mc-head {border:1px solid #1d3a5f; border-radius:12px; padding:18px 22px; background:linear-gradient(135deg,rgba(86,180,233,.10),rgba(12,35,66,.6))}
.mc-title {font-size:1.55rem; font-weight:700; letter-spacing:.01em}
.mc-sub {opacity:.75; margin:2px 0 10px}
.mc-tag {display:inline-block; font-size:.74rem; padding:2px 10px; margin:0 6px 6px 0; border-radius:999px; border:1px solid #2a4f7c;
         background:rgba(86,150,200,.12); color:#aacde6}
.mc-tag.k {background:rgba(200,160,90,.12); border-color:rgba(200,160,90,.4); color:#e0c58f}
.mc-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-top:12px}
.mc-fact {border:1px solid #1d3a5f; border-radius:10px; padding:8px 12px; background:rgba(4,16,31,.55)}
.mc-fact .l {font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; opacity:.65}
.mc-fact .v {font-size:1.02rem; font-weight:600; margin-top:2px}
.mc-tbl {border-collapse:collapse; width:100%; font-size:.92rem}
.mc-tbl th {text-align:left; font-weight:600; opacity:.75; border-bottom:1px solid #1d3a5f; padding:6px 10px; font-size:.78rem; letter-spacing:.05em; text-transform:uppercase}
.mc-tbl td {border-bottom:1px solid rgba(29,58,95,.6); padding:6px 10px; vertical-align:middle}
.mc-bar {height:8px; border-radius:4px; background:rgba(255,255,255,.08); min-width:90px}
.mc-bar > div {height:8px; border-radius:4px}
.mc-up {color:#76B89E} .mc-down {color:#CE937A} .mc-eq {opacity:.5}
.mc-note {font-size:.82rem; opacity:.7}
</style>
"""


def _e(x) -> str:
    return html.escape(str(x))


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def header(card: mc.ModelCard, selected: bool) -> None:
    tags = "".join(f"<span class='mc-tag'>{_e(t)}</span>" for t in card.tags)
    keys = (f"<span class='mc-tag k'>{_e(card.task)}</span><span class='mc-tag k'>{_e(card.library)}</span>"
            f"<span class='mc-tag k'>{_e(card.language)}</span>" + ("<span class='mc-tag k'>selected in sidebar</span>" if selected else ""))
    facts = "".join(f"<div class='mc-fact'><div class='l'>{_e(k)}</div><div class='v'>{_e(v)}</div></div>" for k, v in card.facts.items())
    st.markdown(f"<div class='mc-head'><div class='mc-title'>{_e(card.title)}</div><div class='mc-sub'>{_e(card.subtitle)} · "
                f"base: <b>{_e(card.base)}</b></div>{keys}<div style='margin-top:4px'>{tags}</div><div class='mc-grid'>{facts}</div></div>",
                unsafe_allow_html=True)


def _bullets(items: list[str]) -> None:
    st.markdown("\n".join(f"- {i}" for i in items))


def _label_bars(per_label: dict, support: dict | None) -> str:
    rows = []
    for label in mc.NER_LABELS_ORDER:
        f1 = per_label.get(label)
        if f1 is None:
            continue
        color = config.LABEL_COLORS[label]
        sup = f"{support[label]:,}" if support and label in support else ""
        rows.append(f"<tr><td style='white-space:nowrap'><span style='background:{color};color:{config.INK};border-radius:10px;padding:1px 9px;"
                    f"font-size:.78rem;font-weight:600'>{config.LABEL_TITLES[label]}</span></td>"
                    f"<td style='width:55%'><div class='mc-bar'><div style='width:{f1 * 100:.0f}%;background:{color};opacity:.85'></div></div></td>"
                    f"<td>{f1:.1%}</td><td class='mc-note'>{sup}</td></tr>")
    return ("<table class='mc-tbl'><tr><th>Label</th><th>F1</th><th></th><th>Support</th></tr>" + "".join(rows) + "</table>")


def overview(card: mc.ModelCard) -> None:
    st.markdown(f"#### Model description\n{card.summary}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Intended use")
        _bullets(card.intended_use)
    with c2:
        st.markdown("#### Limitations and risks")
        _bullets(card.limitations)
    st.caption(config.DISCLAIMER)


def metrics_tab(card: mc.ModelCard) -> None:
    if card.key == "asr":
        t = card.metrics.get("test", {})
        c = st.columns(3)
        c[0].metric("WER (test)", _pct(t.get("wer")), help="Word error rate: (substitutions + deletions + insertions) ÷ reference words. Lower is better.")
        c[1].metric("CER (test)", _pct(t.get("cer")), help="Character error rate. Lower is better.")
        c[2].metric("Test utterances", f"{t.get('n', 0):,}")
        rows = "".join(f"<tr><td>{_e(src)}</td><td>{v['n']:,}</td><td>{v['wer']:.1%}</td><td>{v['cer']:.1%}</td></tr>"
                       for src, v in t.get("by_source", {}).items())
        st.markdown("#### By data source")
        st.markdown(f"<table class='mc-tbl'><tr><th>Source</th><th>Utterances</th><th>WER</th><th>CER</th></tr>{rows}</table>", unsafe_allow_html=True)
        st.caption("ATCOSIM is simulated, clean speech; ATC-ASR is real radio with noise and varied accents.")
        st.caption(card.metric_note)
        return
    sets = {k: v for k, v in card.metrics.items() if k in mc.NER_EVAL_SETS}
    rows = "".join(f"<tr><td><b>{_e(mc.NER_EVAL_SETS[k][0])}</b><div class='mc-note'>{_e(mc.NER_EVAL_SETS[k][1])}</div></td>"
                   f"<td>{v['n']:,}</td><td>{_pct(v['precision'])}</td><td>{_pct(v['recall'])}</td><td><b>{_pct(v['f1'])}</b></td>"
                   f"<td>{_pct(v['macro_f1'])}</td></tr>" for k, v in sets.items())
    st.markdown("#### Evaluation sets")
    st.markdown(f"<table class='mc-tbl'><tr><th>Set</th><th>Utterances</th><th>Precision</th><th>Recall</th><th>F1</th><th>Macro F1</th></tr>{rows}</table>",
                unsafe_allow_html=True)
    st.caption(card.metric_note + " Macro F1 averages the per-label F1 scores, so rare labels count as much as common ones.")
    pick = st.selectbox("Per-label F1 on", list(sets), format_func=lambda k: mc.NER_EVAL_SETS[k][0], index=list(sets).index("heldout") if "heldout" in sets else 0,
                        key=f"lbl_{card.key}")
    st.markdown(_label_bars(sets[pick]["per_label_f1"], sets[pick].get("support")), unsafe_allow_html=True)


def training_tab(card: mc.ModelCard) -> None:
    rows = "".join(f"<tr><td style='white-space:nowrap;opacity:.75'>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in card.training.items())
    st.markdown(f"<table class='mc-tbl'>{rows}</table>", unsafe_allow_html=True)


def compare_ner(cards: dict[str, mc.ModelCard]) -> None:
    ner = [c for c in cards.values() if c.key != "asr"]
    if len(ner) < 2:
        st.info("Install both entity models to compare them.")
        return
    a, b = ner[0], ner[1]
    st.markdown(f"#### {a.title} vs {b.title}")
    st.markdown(f"*{config.NER_BLURBS.get(a.key, '')}*  \n*{config.NER_BLURBS.get(b.key, '')}*")
    rows = []
    for k, (title, _) in mc.NER_EVAL_SETS.items():
        if k in a.metrics and k in b.metrics:
            fa, fb = a.metrics[k]["f1"], b.metrics[k]["f1"]
            d = fb - fa
            cls = "mc-eq" if abs(d) < 0.005 else ("mc-up" if d > 0 else "mc-down")
            rows.append(f"<tr><td>{_e(title)}</td><td>{a.metrics[k]['n']:,}</td><td>{fa:.1%}</td><td>{fb:.1%}</td>"
                        f"<td class='{cls}'>{d * 100:+.1f} pts</td></tr>")
    st.markdown(f"<table class='mc-tbl'><tr><th>Set (F1)</th><th>Utterances</th><th>{_e(a.title)}</th><th>{_e(b.title)}</th><th>Difference</th></tr>{''.join(rows)}</table>",
                unsafe_allow_html=True)
    st.markdown("#### Which should I pick?")
    st.markdown("- **Standard** is the safe default. It keeps the label frequencies of real traffic and scores highest on three of the four sets "
                "(dev, test and held-out).\n"
                "- **Rare-label balanced** scored highest on the small hand-labeled gold set, mainly through headings and runways, "
                "the labels with the fewest training examples.\n"
                "- Differences are **at most about 1.5 points** and the human-labeled sets have only ~100 utterances, so treat the two as "
                "**roughly equal** rather than a clear winner.")
    heldout = lambda c: c.metrics.get("heldout", {}).get("per_label_f1", {})
    diffs = [(l, heldout(b).get(l, 0) - heldout(a).get(l, 0)) for l in mc.NER_LABELS_ORDER if l in heldout(a) and l in heldout(b)]
    st.markdown("#### Per-label difference on the held-out set (balanced − standard, F1 points)")
    cells = "".join(f"<tr><td>{config.LABEL_TITLES[l]}</td><td class='{'mc-up' if d > 0.005 else 'mc-down' if d < -0.005 else 'mc-eq'}'>{d * 100:+.1f}</td></tr>" for l, d in diffs)
    st.markdown(f"<table class='mc-tbl' style='max-width:360px'><tr><th>Label</th><th>Δ F1</th></tr>{cells}</table>", unsafe_allow_html=True)


def models_tab(selected_ner: str) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    cards = mc.all_cards()
    options = list(cards) + ["Compare entity models"]
    default = options.index(selected_ner) if selected_ner in options else 0
    choice = st.radio("Model", options, index=default, horizontal=True, label_visibility="collapsed", key="mc_choice")
    if choice == "Compare entity models":
        compare_ner(cards)
        return
    card = cards[choice]
    header(card, selected=(card.key == selected_ner))
    t1, t2, t3 = st.tabs(["Overview", "Metrics", "Training"])
    with t1:
        overview(card)
    with t2:
        metrics_tab(card)
    with t3:
        training_tab(card)
