"""Syncs the page's audio player with the waveform, the Mel spectrogram and the transcript words: a moving playhead line,
the word being spoken lit up, and click-to-seek.

Streamlit renders `st.audio` as a plain <audio> element in the page, and Plotly draws into `.js-plotly-plot` divs, so one small
script can link them. The loop is installed once per page load and re-finds both elements on every frame, so it survives
Streamlit reruns that rebuild the charts. The charts' time axis is the original recording's timeline, so no offset is needed.
"""

from __future__ import annotations

import hashlib

import streamlit as st

_JS = """
<script>
(function () {
  // Install once per distinct version of this script. The token is a hash of the script text (filled in by Python), so a page that is
  // already open with an older copy picks up the new one on the next rerun instead of being blocked by a stale "installed" flag.
  const TOKEN = '__TOKEN__';
  if (window.__ciqToken === TOKEN) return;
  window.__ciqToken = TOKEN;                  // loops and listeners of older versions see the token change and stop
  const COLOR = '#e8f3fb';
  const audioEl = () => document.querySelector('[data-testid="stAudio"] audio') || document.querySelector('audio');

  function ensureLine(gd) {
    const host = gd.querySelector('.svg-container');
    if (!host) return null;
    let line = host.querySelector('.ciq-playhead');
    if (!line) {
      line = document.createElement('div');
      line.className = 'ciq-playhead';
      line.style.cssText = 'position:absolute;width:2px;margin-left:-1px;pointer-events:none;z-index:5;display:none;' +
        'background:' + COLOR + ';box-shadow:0 0 6px rgba(190,230,255,.9),0 0 14px rgba(120,190,240,.5)';
      host.appendChild(line);
    }
    if (gd.__ciqSeek !== TOKEN) {              // click anywhere in the plot area to seek
      gd.__ciqSeek = TOKEN;
      gd.style.cursor = 'pointer';
      gd.addEventListener('click', (ev) => {
        if (window.__ciqToken !== TOKEN) return;
        const a = audioEl(), fl = gd._fullLayout;
        if (!a || !fl || !fl.xaxis || !isFinite(a.duration)) return;
        const r = host.getBoundingClientRect();
        const px = ev.clientX - r.left - fl.xaxis._offset;
        if (px < 0 || px > fl.xaxis._length) return;
        const [x0, x1] = fl.xaxis.range;
        a.currentTime = Math.min(Math.max(x0 + px / fl.xaxis._length * (x1 - x0), 0), a.duration);
      });
    }
    return line;
  }

  let words = [], wordsKey = '';
  function updateWords(a) {
    const nodes = document.querySelectorAll('.ciq-w[data-s]');
    if (!nodes.length) return;
    if (nodes[0] !== words[0]) {            // transcript was re-rendered: refresh the list and wire up click-to-seek
      words = Array.from(nodes);
      words.forEach((el) => {
        el._s = parseFloat(el.dataset.s); el._e = parseFloat(el.dataset.e);
        el.addEventListener('click', () => {
          if (window.__ciqToken !== TOKEN) return;
          const au = audioEl(); if (au) au.currentTime = Math.max(el._s - 0.02, 0);
        });
      });
    }
    const t = a.currentTime, playing = !a.paused || t > 0;
    for (const el of words) {
      const on = playing && t >= el._s && t < Math.max(el._e, el._s + 0.12);
      const done = playing && !on && t >= el._e;
      if (on !== el._on) { el.classList.toggle('on', on); el._on = on; }
      if (done !== el._done) { el.classList.toggle('done', done); el._done = done; }
    }
  }

  function frame() {
    if (window.__ciqToken !== TOKEN) return;  // a newer copy of this script took over
    const a = audioEl();
    if (a) {
      updateWords(a);
      document.querySelectorAll('.js-plotly-plot').forEach((gd) => {
        const fl = gd._fullLayout, line = ensureLine(gd);
        if (!line || !fl || !fl.xaxis || !fl.xaxis.range) return;
        const [x0, x1] = fl.xaxis.range, t = a.currentTime;
        const visible = t >= x0 && t <= x1 && fl.xaxis._length > 0 && (t > 0 || !a.paused);
        line.style.display = visible ? 'block' : 'none';
        if (visible) {
          line.style.left = (fl.xaxis._offset + (t - x0) / (x1 - x0) * fl.xaxis._length) + 'px';
          line.style.top = fl.yaxis._offset + 'px';
          line.style.height = fl.yaxis._length + 'px';
        }
      });
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
"""


def inject() -> None:
    token = hashlib.sha1(_JS.encode()).hexdigest()[:10]
    st.html(_JS.replace("__TOKEN__", token), unsafe_allow_javascript=True)
