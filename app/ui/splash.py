"""Splash page: animated radar / air-traffic canvas backdrop and an Enter button with a slow cross-fade to the app."""

from __future__ import annotations

import time

import streamlit as st

FADE_S = 1.8

_CSS = """
<style>
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], footer {display:none !important}
[data-testid="stAppViewContainer"], .stApp {background:#04101f !important}
.stMain .block-container {padding:0 !important; max-width:100% !important}
.st-key-ciq_enter {position:fixed; left:50%; top:calc(50% + 120px); transform:translateX(-50%); z-index:10001; width:auto}
.st-key-ciq_enter button {background:rgba(86,180,233,.12); color:#e8f4ff; border:1px solid rgba(120,200,255,.7);
  padding:.85rem 3rem; border-radius:999px; font-size:1.05rem; letter-spacing:.18em; text-transform:uppercase;
  backdrop-filter:blur(6px); transition:all .35s ease; box-shadow:0 0 28px rgba(86,180,233,.25)}
.st-key-ciq_enter button:hover {background:rgba(86,180,233,.32); border-color:#bfe6ff; box-shadow:0 0 44px rgba(86,180,233,.55); color:#fff}
.st-key-ciq_enter button:focus-visible {outline:2px solid #fff; outline-offset:3px}
@keyframes ciqIn {from{opacity:0} to{opacity:1}}
.ciq-text {animation:ciqIn 2.2s ease .3s both}
.st-key-ciq_enter {animation:ciqIn 2.2s ease 1.1s both}
</style>
"""

_LEAVE_CSS = f"""
<style>
@keyframes ciqOut {{from{{opacity:1}} to{{opacity:0}}}}
#ciq-splash, .st-key-ciq_enter {{animation:ciqOut {FADE_S}s ease-in-out forwards !important; pointer-events:none}}
</style>
"""

ENTER_CSS = f"""
<style>
@keyframes ciqHomeIn {{from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:none}}}}
[data-testid="stAppViewContainer"] > .stMain, [data-testid="stSidebar"] {{animation:ciqHomeIn {FADE_S}s ease-out both}}
</style>
"""

_HTML = """
<div id="ciq-splash" style="position:fixed;inset:0;z-index:10000;overflow:hidden;
  background:radial-gradient(ellipse at 70% 40%,#0d2a4d 0%,#071a33 45%,#030c18 100%)">
  <canvas id="ciq-canvas" style="position:absolute;inset:0;width:100%;height:100%"></canvas>
  <div class="ciq-text" style="position:absolute;left:0;right:0;top:calc(50% - 190px);text-align:center;color:#eaf5ff;
       font-family:'Source Sans Pro',system-ui,sans-serif;pointer-events:none;padding:0 16px">
    <div style="letter-spacing:.5em;font-size:.8rem;color:#7fc4f0;margin-bottom:14px">AIR TRAFFIC COMMUNICATIONS</div>
    <div style="font-size:clamp(2.6rem,7vw,5rem);font-weight:700;letter-spacing:.04em;text-shadow:0 0 40px rgba(86,180,233,.55)">
      Clearance<span style="color:#56B4E9">IQ</span></div>
    <div style="font-size:clamp(1rem,2vw,1.3rem);opacity:.85;margin-top:10px;font-weight:300">
      From radio transmission to structured clearance</div>
  </div>
  <div class="ciq-text" style="position:absolute;left:0;right:0;bottom:22px;text-align:center;color:#8fb3d1;font-size:.78rem;
       letter-spacing:.06em;font-family:system-ui,sans-serif;padding:0 16px">
    Academic research prototype &middot; not certified or validated for operational use</div>
</div>
<script>
(function () {
  const old = window.__ciqRaf; if (old) cancelAnimationFrame(old);
  const cv = document.getElementById('ciq-canvas'); if (!cv) return;
  const ctx = cv.getContext('2d');
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize(); window.addEventListener('resize', resize);

  const calls = ['DLH613', 'SAS4410', 'AFR1234', 'BAW88X', 'KLM1021', 'SWR72', 'AUA415', 'RYR5KZ', 'EZY18', 'UAL901'];
  const rnd = (a, b) => a + Math.random() * (b - a);
  const planes = calls.map((c, i) => {
    const hdg = rnd(0, Math.PI * 2), spd = rnd(14, 34);
    return {call: c, x: rnd(0, 1), y: rnd(0, 1), vx: Math.cos(hdg) * spd, vy: Math.sin(hdg) * spd,
            fl: 80 + Math.floor(rnd(0, 30)) * 10, trail: []};
  });
  let t0 = performance.now(), last = t0, trailTick = 0;

  function flow(t) {  // slow streamlines, like airflow / airway contours
    ctx.lineWidth = 1;
    for (let i = 0; i < 16; i++) {
      const base = (i + 0.5) / 16 * H;
      ctx.beginPath();
      for (let x = 0; x <= W; x += 14) {
        const y = base + Math.sin(x * 0.004 + t * 0.00018 * (1 + i * 0.07) + i) * 26
                       + Math.sin(x * 0.0011 - t * 0.00011 + i * 2) * 52;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(86,180,233,' + (0.05 + 0.04 * Math.sin(i + t * 0.0004)) + ')';
      ctx.stroke();
    }
  }
  function grid() {
    ctx.strokeStyle = 'rgba(86,180,233,0.045)'; ctx.lineWidth = 1;
    const s = 90;
    for (let x = 0; x < W; x += s) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += s) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  }
  function radar(t) {
    const cx = W * 0.5, cy = H * 0.5, R = Math.min(Math.max(W, H) * 0.62, 760);
    ctx.save(); ctx.translate(cx, cy);
    for (let r = 1; r <= 5; r++) {
      ctx.beginPath(); ctx.arc(0, 0, R * r / 5, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(120,200,255,' + (r === 5 ? 0.22 : 0.11) + ')'; ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(120,200,255,0.10)';
    for (let a = 0; a < 12; a++) {
      const th = a * Math.PI / 6; ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(Math.cos(th) * R, Math.sin(th) * R); ctx.stroke();
    }
    const ang = (t * 0.00035) % (Math.PI * 2);
    for (let i = 0; i < 70; i++) {  // fading sweep trail
      const a1 = ang - i * 0.012, a2 = ang - (i + 1) * 0.012;
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, R, a2, a1); ctx.closePath();
      ctx.fillStyle = 'rgba(86,180,233,' + (0.16 * (1 - i / 70) * (1 - i / 70)) + ')'; ctx.fill();
    }
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(Math.cos(ang) * R, Math.sin(ang) * R);
    ctx.strokeStyle = 'rgba(160,225,255,0.55)'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.restore();
    return {cx, cy, ang};
  }
  function aircraft(dt, t, rd) {
    const doTrail = (trailTick += dt) > 1.2; if (doTrail) trailTick = 0;
    ctx.font = '11px ui-monospace, monospace';
    for (const p of planes) {
      if (!reduce) { p.x += p.vx * dt / W; p.y += p.vy * dt / H; }
      if (p.x < -0.05) p.x = 1.05; if (p.x > 1.05) p.x = -0.05; if (p.y < -0.05) p.y = 1.05; if (p.y > 1.05) p.y = -0.05;
      const X = p.x * W, Y = p.y * H;
      if (doTrail) { p.trail.push([X, Y]); if (p.trail.length > 12) p.trail.shift(); }
      p.trail.forEach((q, i) => {
        ctx.fillStyle = 'rgba(143,214,255,' + (0.05 + 0.3 * i / p.trail.length) + ')';
        ctx.beginPath(); ctx.arc(q[0], q[1], 1.8, 0, 6.3); ctx.fill();
      });
      const sweepDelta = ((Math.atan2(Y - rd.cy, X - rd.cx) - rd.ang) % 6.2832 + 6.2832) % 6.2832;
      const glow = Math.max(0, 1 - sweepDelta / 2.2);  // brightens just after the sweep passes
      ctx.fillStyle = 'rgba(190,235,255,' + (0.45 + 0.55 * glow) + ')';
      ctx.fillRect(X - 3, Y - 3, 6, 6);
      ctx.strokeStyle = 'rgba(143,214,255,' + (0.15 + 0.4 * glow) + ')'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X + 4, Y - 4); ctx.lineTo(X + 22, Y - 22); ctx.stroke();
      ctx.fillStyle = 'rgba(190,235,255,' + (0.35 + 0.5 * glow) + ')';
      ctx.fillText(p.call, X + 25, Y - 24); ctx.fillText('FL' + p.fl, X + 25, Y - 11);
    }
  }
  function voice(t) {  // audio waveform ribbon along the bottom
    const y0 = H - 70;
    for (let k = 0; k < 2; k++) {
      ctx.beginPath();
      for (let x = 0; x <= W; x += 4) {
        const env = Math.pow(Math.sin(x / W * Math.PI), 2) * (0.5 + 0.5 * Math.sin(t * 0.0009 + x * 0.004 + k));
        const y = y0 + Math.sin(x * 0.05 + t * 0.004 + k * 2) * 22 * env;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = k ? 'rgba(86,180,233,0.25)' : 'rgba(230,159,0,0.30)'; ctx.lineWidth = 1.4; ctx.stroke();
    }
  }
  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1); last = now;
    const t = now - t0;
    ctx.clearRect(0, 0, W, H);
    grid(); flow(t); const rd = radar(t); aircraft(dt, t, rd); voice(t);
    if (!reduce) window.__ciqRaf = requestAnimationFrame(frame);
  }
  window.__ciqRaf = requestAnimationFrame(frame);
})();
</script>
"""


def render_splash() -> None:
    """Draws the splash. Returns after the Enter button was clicked and the fade-out finished (caller then reruns)."""
    ss = st.session_state
    st.html(_CSS)
    st.html(_HTML, unsafe_allow_javascript=True)
    if ss.get("leaving"):
        st.html(_LEAVE_CSS)
        st.button("Enter ClearanceIQ", key="ciq_enter", disabled=True)
        time.sleep(FADE_S + 0.1)
        ss.entered, ss.leaving, ss.just_entered = True, False, True
        st.rerun()
    if st.button("Enter ClearanceIQ", key="ciq_enter"):
        ss.leaving = True
        st.rerun()
