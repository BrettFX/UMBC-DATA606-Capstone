"""Waveform and Mel spectrogram figures."""

from __future__ import annotations

import librosa
import numpy as np
import plotly.graph_objects as go

_LAYOUT = dict(margin=dict(l=40, r=10, t=30, b=35), height=240, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
               plot_bgcolor="rgba(12,35,66,0.55)", font=dict(color="#e6f1fb"))


def add_segment_bands(fig: go.Figure, spans: list[tuple[float, float]] | None, t0: float = 0.0) -> go.Figure:
    """Mark where a long recording was split for the ASR model: every second segment is lightly shaded, cuts get a dotted line and
    each segment a small S1, S2... label. `spans` are (start, end) in prepared-audio seconds; `t0` shifts them onto the chart's
    original-recording timeline. Does nothing for a single segment."""
    if not spans or len(spans) < 2:
        return fig
    for i, (a, b) in enumerate(spans, 1):
        x0, x1 = t0 + a, t0 + b
        if i % 2 == 0:
            fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(150,205,235,0.13)", line_width=0, layer="above")
        if i > 1:
            fig.add_vline(x=x0, line=dict(color="rgba(232,243,251,0.5)", width=1, dash="dot"), layer="above")
        fig.add_annotation(x=(x0 + x1) / 2, y=1, yref="paper", yanchor="top", text=f"S{i}", showarrow=False,
                           font=dict(size=10, color="#d6e9f7"), bgcolor="rgba(6,20,42,0.6)", borderpad=2)
    return fig


def waveform_fig(y: np.ndarray, sr: int, t0: float = 0.0, max_points: int = 3000, spans: list[tuple[float, float]] | None = None) -> go.Figure:
    n = len(y)
    bins = min(max_points, n)
    edges = np.linspace(0, n, bins + 1, dtype=int)
    lo = np.array([y[a:b].min() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    hi = np.array([y[a:b].max() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    t = t0 + np.linspace(0, n / sr, len(lo))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", line=dict(width=0.5, color="#5b9fc4"), name="max", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=lo, mode="lines", line=dict(width=0.5, color="#5b9fc4"), fill="tonexty",
                             fillcolor="rgba(91,159,196,0.45)", name="min", hoverinfo="skip"))
    fig.update_layout(title="Waveform", xaxis_title="Time in recording (s)", yaxis_title="Amplitude", dragmode=False, showlegend=False, **_LAYOUT)
    fig.update_yaxes(range=[-1.05, 1.05])
    return add_segment_bands(fig, spans, t0)


def mel_fig(y: np.ndarray, sr: int, t0: float = 0.0, n_mels: int = 80, spans: list[tuple[float, float]] | None = None) -> go.Figure:
    """Log-Mel spectrogram with Whisper's front-end parameters (25 ms window, 10 ms hop, 80 bands)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=n_mels, fmax=sr / 2)
    db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    t = t0 + np.arange(db.shape[1]) * 160 / sr
    f = librosa.mel_frequencies(n_mels=n_mels, fmax=sr / 2)
    fig = go.Figure(go.Heatmap(z=db, x=t, y=f, colorscale=[[0, "#06142a"], [0.25, "#16395e"], [0.5, "#3a6d93"], [0.75, "#7fa6b5"], [1, "#d3c9a6"]], colorbar=dict(title="dB"),
                               hovertemplate="t=%{x:.2f}s  f=%{y:.0f}Hz  %{z:.0f}dB<extra></extra>"))
    fig.update_layout(title="Mel spectrogram", xaxis_title="Time in recording (s)", yaxis_title="Frequency (Hz)", dragmode=False, **{**_LAYOUT, "height": 300})
    return add_segment_bands(fig, spans, t0)
