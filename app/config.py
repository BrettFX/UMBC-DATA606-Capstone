"""Paths, limits and display constants for the ClearanceIQ app."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):  # project modules import both `src.x` and top-level `preprocessing`
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
APP_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = APP_DIR / "samples"

SAMPLE_RATE = 16_000
ALLOWED_EXTENSIONS = ("wav", "mp3", "flac", "ogg", "m4a")
MAX_UPLOAD_MB = 50
MIN_DURATION_S = 0.5
MAX_DURATION_S = 300.0           # hard cap on total audio length
MAX_SEGMENT_S = 28.0             # Whisper window is 30 s; keep a margin
MIN_RMS = 1e-3                   # below this the clip is treated as silence
TRIM_TOP_DB = 40
TRIM_PAD_S = 0.15

ASR_MODEL_NAME = "Whisper medium (ATC fine-tuned, LoRA)"
NER_DIRS = {"spaCy CNN · Standard": REPO_ROOT / "models/ner/spacy/model-best",
            "spaCy CNN · Rare-label balanced": REPO_ROOT / "models/ner/spacy-balanced/model-best"}
DEFAULT_NER = "spaCy CNN · Standard"
NER_BLURBS = {
    "spaCy CNN · Standard": "Trained on the curated data as-is. Best on the large LLM-labeled test set and the adjudicated held-out set.",
    "spaCy CNN · Rare-label balanced": "Same model, trained with rare labels (headings, runways, squawks…) repeated up to 4×. Best on the 95-utterance hand-labeled gold set.",
}

LLM_MODELS = {"Claude Haiku 5.5 (fast)": "claude-haiku-5-5", "Claude Sonnet 5.5 (balanced)": "claude-sonnet-5-5"}
DEFAULT_LLM = "Claude Haiku 5.5 (fast)"

DISCLAIMER = ("ClearanceIQ is an **academic research prototype**. It is not certified, not validated for "
              "operational use, and must never be relied on for air traffic control or flight-safety decisions. "
              "Transcripts and interpretations are automatic and may be wrong.")

# Muted, distinct label colors for the dark theme (easy on the eyes); light enough for dark text (INK) to stay readable.
INK = "#0a1628"
LABEL_COLORS = {
    "CALLSIGN": "#D4A24A", "RUNWAY": "#6DB0CE", "ALTITUDE": "#5BAE92", "HEADING": "#C5BB6A",
    "FREQUENCY": "#C47FA6", "SQUAWK": "#D68663", "COMMAND": "#8296D4", "WAYPOINT": "#9DBE6E", "FACILITY": "#A692CC",
}
LABEL_TITLES = {
    "CALLSIGN": "Callsign", "RUNWAY": "Runway", "ALTITUDE": "Altitude / level", "HEADING": "Heading",
    "FREQUENCY": "Frequency", "SQUAWK": "Squawk", "COMMAND": "Instruction", "WAYPOINT": "Fix / waypoint",
    "FACILITY": "Facility",
}
