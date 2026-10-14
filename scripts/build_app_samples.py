"""Export a small set of held-out test clips (audio + ground truth + gold entities) for the ClearanceIQ app.

Run from the repo root: `python scripts/build_app_samples.py [--per-source 4]`.
Picks clips whose utterance is in `data/ner_gold/heldout_gold.jsonl` and in the test split, so the app can
show ground-truth transcripts and entity scores without any training leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "samples"


def _export(row, split: str, note: str) -> dict:
    audio = row["audio"]
    y = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    name = f"{row['utterance_id']}.wav"
    sf.write(OUT / name, y, sr, subtype="PCM_16")
    return {"id": row["utterance_id"], "file": name, "source": row["dataset_source"], "duration_s": round(len(y) / sr, 2),
            "ground_truth": row["text"], "entities": [], "split": split, "note": note}


def extra_samples(ds_all, args) -> list[dict]:
    """Validation-split clips (a spread of lengths per source) plus the longest clip in the whole dataset."""
    out = []
    val = ds_all["validation"]
    for src in ("atcosim", "atc-asr-dataset"):
        idx = [i for i, (u, e) in enumerate(zip(val["dataset_source"], val["audio_decode_error"])) if u == src and not e]
        durs = np.array([len(val[i]["audio"]["array"]) / val[i]["audio"]["sampling_rate"] for i in idx])
        order = np.argsort(durs)
        for q in (0.25, 0.55, 0.9):  # short, medium, long-ish
            out.append(_export(val[idx[order[int(q * (len(idx) - 1))]]], "validation", "Validation split"))
    best = None
    for split in ("train", "validation", "test"):
        d = ds_all[split]
        lens = [len(a["array"]) / a["sampling_rate"] for a in d.select_columns(["audio"])["audio"]]
        i = int(np.argmax(lens))
        if best is None or lens[i] > best[0]:
            best = (lens[i], d[i], split)
    out.append(_export(best[1], best[2], f"Longest clip in the dataset ({best[2]} split); exercises splitting at pauses"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=4)
    ap.add_argument("--max-seconds", type=float, default=12.0)
    args = ap.parse_args()

    gold = {r["utterance_id"]: r for r in map(json.loads, open(ROOT / "data/ner_gold/heldout_gold.jsonl"))}
    ds_all = load_from_disk(str(ROOT / "data/processed/combined_dataset"))
    ds = ds_all["test"]
    ids = ds["utterance_id"]
    picked: dict[str, list[dict]] = {}
    for i, uid in enumerate(ids):
        g = gold.get(uid)
        if not g or ds[i]["audio_decode_error"]:
            continue
        src = g["dataset_source"]
        if len(picked.setdefault(src, [])) >= args.per_source:
            continue
        audio = ds[i]["audio"]
        y = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        if len(y) / sr > args.max_seconds or len(y) / sr < 2:
            continue
        name = f"{uid}.wav"
        sf.write(OUT / name, y, sr, subtype="PCM_16")
        picked[src].append({"id": uid, "file": name, "source": src, "duration_s": round(len(y) / sr, 2),
                            "ground_truth": g["text"], "entities": g["entities"]})
    samples = [{**s, "split": "test"} for lst in picked.values() for s in lst]
    samples += extra_samples(ds_all, args)
    (OUT / "samples.json").write_text(json.dumps(samples, indent=2))
    print(f"wrote {len(samples)} samples to {OUT}")


if __name__ == "__main__":
    main()
