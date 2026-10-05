"""Resumable full-corpus annotation run: pool selection, config signature, chunked checkpointing.

A 9B-model pass over ~15k utterances takes hours, so the run is built to be interrupted: utterances
are processed in seeded-random chunks (any prefix is a representative sample), each finished chunk
is appended to a JSONL cache immediately, and a re-run skips every utterance already cached for the
same configuration. The cache file name carries a signature of everything that changes the labels
(model, prompt text, flags, and the source of the NER modules), so stale labels are never reused.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .annotate import FORMAT_SUFFIX, Backend, annotate_all
from .prompt import PROMPTS
from .schema import is_trainable_utterance

logger = logging.getLogger(__name__)

_MODULES = ("schema.py", "rules.py", "postprocess.py", "prompt.py", "annotate.py")
POOL_COLUMNS = ["utterance_id", "transcript_normalized", "dataset_split", "dataset_source"]


@dataclass(frozen=True)
class AnnotationConfig:
    """Everything that changes the labels an annotation run produces."""

    model: str
    prompt_version: int = 3
    use_hints: bool = True
    check_missed: bool = True
    postprocess: bool = True
    guided: bool = True
    max_rounds: int = 3

    def signature(self) -> str:
        """Short hash of the config, the prompt text and the source of the NER modules."""
        src = "".join((Path(__file__).parent / m).read_text() for m in _MODULES)
        payload = json.dumps(asdict(self), sort_keys=True) + PROMPTS[self.prompt_version] + FORMAT_SUFFIX + src
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def build_pool(utterances: pd.DataFrame) -> pd.DataFrame:
    """Utterances to annotate: deduplicated on normalized text, minus single-word and filler-only ones."""
    pool = utterances[utterances["transcript_normalized"].map(is_trainable_utterance)]
    return pool.drop_duplicates("transcript_normalized")[POOL_COLUMNS].reset_index(drop=True)


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["utterance_id"]: r for r in map(json.loads, (line for line in f if line.strip()))}


def _write_progress(path: Path, **fields) -> None:
    """Atomically replace the progress file a dashboard polls (never leaves a half-written file)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(fields))
    os.replace(tmp, path)


def run_annotation(backend: Backend, pool: pd.DataFrame, out_dir: Path, config: AnnotationConfig, *,
                   chunk_size: int = 256, limit: int | None = None, seed: int = 42) -> Path:
    """Annotate `pool` chunk by chunk into `out_dir/annotations_<signature>.jsonl`; returns that path.

    `limit` caps the number of utterances annotated in this call (a smoke test or a time-boxed run);
    already-cached utterances count toward neither the cap nor the work.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = config.signature()
    path, manifest_path = out_dir / f"annotations_{sig}.jsonl", out_dir / f"manifest_{sig}.json"
    cached = load_cache(path)
    order = pool.sample(frac=1, random_state=seed).reset_index(drop=True)  # seeded: a stable, representative order
    todo = order[~order["utterance_id"].isin(cached)]
    if limit is not None:
        todo = todo.head(limit)
    logger.info("config %s: %d in pool, %d cached, %d to annotate", sig, len(pool), len(cached), len(todo))

    started = datetime.now(timezone.utc).isoformat()
    system_prompt, t_start = PROMPTS[config.prompt_version], time.perf_counter()
    done, chunk_secs = 0, []
    progress_path = out_dir / f"progress_{sig}.json"

    def report(status: str) -> None:
        rows = list(cached.values())
        rate = done / max(time.perf_counter() - t_start, 1e-9)
        _write_progress(
            progress_path, status=status, signature=sig, model=config.model, pid=os.getpid(),
            pool_size=len(pool), annotated=len(rows), this_run_done=done, this_run_total=len(todo),
            rate_utt_per_s=rate, eta_seconds=(len(todo) - done) / rate if done else None,
            chunk_seconds=chunk_secs[-60:], started_utc=started, updated_utc=datetime.now(timezone.utc).isoformat(),
            ok_rate=sum(r["ok"] for r in rows) / max(len(rows), 1),
            retry_rate=sum(r["n_rounds"] > 1 for r in rows) / max(len(rows), 1),
            labels=dict(Counter(e["label"] for r in rows for e in r["entities"])))

    report("running")
    for lo in range(0, len(todo), chunk_size):
        chunk = todo.iloc[lo:lo + chunk_size]
        t0 = time.perf_counter()
        try:
            result = annotate_all(
                backend, dict(zip(chunk["utterance_id"], chunk["transcript_normalized"])), config.use_hints,
                config.max_rounds, check_missed=config.check_missed, postprocess=config.postprocess,
                system_prompt=system_prompt)
        except BaseException:
            report("error")  # the dashboard shows the failure; the cache still holds every finished chunk
            raise
        with open(path, "a") as f:  # the chunk is written only once it is complete
            for row in chunk.itertuples():
                rec = {"utterance_id": row.utterance_id, "text": row.transcript_normalized,
                       "dataset_split": row.dataset_split, "dataset_source": row.dataset_source,
                       **result[row.utterance_id]}
                cached[row.utterance_id] = rec
                f.write(json.dumps(rec) + "\n")
            f.flush()
        done += len(chunk)
        chunk_secs.append(round(time.perf_counter() - t0, 1))
        report("running")
        rate = done / (time.perf_counter() - t_start)
        logger.info("%d/%d this run (%.2f utt/s, ETA %.0f min); chunk took %.0fs", done, len(todo), rate,
                    (len(todo) - done) / rate / 60, time.perf_counter() - t0)

    final = load_cache(path)
    report("finished")
    manifest_path.write_text(json.dumps({
        "signature": sig, "config": asdict(config), "pool_size": len(pool), "annotated": len(final),
        "complete": len(final) >= len(pool), "seed": seed, "chunk_size": chunk_size,
        "ok_rate": sum(r["ok"] for r in final.values()) / max(len(final), 1),
        "retry_rate": sum(r["n_rounds"] > 1 for r in final.values()) / max(len(final), 1),
        "started_utc": started, "updated_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return path
