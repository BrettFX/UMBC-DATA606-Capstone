"""Collect trained-model artifacts and upload them to S3 under versioned, non-overwriting prefixes.

Layout: s3://<bucket>/<base>/<task-folder>/<run-id>/<model-name>/<files...> plus a manifest.json at the run
level. A fresh run-id per upload means a new model never silently replaces an earlier one, and `upload`
refuses to write into a prefix that already holds objects. Optimizer checkpoints are never included.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Item:
    source: Path
    rel_key: str  # path under the run prefix

    @property
    def size(self) -> int:
        return self.source.stat().st_size


def _tree(root: Path, rel_root: str) -> list[Item]:
    return [Item(p, f"{rel_root}/{p.relative_to(root).as_posix()}") for p in sorted(root.rglob("*")) if p.is_file()]


def collect(task: str, repo_root: Path, variants: list[str] | None = None) -> list[Item]:
    """Files to upload for `task` ('asr' or 'ner'): final models, their configs and scores, never checkpoints."""
    items: list[Item] = []
    if task == "asr":
        results = repo_root / "data/processed/asr_results"
        for model_dir in sorted((repo_root / "models").glob("*-atc-finetuned-full*")):
            if variants and model_dir.name not in variants:
                continue
            for part in ("final", "final-adapter"):  # the merged model, and the LoRA adapters when there are any
                if (model_dir / part).is_dir():
                    items += _tree(model_dir / part, f"{model_dir.name}/{part}")
            items += [Item(model_dir / f, f"{model_dir.name}/{f}") for f in ("train_config.json", "train_metrics.json", "test_scores.json")
                      if (model_dir / f).exists()]
            # this model's own per-utterance test results: models/<safe>-atc-finetuned-full<tag> -> finetuned_<safe>_full<tag>.csv
            safe, _, tag = model_dir.name.partition("-atc-finetuned-full")
            csv = results / f"finetuned_{safe}_full{tag}.csv"
            if csv.exists():
                items.append(Item(csv, f"test_results/{csv.name}"))
        items += [Item(results / name, f"test_results/{name}") for name in ("experiments_summary.md", "experiments_summary.csv")
                  if (results / name).exists()]  # the cross-model comparison table, for context
    elif task == "ner":
        for model_dir in sorted((repo_root / "models" / "ner").glob("*")):
            if not model_dir.is_dir() or (variants and model_dir.name not in variants):
                continue
            if (model_dir / "model-best").is_dir():
                items += _tree(model_dir / "model-best", f"{model_dir.name}/model-best")
            if (model_dir / "results.json").exists():
                items.append(Item(model_dir / "results.json", f"{model_dir.name}/results.json"))
        ner_data = repo_root / "data/processed/ner_dataset"
        items += [Item(p, f"data/{name}") for p, name in ((ner_data / "curated/manifest.json", "curated_manifest.json"),
                                                          (ner_data / "curated/label2id.json", "label2id.json"))
                  if p.exists()]
        items += [Item(p, f"data/{p.name}") for p in sorted((ner_data / "annotations").glob("manifest_*.json"))]
    else:
        raise ValueError(f"unknown task {task!r}")
    return items


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(task: str, run_id: str, items: list[Item], repo_root: Path) -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {"task": task, "run_id": run_id, "git_commit": commit, "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": [{"key": i.rel_key, "bytes": i.size, "sha256": _sha256(i.source)} for i in items]}


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri}")
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix.strip("/")


def upload(items: list[Item], manifest: dict, dest_uri: str, client, *, overwrite: bool = False) -> int:
    """Upload `items` and the manifest under `dest_uri`; returns the number of objects written.

    Refuses to touch a prefix that already contains objects unless `overwrite` is set.
    """
    bucket, prefix = split_s3_uri(dest_uri)
    existing = client.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=1)
    if existing.get("KeyCount", 0) and not overwrite:
        raise FileExistsError(f"s3://{bucket}/{prefix}/ already contains objects; pick a new --run-id or pass --overwrite")
    for item in items:
        client.upload_file(str(item.source), bucket, f"{prefix}/{item.rel_key}")
    client.put_object(Bucket=bucket, Key=f"{prefix}/manifest.json", Body=json.dumps(manifest, indent=2).encode(),
                      ContentType="application/json")
    return len(items) + 1
