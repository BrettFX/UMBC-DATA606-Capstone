"""Collect trained-model artifacts and move them to and from S3 under versioned, labeled prefixes.

Layout: s3://<bucket>/<base>/<task>/<model-name>/<run-id>/<files...> plus a manifest.json (checksums, run id,
git commit) at the run level, and a `latest/` copy of whichever run inference should use. A fresh run-id per
upload means a new model never silently replaces an earlier one; `upload` refuses a prefix that already holds
objects, and `latest/` changes only through an explicit promote. `download` restores files to the same local
paths the repo uses (models/, models/ner/, data/processed/...), so a new device needs no configuration.
Optimizer checkpoints are never included.
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


def build_manifest(task: str, run_id: str, items: list[Item], repo_root: Path, model_name: str | None = None) -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {"task": task, "model_name": model_name, "run_id": run_id, "git_commit": commit, "created_utc": datetime.now(timezone.utc).isoformat(),
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


DEFAULT_BASE = "s3://endurasoft-dev-ml-ops/ml-tasks"


_MODEL_PARTS = ("/final/", "/final-adapter/", "/model-best/")


def unchanged_vs_latest(client, latest_uri: str, manifest: dict) -> str | None:
    """The run id `latest/` already holds if its model files match `manifest`'s checksums, else None.

    Only the model weights and configs are compared (keys under final/, final-adapter/ or model-best/), so
    regenerating a metrics table or results file does not count as a new model.
    """
    bucket, prefix = split_s3_uri(latest_uri)
    try:
        remote = json.loads(client.get_object(Bucket=bucket, Key=f"{prefix}/manifest.json")["Body"].read())
    except Exception:  # no latest yet (or no manifest): treat as changed
        return None
    pick = lambda m: {f["key"]: f["sha256"] for f in m["files"] if any(part in "/" + f["key"] for part in _MODEL_PARTS)}
    local_files, remote_files = pick(manifest), pick(remote)
    return remote.get("run_id") if local_files and local_files == remote_files else None


def model_prefix(base: str, task: str, model_name: str) -> str:
    """`s3://bucket/ml-tasks` + task + model name -> the S3 URI that holds this model's runs and `latest/`."""
    return f"{base.rstrip('/')}/{task}/{model_name}"


def list_keys(client, bucket: str, prefix: str) -> dict[str, int]:
    """Every object under `prefix` as {key relative to prefix: size}, following pagination."""
    out, token = {}, None
    while True:
        page = client.list_objects_v2(Bucket=bucket, Prefix=prefix, **({"ContinuationToken": token} if token else {}))
        out.update({o["Key"][len(prefix):]: o["Size"] for o in page.get("Contents", [])})
        if not page.get("IsTruncated"):
            return out
        token = page["NextContinuationToken"]


def common_prefixes(client, bucket: str, prefix: str) -> list[str]:
    """The immediate sub-folder names under `prefix` (which should end with '/')."""
    out, token = [], None
    while True:
        page = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/", **({"ContinuationToken": token} if token else {}))
        out += [p["Prefix"][len(prefix):].rstrip("/") for p in page.get("CommonPrefixes", [])]
        if not page.get("IsTruncated"):
            return out
        token = page["NextContinuationToken"]


def copy_prefix(client, bucket: str, src: str, dst: str, *, replace: bool = False) -> int:
    """Server-side copy of every object under `src` to `dst`; with `replace`, stale objects already under `dst`
    are deleted first so the destination ends up identical to the source. Returns the objects copied."""
    keys = list_keys(client, bucket, src + "/")
    if not keys:
        raise FileNotFoundError(f"nothing under s3://{bucket}/{src}/")
    if replace:
        stale = list(list_keys(client, bucket, dst + "/"))
        for lo in range(0, len(stale), 1000):
            client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": f"{dst}/{k}"} for k in stale[lo:lo + 1000]]})
    for rel in keys:
        client.copy({"Bucket": bucket, "Key": f"{src}/{rel}"}, bucket, f"{dst}/{rel}")
    return len(keys)


def local_path(task: str, rel_key: str, repo_root: Path) -> Path:
    """Where a downloaded object lives locally, matching the repo's own paths for that task."""
    if task == "asr":
        if rel_key.startswith("test_results/"):
            return repo_root / "data/processed/asr_results" / rel_key.removeprefix("test_results/")
        return repo_root / "models" / rel_key
    if task == "ner":
        if rel_key.startswith("data/"):
            return repo_root / "data/processed/ner_dataset" / rel_key.removeprefix("data/")
        return repo_root / "models/ner" / rel_key
    raise ValueError(f"unknown task {task!r}")


def download(client, prefix_uri: str, task: str, repo_root: Path, *, force: bool = False, dry_run: bool = False) -> list[dict]:
    """Download one run (or `latest/`) into the repo's local layout and verify checksums against its manifest.

    Files that already match are skipped. A local file that differs is a conflict and raises unless `force`.
    Returns the plan: one dict per file with `key`, `path`, `size` and `status` (new / unchanged / conflict).
    """
    bucket, prefix = split_s3_uri(prefix_uri)
    keys = list_keys(client, bucket, prefix + "/")
    if not keys:
        raise FileNotFoundError(f"nothing under {prefix_uri}/")
    manifest = json.loads(client.get_object(Bucket=bucket, Key=f"{prefix}/manifest.json")["Body"].read()) if "manifest.json" in keys else None
    expected = {f["key"]: f["sha256"] for f in (manifest or {}).get("files", [])}
    plan = []
    for rel, size in keys.items():
        if rel == "manifest.json":
            continue
        path = local_path(task, rel, repo_root)
        status = "new"
        if path.exists():
            same = path.stat().st_size == size and (rel not in expected or _sha256(path) == expected[rel])
            status = "unchanged" if same else "conflict"
        plan.append({"key": rel, "path": path, "size": size, "status": status})
    conflicts = [p for p in plan if p["status"] == "conflict"]
    if conflicts and not force:
        raise FileExistsError(f"{len(conflicts)} local file(s) differ from the download, e.g. {conflicts[0]['path']}; pass --force to overwrite")
    if dry_run:
        return plan
    for p in plan:
        if p["status"] == "unchanged":
            continue
        p["path"].parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, f"{prefix}/{p['key']}", str(p["path"]))
        if p["key"] in expected and _sha256(p["path"]) != expected[p["key"]]:
            raise OSError(f"checksum mismatch after downloading {p['key']}")
    return plan
