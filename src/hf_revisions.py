"""Pinned Hugging Face revisions for every dataset and model this project downloads.

A bare repo id means "whatever is on the default branch today", so a maintainer pushing a new commit to a dataset or a base model
could silently change the splits, the baselines or the labels. Each entry below is the exact commit the project's results were
produced with (the cache directories of the original runs name them), and a re-run in a clean clone reproduced the saved
results from these same commits. Pass them to the loaders with `pin()`:

    load_dataset(repo_id, cache_dir=..., **pin(repo_id))
    WhisperProcessor.from_pretrained(model_id, **pin(model_id))

A repo that is not listed (a local path, or a model you are experimenting with) is simply not pinned. To move a pin deliberately,
run `python scripts/check_hf_revisions.py` to see which repos have newer commits, re-run what depends on it, then edit the SHA here.
"""

from __future__ import annotations

DATASETS = {
    "jlvdoorn/atcosim": "b5839d940475b03bf14cbaf6db28a4ceaf62a2a9",           # simulated ATC (ATCOSIM mirror)
    "jacktol/ATC-ASR-Dataset": "075e736bf8aed80579d829092f74355486b10bc7",     # real ATC (UWB + ATCO2 test subset)
    "jlvdoorn/atco2-asr": "3b96f5791582cf6f6f348ca46fd502a981918d85",          # retired proposal-stage source (EDA notebooks only)
}

MODELS = {
    "openai/whisper-medium.en": "2e98eb6279edf5095af0c8dedb36bdec0acd172b",    # ASR baseline and the LoRA fine-tune's base
    "openai/whisper-small.en": "e8727524f962ee844a7319d92be39ac1bd25655a",     # fine-tuned small model and its zero-shot baseline
    "openai/whisper-tiny.en": "87c7102498dcde7456f24cfd30239ca606ed9063",      # unit-test fixture only
    "jacktol/whisper-medium.en-fine-tuned-for-ATC": "d373acbfa93d0a69f679d0dbe121177e0874b779",  # off-the-shelf ATC comparison model
    "RedHatAI/Qwen3.5-9B-quantized.w4a16": "a398088c4228b0ae0c8c78df88fd1e4bf445f068",  # LLM annotation teacher (production labels)
    "RedHatAI/Qwen3.5-4B-quantized.w4a16": "7a613872f394578b0b52b683ff4ac47516b4bcaf",  # faster annotation preset and prompt experiments
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",             # notebook NER prototype
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",             # notebook NER prototype (second model)
    "bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",           # BERT NER student
    "hexgrad/Kokoro-82M": "f3ff3571791e39611d31c381e3a41a3af07b4987",          # text-to-speech read-back in the app
}


def revision(repo_id: str | None) -> str | None:
    """The pinned commit SHA for a repo id, or None if it is not pinned (e.g. a local path)."""
    return DATASETS.get(repo_id) or MODELS.get(repo_id)


def pin(repo_id: str | None) -> dict[str, str]:
    """`{"revision": sha}` to splat into a Hugging Face loader, or `{}` for an unpinned repo."""
    sha = revision(repo_id)
    return {"revision": sha} if sha else {}


def snapshot(repo_id: str) -> str:
    """Local directory of the pinned snapshot of a model repo, downloading it if needed. For loaders that take a path but not a
    revision (vLLM)."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    if Path(repo_id).is_dir():  # already a local model directory
        return repo_id
    return snapshot_download(repo_id, **pin(repo_id))


def hub_file(repo_id: str, filename: str) -> str:
    """Local path of one file of the pinned revision of a repo, downloading it if needed. For libraries that call hf_hub_download
    themselves without a revision (Kokoro)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, **pin(repo_id))
