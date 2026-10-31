#!/usr/bin/env python3
"""Check the pinned Hugging Face revisions in src/hf_revisions.py against the Hub (needs network).

For every pinned repo it confirms that the pinned commit still exists, and reports whether the default branch has moved on
(informational: a pin is meant to stay put). Exit status is 1 if any pinned commit can no longer be found.

    python scripts/check_hf_revisions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hf_revisions  # noqa: E402


def main() -> int:
    from huggingface_hub import HfApi

    api, missing = HfApi(), 0
    for kind, table in (("dataset", hf_revisions.DATASETS), ("model", hf_revisions.MODELS)):
        info = api.dataset_info if kind == "dataset" else api.model_info
        for repo, sha in table.items():
            try:
                info(repo, revision=sha)  # raises if the pinned commit is gone
                head = info(repo).sha
            except Exception as exc:  # noqa: BLE001 -- report every repo, whatever the failure
                missing += 1
                print(f"MISSING  {kind:7} {repo}  {sha[:12]}  ({type(exc).__name__})")
                continue
            note = "pinned commit is the default branch" if head == sha else f"default branch has moved on ({head[:12]})"
            print(f"ok       {kind:7} {repo}  {sha[:12]}  {note}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
