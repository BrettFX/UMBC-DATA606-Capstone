#!/usr/bin/env bash
# Download the trained models into this repo, so a new device is ready in one command.
#
# Default (--source gdrive): the production app models, from a public view-only Google Drive folder: the CTranslate2 int8
# ASR model and both spaCy NER models (~0.7 GB). No AWS account, GPU or PyTorch needed. The zip is verified against
# app-models.sha256 (committed in this repo) before anything is extracted, and a bundle that is already installed is skipped.
# ./setup.sh runs this for you.
#
# Optional (--source s3): the experiment store, kept for experiment tracking and traceability and needed for training and for
# re-running the notebooks' saved-results sections: every run of every model, the full-precision ASR model, the results CSVs and
# the NER data. It needs AWS credentials AND the store's location in MODEL_STORE_BASE (s3://<bucket>/ml-tasks); the original
# store is private. Files land in the paths the training code uses (models/..., data/processed/...), checksums are verified
# against each run's manifest, files that already match are skipped, and a differing local file is never overwritten without --force.
# Which models count as "the best" is listed in scripts/lib/model_common.sh.
#
# Usage:
#   ./download-models.sh                    # production app models from Google Drive
#   ./download-models.sh --force            # download and extract them again even if installed
#   ./download-models.sh --dry-run          # show what would happen, write nothing
#
#   export MODEL_STORE_BASE=s3://<bucket>/ml-tasks
#   ./download-models.sh --source s3                      # everything: full ASR model, CTranslate2 ASR model, NER, results
#   ./download-models.sh --source s3 --profile cpu        # a device without a GPU: CTranslate2 ASR model + NER + results
#   ./download-models.sh --source s3 --profile gpu        # a device with a GPU: full ASR model + NER + results
#   ./download-models.sh --source s3 --only ner           # one group: asr, asr-ct2, asr-small-8ep, asr-results, ner-data, ner or ner-variants
#   ./download-models.sh --source s3 --only asr --run-id 20261006-112237     # a specific run instead of latest
# The `asr-results` group (included in every S3 profile) restores the ASR results CSVs into data/processed/asr_results/, and `ner-data`
# the NER annotations and curated dataset.

set -euo pipefail
# shellcheck source=scripts/lib/model_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/lib/model_common.sh"

only="" profile="" run_id="" dest_root="" force=0 dry=0 source="gdrive"
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) only="${2:?--only needs asr, asr-ct2, asr-small-8ep, asr-results, ner-data, ner or ner-variants}"; shift 2 ;;
        --profile) profile="${2:?--profile needs cpu or gpu}"; shift 2 ;;
        --run-id) run_id="${2:?--run-id needs a value}"; shift 2 ;;
        --dest-root) dest_root="${2:?--dest-root needs a directory}"; shift 2 ;;
        --source) source="${2:?--source needs gdrive or s3}"; shift 2 ;;
        --force) force=1; shift ;;
        --dry-run) dry=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$source" in gdrive|s3) ;; *) echo "Unknown --source '$source' (expected gdrive or s3)." >&2; exit 2 ;; esac

if [[ "$source" == "gdrive" ]]; then
    # The Drive bundle is the CPU profile (CTranslate2 ASR model + spaCy NER); other groups and runs only exist in the S3 store.
    if [[ -n "$run_id" || -n "$dest_root" || ( -n "$only" && "$only" != "asr-ct2" && "$only" != "ner" ) || ( -n "$profile" && "$profile" != "cpu" ) ]]; then
        echo "These options pick other model groups or runs, which only exist in the optional S3 experiment store." >&2
        echo "Add --source s3 (and set MODEL_STORE_BASE), or drop them to get the production models from Google Drive." >&2
        exit 2
    fi
    [[ "$dry" -eq 1 ]] && { echo "Would download clearanceiq-models.zip from Google Drive and extract it into models/ (skipped if already installed)."; exit 0; }
    activate_conda_env
    args=(fetch); [[ "$force" -eq 1 ]] && args+=(--force)
    exec python3 "$REPO_ROOT/scripts/app_bundle.py" "${args[@]}"
fi

# --source s3: the optional experiment store
require_store_base || exit 2
models="$(select_models "$only" "$profile")" || exit 2
if [[ -n "$run_id" && "$(wc -l <<<"$models")" -gt 1 ]]; then
    echo "--run-id needs --only (a run id belongs to one model)." >&2; exit 2
fi

activate_conda_env
mapfile -t base_args < <(store_base_args)
failed=0
while IFS='|' read -r group task name _; do
    echo
    echo "--- Downloading $group: $task/$name (${run_id:-latest}) ---"
    args=(download --task "$task" --model-name "$name")
    [[ -n "$run_id" ]] && args+=(--run-id "$run_id")
    [[ -n "$dest_root" ]] && args+=(--dest-root "$dest_root")
    [[ "$force" -eq 1 ]] && args+=(--force)
    [[ "$dry" -eq 1 ]] && args+=(--dry-run)
    if ! python3 "$REPO_ROOT/scripts/model_store.py" ${base_args[@]+"${base_args[@]}"} "${args[@]}"; then
        echo "FAILED: $task/$name" >&2; failed=1
    fi
done <<<"$models"

echo
if [[ "$failed" -eq 0 ]]; then echo "=== Done ==="; else echo "=== Finished with errors (see above) ===" >&2; fi
exit "$failed"
