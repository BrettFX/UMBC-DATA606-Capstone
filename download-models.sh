#!/usr/bin/env bash
# Download the latest ASR and NER models from S3 into this repo, so a new device is ready in one command.
# A CPU-only device needs just the small CTranslate2 model (no torch required); use --profile cpu for it.
#
# Files land in the same local paths the training code uses (models/..., models/ner/..., data/processed/...),
# checksums are verified against each run's manifest, files that already match are skipped, and a local file
# that differs is never overwritten unless you pass --force. Which models count as "the best" is listed in
# scripts/lib/model_common.sh.
#
# Usage:
#   ./download-models.sh                    # everything: full ASR model, CTranslate2 ASR model, NER
#   ./download-models.sh --profile cpu      # a device without a GPU: CTranslate2 ASR model + NER (~0.8 GB)
#   ./download-models.sh --profile gpu      # a device with a GPU: full ASR model + NER (~1.5 GB)
#   ./download-models.sh --only ner         # one group: asr, asr-ct2, asr-small-8ep, asr-results, ner-data, ner or ner-variants
#   ./download-models.sh --dry-run          # show what would be downloaded, write nothing
#   ./download-models.sh --only asr --run-id 20261006-112237     # a specific run instead of latest
#   ./download-models.sh --force            # overwrite local files that differ
# The `asr-results` group (included in every profile) restores the ASR results CSVs into data/processed/asr_results/, and `ner-data` the NER annotations and curated dataset.
#   ./download-models.sh --source gdrive    # no S3 access: the app-only bundle (CPU ASR model + spaCy NER, ~0.8 GB) from Google Drive
# Needs AWS credentials (environment or ~/.aws) with read access to the bucket. Without them, the default (--source auto)
# falls back to the Google Drive bundle, which is enough to run the Streamlit app but not the notebooks or training.
# The Drive zip is verified against app-models.sha256 (committed in this repo) before anything is extracted.

set -euo pipefail
# shellcheck source=scripts/lib/model_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/lib/model_common.sh"

only="" profile="" run_id="" dest_root="" force=0 dry=0 source="auto"
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) only="${2:?--only needs asr, asr-ct2, asr-small-8ep, asr-results, ner-data, ner or ner-variants}"; shift 2 ;;
        --profile) profile="${2:?--profile needs cpu or gpu}"; shift 2 ;;
        --run-id) run_id="${2:?--run-id needs a value}"; shift 2 ;;
        --dest-root) dest_root="${2:?--dest-root needs a directory}"; shift 2 ;;
        --source) source="${2:?--source needs auto, s3 or gdrive}"; shift 2 ;;
        --force) force=1; shift ;;
        --dry-run) dry=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$source" in auto|s3|gdrive) ;; *) echo "Unknown --source '$source' (expected auto, s3 or gdrive)." >&2; exit 2 ;; esac

if [[ "$source" == "auto" ]]; then
    activate_conda_env
    if python3 -c 'import boto3, sys; sys.exit(0 if boto3.Session().get_credentials() else 1)' 2>/dev/null; then
        source="s3"
    else
        echo "No AWS credentials found; using the Google Drive bundle (app models only)."
        source="gdrive"
    fi
fi
if [[ "$source" == "gdrive" ]]; then
    [[ "$dry" -eq 1 ]] && { echo "Would download clearanceiq-models.zip from Google Drive and extract it into models/."; exit 0; }
    activate_conda_env
    exec python3 "$REPO_ROOT/scripts/app_bundle.py" fetch
fi

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
