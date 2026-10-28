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
# S3 needs AWS credentials (environment, ~/.aws or SSO) that can read the bucket. The default (--source auto) probes that
# first and quietly falls back to the Google Drive bundle if S3 cannot be used for ANY reason (no credentials, a different
# AWS account, an expired SSO login, no network, boto3 missing) or a download fails part-way. The bundle is enough to run
# the Streamlit app but not the notebooks or training. An error is shown only when --source s3 is given and S3 fails, or
# when every download method fails. --source s3 never falls back.
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

activate_conda_env
mapfile -t base_args < <(store_base_args)

# The Google Drive bundle: only what the app loads. Prints its own error (and manual-download hint) on failure.
fetch_from_drive() { command -v python3 >/dev/null 2>&1 && python3 "$REPO_ROOT/scripts/app_bundle.py" fetch; }

auto=0
if [[ "$source" == "auto" ]]; then
    auto=1
    # Having credentials is not enough (a different AWS account or an expired SSO login both have them), so check read access.
    if reason="$(python3 "$REPO_ROOT/scripts/model_store.py" ${base_args[@]+"${base_args[@]}"} probe 2>/dev/null)"; then
        source="s3"
    else
        echo "S3 is not available (${reason:-could not check access}); using the Google Drive bundle (app models only)."
        source="gdrive"
    fi
fi
if [[ "$source" == "gdrive" ]]; then
    [[ "$dry" -eq 1 ]] && { echo "Would download clearanceiq-models.zip from Google Drive and extract it into models/."; exit 0; }
    if fetch_from_drive; then exit 0; fi
    [[ "$auto" -eq 1 ]] && echo "ERROR: could not download the models from S3 or Google Drive; see the messages above." >&2
    exit 1
fi

models="$(select_models "$only" "$profile")" || exit 2
if [[ -n "$run_id" && "$(wc -l <<<"$models")" -gt 1 ]]; then
    echo "--run-id needs --only (a run id belongs to one model)." >&2; exit 2
fi

err_log="$(mktemp)"; trap 'rm -f "$err_log"' EXIT
failed=0; failed_groups=()
while IFS='|' read -r group task name _; do
    echo
    echo "--- Downloading $group: $task/$name (${run_id:-latest}) ---"
    args=(download --task "$task" --model-name "$name")
    [[ -n "$run_id" ]] && args+=(--run-id "$run_id")
    [[ -n "$dest_root" ]] && args+=(--dest-root "$dest_root")
    [[ "$force" -eq 1 ]] && args+=(--force)
    [[ "$dry" -eq 1 ]] && args+=(--dry-run)
    if [[ "$auto" -eq 1 ]]; then
        # Hold the output back so a failure that triggers the Drive fallback is not shown as an error.
        grp_log="$(mktemp)"
        if python3 "$REPO_ROOT/scripts/model_store.py" ${base_args[@]+"${base_args[@]}"} "${args[@]}" 2>"$grp_log"; then
            cat "$grp_log" >&2
        else
            { echo "--- $task/$name ---"; cat "$grp_log"; } >>"$err_log"
            failed=1; failed_groups+=("$group")
        fi
        rm -f "$grp_log"
    elif ! python3 "$REPO_ROOT/scripts/model_store.py" ${base_args[@]+"${base_args[@]}"} "${args[@]}"; then
        echo "FAILED: $task/$name" >&2; failed=1
    fi
done <<<"$models"

echo
if [[ "$failed" -eq 0 ]]; then echo "=== Done ==="; exit 0; fi
if [[ "$auto" -eq 0 ]]; then echo "=== Finished with errors (see above) ===" >&2; exit 1; fi

echo "Note: the S3 download did not complete (not downloaded: ${failed_groups[*]}); using the Google Drive bundle instead."
[[ "$dry" -eq 1 ]] && { echo "Would download clearanceiq-models.zip from Google Drive and extract it into models/."; exit 0; }
if fetch_from_drive; then
    echo "=== Done: app models from Google Drive. The notebooks and training also need S3 access (not downloaded: ${failed_groups[*]}). ==="
    exit 0
fi
echo "ERROR: all download methods failed. The S3 errors were:" >&2
cat "$err_log" >&2
exit 1
