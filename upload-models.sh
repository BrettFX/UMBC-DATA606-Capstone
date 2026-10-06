#!/usr/bin/env bash
# Upload the current best ASR and NER models (and the CTranslate2 copy of the ASR model) to S3 as new runs and make them `latest`.
#
# Each model goes to s3://<bucket>/ml-tasks/<task>/<model-name>/<timestamp>/ and is then copied to .../latest/,
# which is what inference targets. A model whose files already match `latest` is skipped, so re-running this
# never creates duplicate runs. It shows what it will upload and asks before sending anything (unless --yes).
# Which models count as "the best" is listed in scripts/lib/model_common.sh.
#
# Usage:
#   ./upload-models.sh                      # show the plan, ask, then upload ASR + NER
#   ./upload-models.sh --only asr-ct2       # one group: asr, asr-ct2 or ner
#   ./upload-models.sh --dry-run            # show the plan only
#   ./upload-models.sh --yes                # no confirmation prompt (for scripts)
#   ./upload-models.sh --no-latest          # upload the run but leave `latest` where it is
# Needs AWS credentials (environment or ~/.aws) with write access to the bucket.

set -euo pipefail
# shellcheck source=scripts/lib/model_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/lib/model_common.sh"

only="" run_id="" yes=0 dry=0 latest=1
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) only="${2:?--only needs asr, asr-ct2 or ner}"; shift 2 ;;
        --run-id) run_id="${2:?--run-id needs a value}"; shift 2 ;;
        --yes) yes=1; shift ;;
        --dry-run) dry=1; shift ;;
        --no-latest) latest=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

models="$(select_models "$only" "")" || exit 2
if [[ -n "$run_id" && "$(wc -l <<<"$models")" -gt 1 ]]; then
    echo "--run-id needs --only (a run id belongs to one model)." >&2; exit 2
fi

activate_conda_env
mapfile -t base_args < <(store_base_args)
store() { python3 "$REPO_ROOT/scripts/model_store.py" ${base_args[@]+"${base_args[@]}"} "$@"; }

failed=0
while IFS='|' read -r group task name variant parts; do
    echo
    echo "--- Uploading $group: $task/$name (local: $variant${parts:+/$parts}) ---"
    args=(upload --task "$task" --model-name "$name" --variants "$variant" --skip-if-unchanged)
    [[ -n "$parts" ]] && args+=(--parts "$parts")
    [[ -n "$run_id" ]] && args+=(--run-id "$run_id")
    [[ "$latest" -eq 1 ]] && args+=(--set-latest)

    plan="$(store "${args[@]}" 2>&1)" || { echo "$plan" >&2; echo "FAILED: $task/$name" >&2; failed=1; continue; }
    if [[ "$dry" -eq 1 ]]; then echo "$plan" | cut -c1-200; else echo "$plan" | grep -v "dry run: nothing uploaded" | cut -c1-200; fi
    if grep -q "up to date" <<<"$plan"; then continue; fi        # nothing to do for this model
    [[ "$dry" -eq 1 ]] && continue
    if [[ "$yes" -ne 1 ]]; then
        read -r -p "Upload $task/$name as shown above? [y/N]: " reply || reply="n"
        [[ "$reply" =~ ^[Yy]$ ]] || { echo "Skipped $task/$name."; continue; }
    fi
    store "${args[@]}" --yes || { echo "FAILED: $task/$name" >&2; failed=1; }
done <<<"$models"

echo
if [[ "$failed" -eq 0 ]]; then echo "=== Done ==="; else echo "=== Finished with errors (see above) ===" >&2; fi
exit "$failed"
