#!/usr/bin/env bash
# Driver script for the capstone's ML pipeline. Interactively asks, per
# stage (data ingest, training, inference, ...), whether to run it and what
# arguments to pass to its underlying script, then invokes it.
#
# Only the data ingest stage is implemented so far (scripts/run_ingest.py);
# the others are stubbed here so the prompt flow doesn't need to change
# shape once they exist -- adding one is just filling in its run_* function.
#
# Usage:
#   ./run-pipeline.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set by run_data_ingest() and re-used by run_purge_raw() so both call the
# same data-dir and log-level without asking the user a second time.
PIPELINE_DATA_DIR=""
PIPELINE_LOG_LEVEL="INFO"

# The conda environment this project's dependencies (librosa, torch,
# transformers, jiwer, ...) are installed into -- matches the notebooks'
# kernelspec `display_name`. Override if you use a different env name.
CONDA_ENV_NAME="${PIPELINE_CONDA_ENV:-data-science}"

# Best-effort: activate CONDA_ENV_NAME so every stage below gets the right
# interpreter/packages regardless of what environment this script happened
# to be launched from (a plain `python3` on PATH depends on the invoking
# shell already having the right env active, which this driver shouldn't
# assume). Falls back to whatever `python3` is already on PATH if conda
# isn't installed or the named env doesn't exist, rather than failing.
activate_conda_env() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "Note: conda not found on PATH; using current environment's python3."
        return 0
    fi
    local conda_base
    conda_base="$(conda info --base 2>/dev/null)" || {
        echo "Note: couldn't determine conda's base install; using current environment's python3."
        return 0
    }
    if [[ ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
        return 0
    fi

    # conda's own activation scripts reference variables (e.g. $PS1) that
    # don't exist in a non-interactive shell; relax -u around them rather
    # than have this driver abort on conda's internals.
    set +u
    # shellcheck disable=SC1091
    source "$conda_base/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
        conda activate "$CONDA_ENV_NAME"
        set -u
        echo "Activated conda environment: $CONDA_ENV_NAME"
    else
        set -u
        echo "Note: conda environment '$CONDA_ENV_NAME' not found; using current environment's python3."
    fi
}

# Ask a yes/no question. $1 = question text, $2 = default ("y" or "n").
# Falls back to the default if stdin is exhausted (e.g. piped input runs
# out) instead of aborting the whole script under `set -e`.
prompt_yes_no() {
    local question="$1" default="${2:-n}" hint="y/N" reply
    [[ "$default" == "y" ]] && hint="Y/n"
    read -r -p "$question [$hint]: " reply || reply="$default"
    reply="${reply:-$default}"
    [[ "$reply" =~ ^[Yy]$ ]]
}

# Ask for a free-text value. $1 = question text, $2 = default (may be empty).
# Echoes the answer (or the default) to stdout.
prompt_value() {
    local question="$1" default="$2" reply
    if [[ -n "$default" ]]; then
        read -r -p "$question (default: $default): " reply || reply="$default"
    else
        read -r -p "$question: " reply || reply=""
    fi
    echo "${reply:-$default}"
}

run_data_ingest() {
    echo
    echo "--- Stage: Data Ingest ---"
    if ! prompt_yes_no "Run the data ingest pipeline?" "y"; then
        echo "Skipping data ingest."
        return 0
    fi

    local data_dir num_proc log_level force=0
    data_dir=$(prompt_value "Data directory (blank = repo default: ${REPO_ROOT}/data)" "")
    prompt_yes_no "Force recompute even if cached processed output exists?" "n" && force=1
    num_proc=$(prompt_value "Number of worker processes for feature derivation (blank = single-process)" "")
    log_level=$(prompt_value "Log level (DEBUG/INFO/WARNING/ERROR)" "INFO")

    # Persist for use by run_purge_raw() after all stages complete.
    PIPELINE_DATA_DIR="$data_dir"
    PIPELINE_LOG_LEVEL="$log_level"

    local args=()
    [[ -n "$data_dir" ]] && args+=(--data-dir "$data_dir")
    [[ "$force" -eq 1 ]] && args+=(--force)
    [[ -n "$num_proc" ]] && args+=(--num-proc "$num_proc")
    args+=(--log-level "$log_level")

    echo
    echo "Using python3: $(command -v python3)"
    echo "Running: python3 scripts/run_ingest.py ${args[*]}"
    python3 "$REPO_ROOT/scripts/run_ingest.py" "${args[@]}"
}

# Purge raw HuggingFace download caches via --purge-only (no pipeline re-run).
# Called only after all pipeline stages complete so training/inference can still
# read the WAV files during their runs.
run_purge_raw() {
    echo
    echo "--- Stage: Cleanup ---"
    if ! prompt_yes_no "Purge raw data downloads to reclaim disk space?" "n"; then
        echo "Skipping purge."
        return 0
    fi

    local args=(--purge-only --log-level "$PIPELINE_LOG_LEVEL")
    [[ -n "$PIPELINE_DATA_DIR" ]] && args+=(--data-dir "$PIPELINE_DATA_DIR")

    echo
    echo "Running: python3 scripts/run_ingest.py ${args[*]}"
    python3 "$REPO_ROOT/scripts/run_ingest.py" "${args[@]}"
}

# TODO: implement the training stage. For now, just prompt and skip.
run_training() {
    echo
    echo "--- Stage: Model Training ---"
    if ! prompt_yes_no "Run the training pipeline? (not yet implemented)" "n"; then
        echo "Skipping training."
        return 0
    fi
    echo "Training pipeline isn't implemented yet -- nothing to run. Skipping."
}

# TODO: implement the inference stage. For now, just prompt and skip.
run_inference() {
    echo
    echo "--- Stage: Inference ---"
    if ! prompt_yes_no "Run the inference pipeline? (not yet implemented)" "n"; then
        echo "Skipping inference."
        return 0
    fi
    echo "Inference pipeline isn't implemented yet -- nothing to run. Skipping."
}

main() {
    echo "=== ATC Capstone Pipeline Driver ==="
    activate_conda_env
    run_data_ingest
    run_training
    run_inference
    run_purge_raw
    echo
    echo "=== Pipeline driver complete ==="
}

main "$@"
