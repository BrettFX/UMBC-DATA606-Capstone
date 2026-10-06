# Shared by download-models.sh and upload-models.sh (sourced, not run directly).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The models that count as "the best/latest": task | name in S3 | local model directory (what upload picks up).
# When a better model arrives, change its row here (and re-run upload-models.sh).
SHIP_MODELS=(
    "asr|lora-whisper-medium-en|whisper-medium-en-atc-finetuned-full-lora"
    "ner|spacy-balanced|spacy-balanced"
)

# The conda environment the project's dependencies (boto3, spaCy, transformers, ...) live in. Override if yours differs.
CONDA_ENV_NAME="${PIPELINE_CONDA_ENV:-data-science}"

# Optional override of the S3 location, e.g. MODEL_STORE_BASE=s3://other-bucket/ml-tasks
MODEL_STORE_BASE="${MODEL_STORE_BASE:-}"

# Best-effort conda activation, same approach as run-pipeline.sh: fall back to whatever python3 is on PATH.
activate_conda_env() {
    command -v conda >/dev/null 2>&1 || { echo "Note: conda not found; using the current python3."; return 0; }
    local conda_base
    conda_base="$(conda info --base 2>/dev/null)" || return 0
    [[ -f "$conda_base/etc/profile.d/conda.sh" ]] || return 0
    set +u   # conda's activation scripts reference variables that are unset in a non-interactive shell
    # shellcheck disable=SC1091
    source "$conda_base/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
        conda activate "$CONDA_ENV_NAME"
        echo "Activated conda environment: $CONDA_ENV_NAME"
    else
        echo "Note: conda environment '$CONDA_ENV_NAME' not found; using the current python3."
    fi
    set -u
}

# Echo the SHIP_MODELS rows whose task matches $1 (blank = all); fail if none match.
select_models() {
    local only="$1" row found=0
    for row in "${SHIP_MODELS[@]}"; do
        if [[ -z "$only" || "${row%%|*}" == "$only" ]]; then echo "$row"; found=1; fi
    done
    [[ "$found" -eq 1 ]] || { echo "No model configured for task '$only' (expected asr or ner)." >&2; return 1; }
}

# Arguments common to every model_store.py call.
store_base_args() {
    [[ -n "$MODEL_STORE_BASE" ]] && printf '%s\n' --base "$MODEL_STORE_BASE"
    return 0
}
