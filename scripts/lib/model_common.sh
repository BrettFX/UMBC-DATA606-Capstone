# Shared by download-models.sh and upload-models.sh (sourced, not run directly).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The models that count as "the best/latest": group | task | name in S3 | local model directory | sub-folders to upload
# (blank = the usual files). When a better model arrives, change its row here (and re-run upload-models.sh).
#   asr      the full-precision merged Whisper model (GPU, or the source for conversion)  ~1.5 GB
#   asr-ct2  the same model converted to CTranslate2 int8, all a CPU device needs        ~0.8 GB
#   ner      the spaCy NER model                                                          ~4 MB
SHIP_MODELS=(
    "asr|asr|lora-whisper-medium-en|whisper-medium-en-atc-finetuned-full-lora|"
    "asr-ct2|asr|lora-whisper-medium-en-ct2-int8|whisper-medium-en-atc-finetuned-full-lora|ct2-int8"
    "ner|ner|spacy-balanced|spacy-balanced|"
)
# Device profiles for downloading: which groups a device needs.
PROFILE_CPU="asr-ct2 ner"   # no GPU: the CTranslate2 model (no torch needed) + NER
PROFILE_GPU="asr ner"       # GPU: the full-precision model + NER

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

# Echo the SHIP_MODELS rows for group $1 (--only) or profile $2 (cpu|gpu); both blank = every model. Fails if none match.
select_models() {
    local only="$1" profile="$2" row found=0 wanted=""
    case "$profile" in
        "") ;;
        cpu) wanted="$PROFILE_CPU" ;;
        gpu) wanted="$PROFILE_GPU" ;;
        *) echo "Unknown profile '$profile' (expected cpu or gpu)." >&2; return 1 ;;
    esac
    [[ -n "$only" && -n "$profile" ]] && { echo "Use --only or --profile, not both." >&2; return 1; }
    for row in "${SHIP_MODELS[@]}"; do
        local group="${row%%|*}"
        if [[ -n "$only" && "$group" != "$only" ]]; then continue; fi
        if [[ -n "$wanted" && " $wanted " != *" $group "* ]]; then continue; fi
        echo "$row"; found=1
    done
    [[ "$found" -eq 1 ]] || { echo "No model matches '${only:-$profile}' (groups: asr, asr-ct2, ner)." >&2; return 1; }
}

# Arguments common to every model_store.py call.
store_base_args() {
    [[ -n "$MODEL_STORE_BASE" ]] && printf '%s\n' --base "$MODEL_STORE_BASE"
    return 0
}
