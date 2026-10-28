#!/usr/bin/env bash
# Creates (or updates) the conda environment defined in environment.yml and
# registers it as a Jupyter kernel. Also installs system packages the app needs (espeak-ng, ffmpeg).
# Safe to re-run: an existing environment is updated (never pruned), a broken pyzmq/ipykernel (left by an earlier pip install) is repaired,
# and failing to register the Jupyter kernel only warns, since the app does not need it.
#
# Usage:
#   ./setup.sh
#
# Override the env name with SETUP_CONDA_ENV if you changed `name:` in environment.yml.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${SETUP_CONDA_ENV:-data-science}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH. Install Anaconda/Miniconda first." >&2
    exit 1
fi

# System packages the app uses: espeak-ng (Kokoro voice read-back) and ffmpeg (decoding m4a/mp3 uploads).
# Installed via apt when missing; elsewhere, or if that fails, we only warn, since the app works without them.
ensure_system_dep() {  # $1 = command, $2 = package name, $3 = what needs it
    if command -v "$1" >/dev/null 2>&1; then
        echo "$2 found."
        return
    fi
    if command -v apt-get >/dev/null 2>&1; then
        echo "$2 not found; installing via apt (may prompt for sudo)..."
        local sudo=""
        [ "$(id -u)" -ne 0 ] && sudo="sudo"
        if $sudo apt-get install -y "$2"; then
            return
        fi
        echo "WARNING: could not install $2; $3 may not work." >&2
    else
        echo "WARNING: $2 not found and apt-get is unavailable. Install it with your package manager" \
             "(e.g. 'brew install $2') for $3." >&2
    fi
}
ensure_system_deps() {
    ensure_system_dep espeak-ng espeak-ng "the app's voice read-back"
    ensure_system_dep ffmpeg ffmpeg "decoding m4a and mp3 uploads in the app"
}
ensure_system_deps

# `conda env create` fails (or prompts to overwrite) when the env already exists, so update in place instead.
# No --prune on purpose: the env may be an existing one that holds other work (CUDA toolkit, Qt, lxml, ...), and --prune
# silently uninstalls every conda package that environment.yml does not list. The cost is that packages dropped from
# environment.yml linger; to clean those up, remove the env (`conda env remove -n <name>`) and re-run this script.
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    echo "Environment '$CONDA_ENV_NAME' exists; updating..."
    conda env update -n "$CONDA_ENV_NAME" -f "$REPO_ROOT/environment.yml"
else
    echo "Creating environment '$CONDA_ENV_NAME'..."
    conda env create -f "$REPO_ROOT/environment.yml"
fi

# Register the env as a Jupyter kernel whose display name matches the notebooks'
# kernelspec, so it shows up in Jupyter/VS Code without manual setup. Optional for the app, so a failure only warns.
# shellcheck source=scripts/lib/jupyter_env.sh
source "$REPO_ROOT/scripts/lib/jupyter_env.sh"
kernel_ok=1
register_jupyter_kernel "$CONDA_ENV_NAME" || kernel_ok=0

echo "Done. Activate with: conda activate $CONDA_ENV_NAME"
[[ "$kernel_ok" -eq 1 ]] || echo "(The Jupyter kernel was not registered; see the warning above.)"
