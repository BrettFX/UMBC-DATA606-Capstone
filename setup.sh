#!/usr/bin/env bash
# Creates (or updates) the conda environment defined in environment.yml and
# registers it as a Jupyter kernel.
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

# `conda env create` fails (or prompts to overwrite) when the env already exists,
# so update in place instead; --prune drops packages no longer in the file.
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    echo "Environment '$CONDA_ENV_NAME' exists; updating..."
    conda env update -n "$CONDA_ENV_NAME" -f "$REPO_ROOT/environment.yml" --prune
else
    echo "Creating environment '$CONDA_ENV_NAME'..."
    conda env create -f "$REPO_ROOT/environment.yml"
fi

# Register the env as a Jupyter kernel whose display name matches the notebooks'
# kernelspec, so it shows up in Jupyter/VS Code without manual setup.
conda run -n "$CONDA_ENV_NAME" python -m ipykernel install --user \
    --name "$CONDA_ENV_NAME" --display-name "$CONDA_ENV_NAME"

echo "Done. Activate with: conda activate $CONDA_ENV_NAME"
