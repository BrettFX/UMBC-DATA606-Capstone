#!/usr/bin/env bash
# Build the single-zip model bundle that the Streamlit app needs (the Google Drive fallback to S3).
# Run this after retraining or promoting a model, then upload the zip to the Drive folder (replacing the old one)
# and commit the updated app-models.sha256.
#
# Usage:
#   ./package-app-models.sh                    # writes dist/clearanceiq-models.zip and updates app-models.sha256
#   ./package-app-models.sh --out /tmp/x.zip   # write the zip somewhere else
#   ./package-app-models.sh fetch              # the reverse: download from Drive, verify and extract into models/
#   ./package-app-models.sh fetch --zip FILE   # use a zip you already downloaded
# Any arguments are passed through to scripts/app_bundle.py (see its --help).

set -euo pipefail
# shellcheck source=scripts/lib/model_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/lib/model_common.sh"

case "${1:-}" in
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    fetch|package) cmd="$1"; shift ;;
    *) cmd="package" ;;
esac

activate_conda_env
python3 "$REPO_ROOT/scripts/app_bundle.py" "$cmd" "$@"

if [[ "$cmd" == "package" ]]; then
    echo
    echo "Next: upload the zip to the Drive folder (replace the old file), then commit app-models.sha256."
    echo "  https://drive.google.com/drive/folders/1F4tGwPjIgFPdo3ZClRX5-y3gFGThBWZ1"
fi
