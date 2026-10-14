#!/usr/bin/env bash
# Launch ClearanceIQ from a clean shell: ./run-app.sh [streamlit args]
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$HOME/anaconda3/envs/data-science/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
"$PY" -c "import streamlit" 2>/dev/null || { echo "streamlit missing: $PY -m pip install streamlit plotly anthropic"; exit 1; }
exec "$PY" -m streamlit run app/main.py --server.headless true "$@"
