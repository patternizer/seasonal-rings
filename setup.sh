#!/usr/bin/env bash
# permissions: chmod +x setup.sh
# run: ./setup.sh

set -Eeuo pipefail

# ---- preflight: environment.yml presence ----
if [[ ! -f "environment.yml" ]]; then
  echo "Error: environment.yml not found in the current directory."
  exit 1
fi

# ---- preflight: forbid tabs in YAML (YAML does not allow tabs) ----
if grep -n $'\t' environment.yml >/dev/null; then
  echo "Error: Tabs found in environment.yml (YAML forbids tabs). Lines:"
  grep -n $'\t' environment.yml
  echo
  echo "Fix them by replacing tabs with spaces, e.g.:"
  echo "  perl -i.bak -pe 's/\t/  /g' environment.yml"
  echo "…then re-run ./setup.sh"
  exit 1
fi

# ---- ensure conda is available & activate hook (works in Git Bash) ----
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found on PATH. Install Miniconda/Anaconda or open this in a conda-enabled shell."
  exit 1
fi

# Initialize conda for this shell session (portable)
eval "$(conda 'shell.bash' hook)"

# ---- determine env name ----
# Try to read `name:` from environment.yml; fallback to 'quantum-poetry'
ENV_NAME="$(awk -F: '/^[[:space:]]*name[[:space:]]*:/ {gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2); print $2; exit}' environment.yml || true)"
ENV_NAME="${ENV_NAME:-quantum-poetry}"

echo "Using conda environment name: ${ENV_NAME}"

# ---- create or update the environment idempotently ----
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Updating existing env '${ENV_NAME}'..."
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  echo "Creating env '${ENV_NAME}'..."
  conda env create -n "${ENV_NAME}" -f environment.yml
fi

# ---- activate and verify dependencies ----
conda activate "${ENV_NAME}"

python -V
python -c "import numpy, pandas, scipy, networkx, matplotlib, plotly, skimage, PIL; print('Deps OK')"

python - <<'PY'
import numpy, pandas, scipy, networkx, matplotlib, plotly, skimage, PIL, flask, dash
print("Dependencies imported successfully.")
PY

# ---- done ----
echo "Env ready. Run with:"
echo "  conda activate ${ENV_NAME} && python <your_script>.py"
echo "NOTE: Place your poem file at ./poem-v1.txt (or update input_file in the script)."
