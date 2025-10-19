#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="seasonal-rings"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

if have_cmd conda; then
  echo "[setup] Using Conda environment '${ENV_NAME}' from environment.yml"
  conda env update -n "${ENV_NAME}" -f environment.yml || conda env create -n "${ENV_NAME}" -f environment.yml
  echo
  echo "[setup] Done. Activate with:"
  echo "  conda activate ${ENV_NAME}"
elif have_cmd mamba; then
  echo "[setup] Using Mamba environment '${ENV_NAME}' from environment.yml"
  mamba env update -n "${ENV_NAME}" -f environment.yml || mamba env create -n "${ENV_NAME}" -f environment.yml
  echo
  echo "[setup] Done. Activate with:"
  echo "  conda activate ${ENV_NAME}"
else
  echo "[setup] Conda/Mamba not found. Falling back to a local virtualenv: .venv"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install numpy pandas matplotlib
  echo
  echo "[setup] Done. Activate with:"
  echo "  source .venv/bin/activate"
fi

