#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_HOME="${CONDA_HOME:-/root/anaconda3}"
CONDA_SH="${CONDA_HOME}/etc/profile.d/conda.sh"

if [[ ! -f "${CONDA_SH}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
  else
    echo "conda not found. Please install Anaconda/Miniconda first, or set CONDA_HOME." >&2
    exit 1
  fi
fi

source "${CONDA_SH}"

create_env_if_missing() {
  local env_name="$1"
  local python_version="$2"

  if conda env list | awk '{print $1}' | grep -Fxq "${env_name}"; then
    echo "[setup_envs] Reusing existing env: ${env_name}"
  else
    echo "[setup_envs] Creating env: ${env_name}"
    conda create -y -n "${env_name}" python="${python_version}" pip setuptools wheel
  fi
}

create_env_if_missing "coreset" "3.8"
create_env_if_missing "opencompass" "3.10"

echo "[setup_envs] Installing coreset packages"
conda activate coreset
bash "${ROOT_DIR}/env/coreset.sh"
conda deactivate

echo "[setup_envs] Installing opencompass packages"
conda activate opencompass
bash "${ROOT_DIR}/env/opencompass.sh"
conda deactivate

echo "[setup_envs] Done."
echo "[setup_envs] Verify with:"
echo "  conda activate coreset && python -V"
echo "  conda activate opencompass && opencompass --version"
