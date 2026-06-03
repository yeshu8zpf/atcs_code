#!/bin/bash
set -euo pipefail

CONDA_DIR="/root/anaconda3"
MODEL_NAME_OR_PATH="models/Qwen2.5-7B"
DATA_FILES="freqs/C4/*.json.gz"
NUM_DOCS=15
OUTPUT_PATH="freqs/c4_Qwen_freq.pt"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate coreset

python freqs/compute_freq.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --data_files "${DATA_FILES}" \
  --num_docs "${NUM_DOCS}" \
  --output_path "${OUTPUT_PATH}"
