#!/bin/bash
set -euo pipefail

CONDA_DIR="/root/anaconda3"
MODEL_NAME_OR_PATH="models/pythia-410m"
NUM_DOCS=15
DATA_FILES="freqs/C4/*.json.gz"
CACHE_DIR="freqs/.hf_datasets_cache"
OUTPUT_PATH="freqs/c4_pythia_freq.pt"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate coreset

python preprocess/freq/freq_stats_pythia.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --num_docs "${NUM_DOCS}" \
  --data_files "${DATA_FILES}" \
  --cache_dir "${CACHE_DIR}" \
  --output_path "${OUTPUT_PATH}"
