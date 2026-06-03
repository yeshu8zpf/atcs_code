#!/bin/bash
set -euo pipefail

CONDA_DIR="/root/anaconda3"
export CUDA_VISIBLE_DEVICES=0
DATA_DIR="data_new"
MODEL_NAME="models/gpt2"

JSONL_DATA_FILE="instag_mix.json"
CLEANED_DATA_FILE="instag_cleaned.jsonl"
XY_DATA_FILE="instag_xytext.jsonl"

X_MAX_TOKEN=500
Y_MAX_TOKEN=100

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate coreset

# python preprocess/clean_instag.py \
#   --dir "${DATA_DIR}" \
#   --input_file "${JSONL_DATA_FILE}" \
#   --output_file "${CLEANED_DATA_FILE}"

python preprocess/convert_to_xytext.py \
  --dir "${DATA_DIR}" \
  --input_file "${CLEANED_DATA_FILE}" \
  --output_file "${XY_DATA_FILE}" \
  --x_max_token "${X_MAX_TOKEN}" \
  --y_max_token "${Y_MAX_TOKEN}" \
  --model_name "${MODEL_NAME}"
