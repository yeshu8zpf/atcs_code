#!/bin/bash
set -euo pipefail

CONDA_DIR="/root/anaconda3"
export CUDA_VISIBLE_DEVICES=0
DATA_DIR="data_new"

INPUT_FILE="instag_xytext.jsonl"
ATTN_MODEL_NAME="models/gpt2"
LOSS_MODEL_NAME="models/gpt2"
THRESHOLDS="0.3,0.5,0.7,0.8,0.9,1.0"

# Keep these explicit because they control loss evaluation length, not sentence selection.
# Use -1 for no truncation.
X_MAX_TOKENS=-1
Y_MAX_TOKENS=-1
MAX_SAMPLES=1000

OUTPUT_FILE="analysis/attn_threshold_loss_sweep.jsonl"
SUMMARY_FILE="analysis/attn_threshold_loss_summary.json"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate coreset

python find_key_sentence/attn_threshold_loss_sweep.py \
  --dir "${DATA_DIR}" \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --summary_file "${SUMMARY_FILE}" \
  --attn_model_name "${ATTN_MODEL_NAME}" \
  --loss_model_name "${LOSS_MODEL_NAME}" \
  --thresholds "${THRESHOLDS}" \
  --x_max_tokens "${X_MAX_TOKENS}" \
  --y_max_tokens "${Y_MAX_TOKENS}" \
  --max_samples "${MAX_SAMPLES}"
