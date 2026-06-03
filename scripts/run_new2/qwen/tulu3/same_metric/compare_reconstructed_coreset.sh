#!/bin/bash
set -euo pipefail

CONDA_DIR="/root/anaconda3"
export CUDA_VISIBLE_DEVICES=0

INPUT_FILE="data_new/tulu3_xytext.jsonl"
MODEL_NAME="models/Qwen2.5-7B"
MAX_SAMPLES=1000
ATTN_THRESHOLD=0.6
DETAIL_JSONL="analysis/results/qwen_tulu3_reconstructed_vs_full_details.jsonl"
SUMMARY_JSON="analysis/results/qwen_tulu3_reconstructed_vs_full_summary.json"
SUMMARY_CSV="analysis/results/qwen_tulu3_reconstructed_vs_full_summary.csv"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate coreset

python analyze/compare_reconstructed_coreset.py \
  --input_file "${INPUT_FILE}" \
  --model_name "${MODEL_NAME}" \
  --max_samples "${MAX_SAMPLES}" \
  --sentence_select_method "attn_threshold" \
  --attn_threshold "${ATTN_THRESHOLD}" \
  --detail_jsonl "${DETAIL_JSONL}" \
  --summary_json "${SUMMARY_JSON}" \
  --summary_csv "${SUMMARY_CSV}"
