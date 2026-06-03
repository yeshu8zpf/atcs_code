#!/bin/bash
set -euo pipefail

CONDA_DIR="/root/anaconda3"
export CUDA_VISIBLE_DEVICES=1
DATA_DIR="data_new"

MODEL_NAME="models/pythia-410m"
COARSE_MODEL_NAME="models/pythia-410m"

model="pythia_qwen"
dataset="tulu3"
PARQUET_DATA_DIR="tulu-3-sft-mixture"
JSONL_DATA_FILE="tulu3.jsonl"
CLEANED_DATA_FILE="tulu3_cleaned.jsonl"
XY_DATA_FILE="tulu3_xytext.jsonl"
TRUNCATED_SHAREGPT_FILE="tulu3_xytext_sharegpt.jsonl"
COARSE_DIR="coarse_results/${model}/${dataset}"
IFD_FILE="ifd_top10.jsonl"
NLL_FILE="nll_top10.jsonl"
UFS_FILE="ufs_top10.jsonl"
SENTENCE_SELECT_METHOD="attn_threshold"
ATTN_THRESHOLD="0.6"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate coreset

# cleaned/xytext are assumed to already exist under ${DATA_DIR}
# python preprocess/convert_parquet_to_jsonl.py \
#   --dir "${DATA_DIR}" \
#   --input_file "${PARQUET_DATA_DIR}" \
#   --output_file "${JSONL_DATA_FILE}"
# python preprocess/clean_tulu3.py \
#   --dir "${DATA_DIR}" \
#   --input "${JSONL_DATA_FILE}" \
#   --output "${CLEANED_DATA_FILE}"
# python preprocess/convert_to_xytext.py \
#   --dir "${DATA_DIR}" \
#   --input_file "${CLEANED_DATA_FILE}" \
#   --output_file "${XY_DATA_FILE}" \
#   --sharegpt_output_file "${TRUNCATED_SHAREGPT_FILE}" \
#   --model_name "${MODEL_NAME}"

python find_key_sentence/coarse_max_token.py \
  --dir "${DATA_DIR}" \
  --input_file "${XY_DATA_FILE}" \
  --output_dir "${COARSE_DIR}" \
  --ifd_file "${IFD_FILE}" \
  --nll_file "${NLL_FILE}" \
  --ufs_file "${UFS_FILE}" \
  --coarse_model "${COARSE_MODEL_NAME}" \
  --y_max_tokens -1 \
  --sentence_select_method "${SENTENCE_SELECT_METHOD}" \
  --attn_threshold "${ATTN_THRESHOLD}" \
  --save_rate 0.1
