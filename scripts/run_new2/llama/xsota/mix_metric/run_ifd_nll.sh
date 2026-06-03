#!/bin/bash
set -euo pipefail

CONDA_DIR="/root/anaconda3"
export CUDA_VISIBLE_DEVICES=0
DATA_DIR="data_new"
export MODEL_NAME="models/Meta-Llama-3-8B"

model="llama"
dataset="xsota"
JSONL_DATA_FILE="instag_mix.json"
CLEANED_DATA_FILE="instag_cleaned.jsonl"
XY_DATA_FILE="instag_xytext.jsonl"
TRUNCATED_SHAREGPT_FILE="instag_xytext_sharegpt.jsonl"

COARSE_UTILITY="ifd"
FINE_UTILITY="nll"
base_run="key/${COARSE_UTILITY}_${FINE_UTILITY}"
train_run="${base_run}_lr5e5_ep3_bf16"
export CORESET_DIR="coreset/${model}/${dataset}/${base_run}"
CORESET_PATH="${CORESET_DIR}/coreset.jsonl"
export SFT_DIR="sft_model/${model}/${dataset}/${train_run}"
export MERGE_DIR="merge_model/${model}/${dataset}/${train_run}"
export TEMPLATE="llama3"
export EVAL_DIR="eval_results/${model}/${dataset}"
export EVAL_NAME="${train_run}"

COARSE_SCORE_FILE="coarse_results/${dataset}/${COARSE_UTILITY}_top10.jsonl"
FINE_SCORE_FILE="score/${model}/${dataset}/${FINE_UTILITY}_all/score.jsonl"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate coreset

python fine_compute/fine_compute.py \
  --input_file "${COARSE_SCORE_FILE}" \
  --output_file "${FINE_SCORE_FILE}" \
  --model_name "${MODEL_NAME}" \
  --x_max_tokens -1 \
  --y_max_tokens -1

python select/select.py \
  --ori_file "${DATA_DIR}/${TRUNCATED_SHAREGPT_FILE}" \
  --input_file "${FINE_SCORE_FILE}" \
  --output_file "${CORESET_PATH}" \
  --utility "${FINE_UTILITY}" \
  --topk 10000

bash scripts/train.sh
bash scripts/merge.sh

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate opencompass

if [[ ! -d "${MERGE_DIR}" ]]; then
  echo "[ERROR] MERGE_DIR does not exist: ${MERGE_DIR}" >&2
  echo "[ERROR] Run fine scoring, selection, training, and merge before evaluation." >&2
  exit 1
fi

run_eval() {
  local task_name="$1"
  local config_file="opencompass/${task_name}.py"
  local work_dir="${EVAL_DIR}/${task_name}/${EVAL_NAME}"

  opencompass "${config_file}" --work-dir "${work_dir}"

  local latest_run
  latest_run=$(ls -dt "${work_dir}"/* | head -n 1)
  local latest_csv
  latest_csv=$(find "${latest_run}/summary" -name '*.csv' | head -n 1)
  python opencompass/recode.py --file "${latest_csv}"
}

run_eval arc
run_eval bbh
run_eval ifeval
run_eval mmlu
