#!/bin/bash
set -euo pipefail

CONDA_DIR="anaconda3" ### todo
export CUDA_VISIBLE_DEVICES=0
DATA_DIR="data"  ### todo
MODEL_NAME="models/Meta-Llama-3-8B"


model="llama"
dataset="xsota"
JSONL_DATA_FILE="instag_mix.json"
CLEANED_DATA_FILE="instag_cleaned.jsonl"
XY_DATA_FILE="instag_xytext_fine.jsonl"
SCORE_FILE="score/${model}/${dataset}/fine_3000/score.jsonl"

source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate coreset   ### todo

python preprocess/clean_instag.py --dir ${DATA_DIR} --input_file ${JSONL_DATA_FILE} --output_file ${CLEANED_DATA_FILE}
python preprocess/convert_to_xytext.py --dir ${DATA_DIR} --input_file ${CLEANED_DATA_FILE} --output_file ${XY_DATA_FILE} \
                                        --x_max_token 4000 --y_max_token 4000

python fine_compute/fine_compute.py --input_file ${DATA_DIR}/${XY_DATA_FILE} --output_file ${SCORE_FILE} --model_name ${MODEL_NAME} \
                                    --x_max_tokens 3000 --y_max_tokens 3000

