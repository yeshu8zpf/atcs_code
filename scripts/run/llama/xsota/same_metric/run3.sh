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
XY_DATA_FILE="instag_xytext.jsonl"

COARSE_UTILITY="ufs"
FINE_UTILITY="ufs"
run="key/${COARSE_UTILITY}_${FINE_UTILITY}"
export CORESET_DIR="coreset/${model}/${dataset}/${run}"
CORESET_PATH="${CORESET_DIR}/coreset.jsonl"
export SFT_DIR="sft_model/${model}/${dataset}/${run}"
export MERGE_DIR="merge_model/${model}/${dataset}/${run}"
export TEMPLATE="llama3"
export EVAL_DIR="eval_results/${model}/${dataset}"
export EVAL_NAME="${run}"

COARSE_SCORE_FILE="coarse_results/${dataset}/${COARSE_UTILITY}_top10.jsonl"
FINE_SCORE_FILE="score/${model}/${dataset}/${FINE_UTILITY}_all/score.jsonl"

source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate coreset  ### todo


python fine_compute/fine_compute.py --input_file ${COARSE_SCORE_FILE} --output_file ${FINE_SCORE_FILE} --model_name ${MODEL_NAME} \
                                       --x_max_tokens 200 --y_max_tokens 50
python select/select.py --ori_file ${DATA_DIR}/${CLEANED_DATA_FILE} --input_file ${FINE_SCORE_FILE} --output_file ${CORESET_PATH} \
                         --utility ${FINE_UTILITY} --topk 10000
bash scripts/train.sh 
bash scripts/merge.sh

source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate opencompass  ### todo

################ arc 
opencompass opencompass/arc.py \
    --work-dir ${EVAL_DIR}/arc/${EVAL_NAME} \

LATEST_CSV=$(ls -dt ${EVAL_DIR}/arc/${EVAL_NAME}/* | head -n 1)/summary/*.csv
python opencompass/recode.py --file ${LATEST_CSV} 

############### bbh
opencompass opencompass/bbh.py \
    --work-dir ${EVAL_DIR}/bbh/${EVAL_NAME} \

LATEST_CSV=$(ls -dt ${EVAL_DIR}/bbh/${EVAL_NAME}/* | head -n 1)/summary/*.csv
python opencompass/recode.py --file ${LATEST_CSV} 

############### ifeval
opencompass opencompass/ifeval.py \
    --work-dir ${EVAL_DIR}/ifeval/${EVAL_NAME} \

LATEST_CSV=$(ls -dt ${EVAL_DIR}/ifeval/${EVAL_NAME}/* | head -n 1)/summary/*.csv
python opencompass/recode.py --file ${LATEST_CSV} 

############### mmlu
opencompass opencompass/mmlu.py \
    --work-dir ${EVAL_DIR}/mmlu/${EVAL_NAME} \

LATEST_CSV=$(ls -dt ${EVAL_DIR}/mmlu/${EVAL_NAME}/* | head -n 1)/summary/*.csv
python opencompass/recode.py --file ${LATEST_CSV} 
