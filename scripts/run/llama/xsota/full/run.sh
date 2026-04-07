#!/bin/bash
set -euo pipefail


CONDA_DIR="anaconda3"  ### todo
export CUDA_VISIBLE_DEVICES=0    ### todo
DATA_DIR="data"        ### todo
export MODEL_NAME="models/Meta-Llama-3-8B"



model="llama"
dataset="xsota"
CLEANED_DATA_FILE="instag_cleaned.jsonl"
export CORESET_DIR="coreset/${model}/${dataset}/full"
CORESET_PATH="${CORESET_DIR}/coreset.jsonl"
export SFT_DIR="sft_model/${model}/${dataset}/full"
export MERGE_DIR="merge_model/${model}/${dataset}/full"
export TEMPLATE="llama3"
export EVAL_DIR="eval_results/${model}/${dataset}"
export EVAL_NAME="full"

mkdir -p ${CORESET_DIR}
cp "${DATA_DIR}/${CLEANED_DATA_FILE}" "${CORESET_PATH}"
cp "dataset_info_template/${dataset}/dataset_info.json" "${CORESET_DIR}"


source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate coreset  ### todo

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


