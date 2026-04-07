#!/bin/bash
set -euo pipefail


CONDA_DIR="anaconda3"  ### todo
export CUDA_VISIBLE_DEVICES=1    ### todo
DATA_DIR="data"        ### todo
export MODEL_NAME="models/Meta-Llama-3-8B"   ### todo

model="llama"
dataset="xsota"
export COARSE_UTILITY='nll'
run="coarse/${COARSE_UTILITY}"
COARSE_FILE="coarse_results/instag/${COARSE_UTILITY}_top10.jsonl"
CLEANED_DATA_FILE="instag_cleaned.jsonl"
export CORESET_DIR="coreset/${model}/${dataset}/${run}"
CORESET_PATH="${CORESET_DIR}/coreset.jsonl"
export SFT_DIR="sft_model/${model}/${dataset}/${run}"
export MERGE_DIR="merge_model/${model}/${dataset}/${run}"
export TEMPLATE="llama3"
export EVAL_DIR="eval_results/${model}/${dataset}"
export EVAL_NAME="${run}"


source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate coreset ### todo

python select/select.py --ori_file ${DATA_DIR}/${CLEANED_DATA_FILE} --input_file ${COARSE_FILE} --output_file ${CORESET_PATH} --utility ${COARSE_UTILITY}
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
