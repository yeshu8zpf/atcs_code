#!/bin/bash
set -euo pipefail

CONDA_DIR="anaconda3" ### todo
export CUDA_VISIBLE_DEVICES=0
DATA_DIR="data"  ### todo
export MODEL_NAME="models/Meta-Llama-3-8B"



model="llama"
dataset="instag"
COARSE_UTILITY="nll"
FINE_UTILITY="nll"
JSONL_DATA_FILE="instag_mix.json"
CLEANED_DATA_FILE="instag_cleaned.jsonl"
XY_DATA_FILE="instag_xytext.jsonl"
COARSE_DIR="coarse_results/${dataset}"
IFD_FILE="ifd_top10.jsonl"
NLL_FILE="nll_top10.jsonl"
UFS_FILE="ufs_top10.jsonl"

SCORE_FILE="score/${model}/${dataset}/${FINE_UTILITY}_all_left_truncate/score.jsonl"
run="left_truncatekey/${COARSE_UTILITY}_${FINE_UTILITY}"
export CORESET_DIR="coreset/${model}/${dataset}/${run}"
CORESET_PATH="${CORESET_DIR}/coreset.jsonl"
export SFT_DIR="sft_model/${model}/${dataset}/${run}"
export MERGE_DIR="merge_model/${model}/${dataset}/${run}"
export TEMPLATE="llama3"
export EVAL_DIR="eval_results/${model}/${dataset}"
export EVAL_NAME="${run}"

source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate coreset   ### todo

python preprocess/clean_instag.py --dir ${DATA_DIR} --input_file ${JSONL_DATA_FILE} --output_file ${CLEANED_DATA_FILE} 
python preprocess/convert_to_xytext.py --dir ${DATA_DIR} --input_file ${CLEANED_DATA_FILE} --output_file ${XY_DATA_FILE} 
python find_key_sentence/coarse_max_token.py --dir ${DATA_DIR} --input_file ${XY_DATA_FILE} --output_dir ${COARSE_DIR} \
                                        --ifd_file ${IFD_FILE} --nll_file ${NLL_FILE} --ufs_file ${UFS_FILE}

python fine_compute/fine_compute_left_truncate.py --raw_file ${DATA_DIR}/${XY_DATA_FILE} --score_file "${COARSE_DIR}/${COARSE_UTILITY}_top10.jsonl" --model_name ${MODEL_NAME} \
                                                  --output_file ${SCORE_FILE} --x_trunc_side "left" 

python select/select.py --ori_file ${DATA_DIR}/${CLEANED_DATA_FILE} --input_file ${SCORE_FILE} --output_file ${CORESET_PATH} --utility ${FINE_UTILITY}
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




