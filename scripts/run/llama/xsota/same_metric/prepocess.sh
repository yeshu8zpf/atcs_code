#!/bin/bash
set -euo pipefail

CONDA_DIR="/root/anaconda3" ### todo
export CUDA_VISIBLE_DEVICES=0
DATA_DIR="data"  ### todo
MODEL_NAME="models/Meta-Llama-3-8B"


model="llama"
dataset="xsota"
JSONL_DATA_FILE="instag_mix.json"
CLEANED_DATA_FILE="instag_cleaned.jsonl"
XY_DATA_FILE="instag_xytext.jsonl"
COARSE_DIR="coarse_results/${dataset}"
IFD_FILE="ifd_top10.jsonl"
NLL_FILE="nll_top10.jsonl"
UFS_FILE="ufs_top10.jsonl"
SENTENCE_SELECT_METHOD="max_tokens"
ATTN_THRESHOLD="0.8"



source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate coreset   ### todo

# python preprocess/clean_instag.py --dir ${DATA_DIR} --input_file ${JSONL_DATA_FILE} --output_file ${CLEANED_DATA_FILE} 
# python preprocess/convert_to_xytext.py --dir ${DATA_DIR} --input_file ${CLEANED_DATA_FILE} --output_file ${XY_DATA_FILE} 

python find_key_sentence/coarse_max_token.py --dir ${DATA_DIR} --input_file ${XY_DATA_FILE} --output_dir ${COARSE_DIR} \
                                        --ifd_file ${IFD_FILE} --nll_file ${NLL_FILE} --ufs_file ${UFS_FILE} \
                                        --sentence_select_method ${SENTENCE_SELECT_METHOD} --attn_threshold ${ATTN_THRESHOLD} \
                                        --save_rate 0.1




