#!/bin/bash
set -euo pipefail

CONDA_DIR="anaconda3" ### todo
export CUDA_VISIBLE_DEVICES=0
DATA_DIR="data"  ### todo
MODEL_NAME="models/Qwen2.5-7B"


model="qwen"
dataset="tulu3"
JSONL_DATA_FILE="tulu3.json"
CLEANED_DATA_FILE="tulu3_cleaned.jsonl"
XY_DATA_FILE="tulu3_xytext.jsonl"

source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate coreset   ### todo

python preprocess/convert_parquet_to_jsonl.py
python preprocess/clean_tulu3.py --dir ${DATA_DIR} --input_file ${JSONL_DATA_FILE} --output_file ${CLEANED_DATA_FILE} 
python preprocess/convert_to_xytext.py --dir ${DATA_DIR} --input_file ${CLEANED_DATA_FILE} --output_file ${XY_DATA_FILE} 

python find_key_sentence/coarse_max_token.py --dir ${DATA_DIR} --input_file ${XY_DATA_FILE} --output_dir ${COARSE_DIR} \
                                        --ifd_file ${IFD_FILE} --nll_file ${NLL_FILE} --ufs_file ${UFS_FILE} --save_rate 0.1





