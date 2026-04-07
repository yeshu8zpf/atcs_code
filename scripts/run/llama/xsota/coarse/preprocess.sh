#!/bin/bash
set -euo pipefail

CONDA_DIR="anaconda3" ### todo
export CUDA_VISIBLE_DEVICES=1
DATA_DIR="data"  ### todo
MODEL_NAME="models/Meta-Llama-3-8B"

model="llama"
dataset="xsota"
JSONL_DATA_FILE="instag_mix.json"
CLEANED_DATA_FILE="instag_cleaned.jsonl"
XY_DATA_FILE="instag_xytext.jsonl"
COARSE_DIR="coarse_results/instag"
IFD_FILE="ifd_top10.jsonl"
NLL_FILE="nll_top10.jsonl"
UFS_FILE="ufs_top10.jsonl"
IFD_SCORE_FILE="score/${model}/instag/ifd_all/score.jsonl"
NLL_SCORE_FILE="score/${model}/instag/nll_all/score.jsonl"
UFS_SCORE_FILE="score/${model}/instag/ufs_all/score.jsonl"

source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate coreset   ### todo

python find_key_sentence/coarse_max_token.py --dir ${DATA_DIR} --input_file ${XY_DATA_FILE} --output_dir ${COARSE_DIR} \
                                        --ifd_file ${IFD_FILE} --nll_file ${NLL_FILE} --ufs_file ${UFS_FILE}