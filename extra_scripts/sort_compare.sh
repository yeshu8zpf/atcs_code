#!/bin/bash

set -euo pipefail

python preprocess/convert_to_xytext.py --dir data --input_file instag_cleaned.jsonl --output_file instag_xytext_fine.jsonl \
                                        --x_max_token 4000 --y_max_token 4000

python fine_compute/fine_compute_for_topk.py \
            --score_file score/llama/xsota/ifd_all/score.jsonl \
            --output_file score/llama/xsota_fine_for_topk/ifd_score.jsonl

python fine_compute/fine_compute_for_topk.py \
            --score_file score/llama/xsota/nll_all/score.jsonl \
            --output_file score/llama/xsota_fine_for_topk/nll_score.jsonl

python fine_compute/fine_compute_for_topk.py \
            --score_file score/llama/xsota/ufs_all/score.jsonl \
            --output_file score/llama/xsota_fine_for_topk/ufs_score.jsonl

python analyze/score_sort_compare.py \
            --file1 score/llama/xsota/ifd_all/score.jsonl \
            --file2 score/llama/xsota_fine_for_topk/ifd_score.jsonl \
            --score_key ifd \
            --metrics_output_path analyze/results/ifd_sort_compare.json

python analyze/score_sort_compare.py \
            --file1 score/llama/xsota/nll_all/score.jsonl \
            --file2 score/llama/xsota_fine_for_topk/nll_score.jsonl \
            --score_key nll \
            --metrics_output_path analyze/results/nll_sort_compare.json

python analyze/score_sort_compare.py \
            --file1 score/llama/xsota/ufs_all/score.jsonl \
            --file2 score/llama/xsota_fine_for_topk/ufs_score.jsonl \
            --score_key ufs \
            --metrics_output_path analyze/results/ufs_sort_compare.json
