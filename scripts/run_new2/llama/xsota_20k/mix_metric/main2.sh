#!/bin/bash
set -euo pipefail

COARSE_DIR="coarse_results/xsota_20K"
NEEDED_FILES=(
  "${COARSE_DIR}/ifd_top10.jsonl"
  "${COARSE_DIR}/nll_top10.jsonl"
  "${COARSE_DIR}/ufs_top10.jsonl"
)

need_preprocess=0
for file in "${NEEDED_FILES[@]}"; do
  if [[ ! -f "${file}" ]]; then
    need_preprocess=1
    break
  fi
done

if [[ "${need_preprocess}" -eq 1 ]]; then
  bash scripts/run_new2/llama/xsota_20k/mix_metric/preprocess.sh
fi

bash scripts/run_new2/llama/xsota_20k/mix_metric/run_ifd_nll.sh
# bash scripts/run_new2/llama/xsota_20k/mix_metric/run_ifd_ufs.sh
# bash scripts/run_new2/llama/xsota_20k/mix_metric/run_nll_ifd.sh
# bash scripts/run_new2/llama/xsota_20k/mix_metric/run_nll_ufs.sh
bash scripts/run_new2/llama/xsota_20k/mix_metric/run_ufs_ifd.sh
bash scripts/run_new2/llama/xsota_20k/mix_metric/run_ufs_nll.sh
