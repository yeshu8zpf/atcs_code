#!/bin/bash
set -euo pipefail

# bash scripts/run_new2/llama/xsota/mix_metric/preprocess.sh
# bash scripts/run_new2/llama/xsota/mix_metric/run_ifd_nll.sh
bash scripts/run_new2/llama/xsota/mix_metric/run_ifd_ufs.sh
bash scripts/run_new2/llama/xsota/mix_metric/run_nll_ifd.sh
bash scripts/run_new2/llama/xsota/mix_metric/run_nll_ufs.sh
# bash scripts/run_new2/llama/xsota/mix_metric/run_ufs_ifd.sh
# bash scripts/run_new2/llama/xsota/mix_metric/run_ufs_nll.sh
