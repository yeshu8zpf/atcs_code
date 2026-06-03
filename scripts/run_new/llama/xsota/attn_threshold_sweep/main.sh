#!/bin/bash
set -euo pipefail

bash scripts/run_new/llama/xsota/attn_threshold_sweep/01_build_xytext.sh
bash scripts/run_new/llama/xsota/attn_threshold_sweep/02_loss_sweep.sh
