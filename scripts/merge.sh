#!/usr/bin/env bash
set -euo pipefail

llamafactory-cli export \
  --model_name_or_path ${MODEL_NAME} \
  --adapter_name_or_path ${SFT_DIR} \
  --finetuning_type lora \
  --template ${TEMPLATE} \
  --export_dir ${MERGE_DIR} \
  --export_size 4096 \
  --export_device auto