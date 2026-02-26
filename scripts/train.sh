#!/usr/bin/env bash
set -euo pipefail

llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME}" \
  --dataset coreset \
  --dataset_dir "${CORESET_DIR}" \
  --template ${TEMPLATE} \
  --finetuning_type lora \
  --output_dir "${SFT_DIR}" \
  --overwrite_output_dir True \
  --preprocessing_num_workers 8 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --warmup_steps 60 \
  --learning_rate 5e-6 \
  --num_train_epochs 5 \
  --fp16 \
  --save_strategy epoch
