# -*- coding: utf-8 -*-
from opencompass.models import VLLM
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask


from mmengine.config import read_base
import os as _os
MODEL_PATH = _os.environ.get("MERGE_DIR", "merge_model/qwen/tulu3/key/ifd_ifd")
VLLM_MAX_MODEL_LEN = int(_os.environ.get("VLLM_MAX_MODEL_LEN", "8000"))
VLLM_GPU_MEMORY_UTILIZATION = float(_os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.80"))
VLLM_BATCH_SIZE = int(_os.environ.get("VLLM_BATCH_SIZE", "16"))

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
datasets = mmlu_datasets



models = [
    dict(
        type=VLLM,
        path=MODEL_PATH,
        max_out_len=1024,
        max_seq_len=4096,
        model_kwargs=dict(
            max_model_len=VLLM_MAX_MODEL_LEN,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        ),
        batch_size=VLLM_BATCH_SIZE,
        generation_kwargs=dict(
            temperature=0.0,
            top_p=1.0,
        ),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

# -----------------------------
# Inference Pipeline ONLY
# -----------------------------
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLInferTask),
    ),
)
