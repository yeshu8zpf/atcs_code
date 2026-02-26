# -*- coding: utf-8 -*-
from opencompass.models import VLLM
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask


from mmengine.config import read_base
import os as _os
MODEL_PATH = _os.environ.get("MERGE_DIR", "merge_model/qwen/tulu3/key/ifd_ifd")

with read_base():
    from opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets
datasets = bbh_datasets


models = [
    dict(
        type=VLLM,
        path=MODEL_PATH,
        max_out_len=1024,
        max_seq_len=4096,
        model_kwargs=dict(
            max_model_len=8000,
            gpu_memory_utilization=0.80,
        ),
        batch_size=16,
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
