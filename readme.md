# ATCS: Accelerating Target-Model-Aware Coreset Selection

## Step 1: Download Datasets
Download the following datasets and place them in the `data` directory:
- Tulu: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
- XSOTA: https://huggingface.co/datasets/AndrewZeng/deita_sota_pool

```bash
git clone https://huggingface.co/datasets/allenai/tulu-3-sft-mixture data/tulu-3-sft-mixture
git clone https://huggingface.co/datasets/AndrewZeng/deita_sota_pool data/deita_sota_pool
```

## Step 2: Create Environment
Create the conda environments using the yaml files in the `env` directory:
```bash
cd env
conda env create -f coreset.yaml
conda env create -f opencompass.yaml
cd ..
```

## Step 3: Download Models
Download the following models from Hugging Face and place them in the `models` directory:
- Qwen2.5-7B
- LLama3-8B

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B models/Qwen2.5-7B
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B models/Llama-3-8B
```

## Step 4: Run Experiments
For homogeneous utility:
```bash
bash scripts/run/llama/xsota/same_metric/main.sh
```

For mixed utility:
```bash
bash scripts/run/llama/xsota/same_metric/main.sh
```