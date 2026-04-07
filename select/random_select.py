import json
import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ori_data_path', type=str,
                    default="/SSD/zpf/LLM/sft_85/tulu3_cleaned.jsonl")
parser.add_argument('--save_path', type=str,
                    default='coreset/qwen/tulu3/random/coreset.jsonl')
parser.add_argument('--num_samples', type=int, default=10000)
args = parser.parse_args()

ori_data_path = args.ori_data_path
save_path = args.save_path
num_samples = args.num_samples

# ---------- 1. 读取所有行 ----------
with open(ori_data_path, 'r', encoding='utf-8') as f:
    all_lines = f.readlines()

# ---------- 2. 随机抽样 ----------
num_samples = min(num_samples, len(all_lines))
selected_sample_lines = random.sample(all_lines, num_samples)

# ---------- 3. 保存 coreset ----------
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w', encoding='utf-8') as f:
    for line in selected_sample_lines:
        f.write(line)

# ---------- 4. 复制 dataset_info.json ----------
src_coreset = 'dataset_info_template/tulu3/dataset_info.json'
dst_dir = os.path.dirname(save_path)
shutil.copy(src_coreset, dst_dir)

print(f'[INFO] Randomly sampled {num_samples} lines')
print(f'[INFO] Saved to {save_path}')
print(f'[INFO] Copied {src_coreset} -> {dst_dir}')
