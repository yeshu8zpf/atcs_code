import argparse, os, json, shutil

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='coarse_results/instag_id/ifd_top10.jsonl')
parser.add_argument('--ori_file', type=str, default='data/instag_cleaned_new.jsonl')
parser.add_argument('--output_file', type=str, default='coreset/llama/xsota/coarse_id/ifd/coreset.jsonl')
parser.add_argument('--utility', type=str, default='ifd')
parser.add_argument('--topk', type=int, default=10000) 
args = parser.parse_args()


id2score = {}
print(f"[INFO] Loading score file: {args.input_file}")
with open(args.input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            raise ValueError("1")
        try:
            obj = json.loads(line)
            idx = obj.get('idx')
            score = obj.get(args.utility)
            
            if idx is not None and score is not None:
                id2score[idx] = score
        except json.JSONDecodeError:
            raise KeyError("2")


idx2line = {}
print(f"[INFO] Loading original file: {args.ori_file}")
with open(args.ori_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            raise ValueError("1")
        obj = json.loads(line)
        if 'instag' in args.ori_file:
            id = f"{obj['source']}_{obj['id']}"
        else:
            id = obj['id']
        if id not in id2score.keys():
            continue
        idx2line[id] = line




sorted_ids = sorted(id2score.keys(), key=lambda x: id2score[x], reverse=True)
top_ids = sorted_ids[:args.topk]


selected_lines = []
missing_count = 0

for idx in top_ids:
    if idx in idx2line:
        selected_lines.append(idx2line[idx] + "\n") 
    else:
        raise ValueError("3")

print(f"[INFO] Top K: {args.topk}")
print(f"[INFO] Selected: {len(selected_lines)}")
print(f"[INFO] Missing/Not Found: {missing_count}")

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
with open(args.output_file, 'w', encoding='utf-8') as f:
    f.writelines(selected_lines)


src_coreset = 'dataset_info_template/tulu3/dataset_info.json'
dst_dir = os.path.dirname(args.output_file)

if os.path.exists(src_coreset):
    shutil.copy(src_coreset, dst_dir)
    print(f'[INFO] Copied {src_coreset} -> {dst_dir}')
else:
    print(f'[WARNING] Template file not found: {src_coreset}')
