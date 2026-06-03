import argparse, os, json, shutil, re

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='coarse_results/instag_id/ifd_top10.jsonl')
parser.add_argument('--ori_file', type=str, default='data/instag_cleaned_new.jsonl')
parser.add_argument('--output_file', type=str, default='coreset/llama/xsota/coarse_id/ifd/coreset.jsonl')
parser.add_argument('--utility', type=str, default='ifd')
parser.add_argument('--topk', type=int, default=10000) 
parser.add_argument('--ori_format', type=str, default='auto', choices=['auto', 'cleaned', 'xytext'])
args = parser.parse_args()


ROLE_PATTERN = re.compile(r'(^|\n)(human|gpt): ')


def build_idx(obj):
    if 'idx' in obj:
        return obj['idx']

    if 'id' in obj and ('source' not in obj or 'instag' not in args.ori_file):
        return obj['id']

    if 'instag' in args.ori_file and 'source' in obj and 'id' in obj:
        return f"{obj['source']}_{obj['id']}"

    return obj['id']


def infer_ori_format(sample_obj):
    if args.ori_format != 'auto':
        return args.ori_format
    if 'x_text' in sample_obj and 'y_text' in sample_obj:
        return 'xytext'
    if 'conversations' in sample_obj:
        return 'cleaned'
    raise ValueError("Unable to infer ori_format from input sample.")


def xytext_to_conversations(x_text, y_text):
    matches = list(ROLE_PATTERN.finditer(x_text))
    conversations = []

    for i, match in enumerate(matches):
        role = match.group(2)
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(x_text)
        content = x_text[content_start:content_end]
        if content.startswith("\n"):
            content = content[1:]
        content = content.rstrip("\n")
        conversations.append({
            "from": role,
            "value": content,
        })

    if not conversations:
        raise ValueError("Failed to parse x_text into conversations.")

    if conversations[-1]["from"] == "gpt" and conversations[-1]["value"].strip() == "":
        conversations[-1]["value"] = y_text
    else:
        conversations.append({
            "from": "gpt",
            "value": y_text,
        })

    return conversations


def normalized_output_line(obj, ori_format):
    if ori_format == 'cleaned':
        return json.dumps(obj, ensure_ascii=False)

    if ori_format == 'xytext':
        out_obj = {
            "id": build_idx(obj),
            "conversations": xytext_to_conversations(obj["x_text"], obj["y_text"]),
        }
        return json.dumps(out_obj, ensure_ascii=False)

    raise ValueError(f"Unsupported ori_format: {ori_format}")


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
    ori_format = None
    for line in f:
        line = line.strip()
        if not line:
            raise ValueError("1")
        obj = json.loads(line)
        if ori_format is None:
            ori_format = infer_ori_format(obj)
            print(f"[INFO] Original data format: {ori_format}")
        id = build_idx(obj)
        if id not in id2score.keys():
            continue
        idx2line[id] = normalized_output_line(obj, ori_format)




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


# Training always uses the ShareGPT-style dataset schema, so we keep a single
# dataset_info template regardless of the original corpus name.
src_dataset_info_template = 'dataset_info_template/tulu3/dataset_info.json'
dst_dir = os.path.dirname(args.output_file)

if os.path.exists(src_dataset_info_template):
    shutil.copy(src_dataset_info_template, dst_dir)
    print(f'[INFO] Copied {src_dataset_info_template} -> {dst_dir}')
else:
    print(f'[WARNING] Template file not found: {src_dataset_info_template}')
