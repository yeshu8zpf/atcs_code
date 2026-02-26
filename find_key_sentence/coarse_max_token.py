import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/SSD/zpf/LLM/sft_85")
parser.add_argument('--input_file', type=str, default="tulu3_gpt2.jsonl")
parser.add_argument('--output_dir', type=str, default='coarse_results')
parser.add_argument('--ifd_file', type=str, default='ifd_top10.jsonl')
parser.add_argument('--nll_file', type=str, default='nll_top10.jsonl')
parser.add_argument('--ufs_file', type=str, default='ufs_top10.jsonl')
parser.add_argument('--cache_file', type=str, default='forward_cache.jsonl')
parser.add_argument('--coarse_model', type=str, default='models/gpt2')
parser.add_argument('--max_select_tokens', type=int, default=200)
parser.add_argument('--save_rate', type=float, default=0.1)

args = parser.parse_args()

BASE_DIR = args.dir
DATA_FILE = args.input_file

OUT_DIR = args.output_dir
CACHE_FILE = args.cache_file

IFD_TOP_FILE = args.ifd_file
NLL_TOP_FILE = args.nll_file
UFS_TOP_FILE = args.ufs_file

MODEL_NAME = args.coarse_model
Y_MAX_TOKENS = 50

MAX_SELECT_TOKENS = args.max_select_tokens
FREQ_PATH = "freqs/c4_gpt2_freq.pt"

def is_punctuation_only(s: str) -> bool:
    return re.fullmatch(r"[\s\W]+", s) is not None

def split_into_sentences(text: str) -> List[str]:
    gpt_prefix_pattern = r"\s*(gpt|GPT):\s*$"
    gpt_prefix_match = re.search(gpt_prefix_pattern, text)
    gpt_prefix = ""
    main_text = text
    if gpt_prefix_match:
        gpt_prefix = gpt_prefix_match.group(0).strip()
        main_text = re.sub(gpt_prefix_pattern, "", text).strip()
    main_text = main_text.replace("\n", " ").strip()
    pattern = r"(?<=[.!?。！？])\s+"
    sents = re.split(pattern, main_text)
    clean = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if is_punctuation_only(s):
            continue
        clean.append(s)
    if gpt_prefix:
        clean.append(gpt_prefix)
    return clean

def build_x_for_model(tokenizer, x_text: str) -> Optional[Dict[str, Any]]:
    sentences = split_into_sentences(x_text)
    if not sentences:
        return None
    x_text_for_model = " ".join(sentences)
    return {
        "x_text_raw": x_text,
        "x_text_for_model": x_text_for_model,
        "sentences": sentences,
    }

def forward_with_attn_for_xy(model, tokenizer, x_text: str, y_text: str):
    x_ids = tokenizer.encode(x_text, add_special_tokens=False)
    y_ids = tokenizer.encode(y_text, add_special_tokens=False)[:Y_MAX_TOKENS]
    if not x_ids or not y_ids:
        return None
    input_ids = x_ids + y_ids
    input_tensor = torch.tensor([input_ids], device=model.device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_tensor,
            attention_mask=torch.ones_like(input_tensor),
            output_attentions=True,
        )
    logits = outputs.logits[0]
    attn = outputs.attentions[-1][0].mean(dim=0)
    x_len = len(x_ids)
    y_len = len(y_ids)
    logits_y = logits[x_len - 1 : x_len - 1 + y_len]
    attn_y_to_x = attn[x_len : x_len + y_len, :x_len]
    return {
        "x_ids": x_ids,
        "y_ids": y_ids,
        "logits_y": logits_y.cpu(),
        "attn_y_to_x": attn_y_to_x.cpu(),
    }

def forward_no_x_for_y(model, tokenizer, y_text: str):
    y_ids = tokenizer.encode(y_text, add_special_tokens=False)[:Y_MAX_TOKENS]
    if not y_ids:
        return None
    input_tensor = torch.tensor([y_ids], device=model.device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_tensor,
            attention_mask=torch.ones_like(input_tensor),
        )
    logits = outputs.logits[0]
    return {
        "y_ids": y_ids,
        "logits_y": logits.cpu(),
    }

def compute_nll(logits_y: torch.Tensor, y_ids: List[int]) -> float:
    log_probs = torch.log_softmax(logits_y, dim=-1)
    return -sum(float(log_probs[i, tid]) for i, tid in enumerate(y_ids)) / len(y_ids)

def compute_token_unfamiliarity_score(logits_y: torch.Tensor, y_ids: List[int], log_freq: torch.Tensor) -> float:
    probs = torch.softmax(logits_y, dim=-1)
    scores = []
    for i, tid in enumerate(y_ids):
        if tid >= log_freq.size(0):
            continue
        p = probs[i, tid].item()
        log_f = log_freq[tid].item()
        scores.append(p * log_f)
    return sum(scores) / len(scores) if scores else 0.0

def compute_sentence_scores_gpt2(tokenizer, sentences: List[str], x_scores_1d: torch.Tensor) -> Tuple[List[float], List[int]]:
    sent_scores = []
    sent_lens = []
    offset = 0
    for i, s in enumerate(sentences):
        piece = s if i == 0 else (" " + s)
        ids = tokenizer.encode(piece, add_special_tokens=False)
        curr_len = len(ids)
        if curr_len > 0:
            score = float(x_scores_1d[offset : offset + curr_len].mean())
            sent_scores.append(score)
        else:
            sent_scores.append(0.0)
        sent_lens.append(curr_len)
        offset += curr_len
    return sent_scores, sent_lens

def select_key_sentences(sentences, sent_scores, sent_lens, max_tokens):
    candidates = []
    for i in range(len(sentences)):
        candidates.append((i, sent_scores[i], sent_lens[i]))
    candidates.sort(key=lambda x: x[1], reverse=True)
    keep_indices = []
    current_tokens = 0
    for i, score, length in candidates:
        if current_tokens + length <= max_tokens:
            keep_indices.append(i)
            current_tokens += length
    return sorted(keep_indices)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cache_path = os.path.join(OUT_DIR, CACHE_FILE)
    error_log_path = os.path.join(OUT_DIR, "error_samples.log")

    done = set()
    error_ids = set()

    # Pre-initialize record lists
    ifd_records = []
    nll_records = []
    ufs_records = []

    if os.path.exists(error_log_path):
        with open(error_log_path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    error_ids.add(l.strip())

    if os.path.exists(cache_path):
        print(f"Cache file detected: {cache_path}. Restoring progress...")
        with open(cache_path, "r", encoding="utf-8") as f:
            for line_no, l in enumerate(f):
                if not l.strip():
                    continue
                try:
                    record = json.loads(l)
                    idx = record["idx"]
                    done.add(idx)

                    ifd_records.append({
                        "idx": idx,
                        "ifd": record["ifd"],
                        "x_text": record["x_text"],
                        "y_text": record["y_text"]
                    })
                    nll_records.append({
                        "idx": idx,
                        "nll": record["nll_with_x"],
                        "x_text": record["x_text"],
                        "y_text": record["y_text"]
                    })
                    ufs_records.append({
                        "idx": idx,
                        "ufs": record["ufs"],
                        "x_text": record["x_text"],
                        "y_text": record["y_text"]
                    })
                except json.JSONDecodeError:
                    print(f"Warning: cache file line {line_no} is corrupted; skipping.")
                    continue
        print(f"Restored {len(done)} records.")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    if os.path.exists(FREQ_PATH):
        freq_data = torch.load(FREQ_PATH, map_location="cpu")
        freq_tensor = freq_data["freq"]
        log_freq = torch.log(freq_tensor + 1e-12)
    else:
        print(f"Warning: {FREQ_PATH} not found. UFS will be 0.")
        freq_tensor = torch.zeros(50257)
        log_freq = torch.zeros(50257)

    data = []
    with open(os.path.join(BASE_DIR, DATA_FILE), "r", encoding="utf-8") as f:
        for l in f.readlines():
            if l.strip():
                data.append(json.loads(l))

    f_cache = open(cache_path, "a", encoding="utf-8")
    f_error = open(error_log_path, "a", encoding="utf-8")

    error_count = 0

    remaining_data = [ex for ex in data if ex["idx"] not in done and str(ex["idx"]) not in error_ids]
    print(f"Total samples: {len(data)}; remaining to process: {len(remaining_data)}")

    for ex in tqdm(remaining_data, desc="processing"):
        idx = ex["idx"]
        if idx in done or str(idx) in error_ids:
            continue

        try:
            x_pack = build_x_for_model(tokenizer, ex["x_text"])
            if x_pack is None:
                continue

            # Forward
            fw = forward_with_attn_for_xy(model, tokenizer, x_pack["x_text_for_model"], ex["y_text"])
            fw_no_x = forward_no_x_for_y(model, tokenizer, ex["y_text"])

            if fw is None or fw_no_x is None:
                continue

            # Metrics
            y_ids = fw["y_ids"]
            logits_y = fw["logits_y"]
            log_probs = torch.log_softmax(logits_y, dim=-1)
            target_logits = [float(logits_y[i, tid]) for i, tid in enumerate(y_ids)]
            target_logprobs = [float(log_probs[i, tid]) for i, tid in enumerate(y_ids)]
            nll_with_x = compute_nll(logits_y, y_ids)

            y_ids_nx = fw_no_x["y_ids"]
            logits_y_nx = fw_no_x["logits_y"]
            log_probs_nx = torch.log_softmax(logits_y_nx, dim=-1)
            target_logits_no_x = [float(logits_y_nx[i, tid]) for i, tid in enumerate(y_ids_nx)]
            target_logprobs_no_x = [float(log_probs_nx[i, tid]) for i, tid in enumerate(y_ids_nx)]
            nll_no_x = compute_nll(logits_y_nx, y_ids_nx)

            ifd = nll_with_x / (nll_no_x + 1e-8)
            ufs = compute_token_unfamiliarity_score(logits_y, y_ids, log_freq)

            # Selection
            x_scores_1d = fw["attn_y_to_x"].mean(dim=0)
            sent_scores, sent_lens = compute_sentence_scores_gpt2(tokenizer, x_pack["sentences"], x_scores_1d)

            keep_idx = select_key_sentences(
                sentences=x_pack["sentences"],
                sent_scores=sent_scores,
                sent_lens=sent_lens,
                max_tokens=MAX_SELECT_TOKENS
            )

            final_x = " ".join([x_pack["sentences"][i] for i in keep_idx])

            if not final_x.strip().endswith(("gpt:", "GPT:")):
                final_x = f"{final_x.strip()} gpt:"

            record = {
                "idx": idx,
                "ifd": ifd,
                "nll_with_x": nll_with_x,
                "nll_no_x": nll_no_x,
                "ufs": ufs,
                "target_logits": target_logits,
                "target_logprobs": target_logprobs,
                "target_logits_no_x": target_logits_no_x,
                "target_logprobs_no_x": target_logprobs_no_x,
                "sent_scores": sent_scores,
                "sent_lens": sent_lens,
                "keep_indices": keep_idx,
                "x_text": final_x,
                "y_text": ex["y_text"],
            }

            f_cache.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_cache.flush()

            ifd_records.append({"idx": idx, "ifd": ifd, "x_text": final_x, "y_text": ex["y_text"]})
            nll_records.append({"idx": idx, "nll": nll_with_x, "x_text": final_x, "y_text": ex["y_text"]})
            ufs_records.append({"idx": idx, "ufs": ufs, "x_text": final_x, "y_text": ex["y_text"]})

            done.add(idx)

        except Exception:
            error_count += 1
            f_error.write(f"{idx}\n")
            f_error.flush()
            error_ids.add(str(idx))
            continue

    f_cache.close()
    f_error.close()

    # ========== Save Top 10% ==========
    def save_top10(records, key, out_name, sort_ascending: bool):
        if not records:
            return None
        records.sort(key=lambda x: x[key], reverse=not sort_ascending)
        top_k = max(1, int(len(records) * args.save_rate))
        out_path = os.path.join(OUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records[:top_k]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Saving final results...")
    save_top10(ifd_records, "ifd", IFD_TOP_FILE, sort_ascending=True)
    save_top10(nll_records, "nll", NLL_TOP_FILE, sort_ascending=False)
    save_top10(ufs_records, "ufs", UFS_TOP_FILE, sort_ascending=True)

    print("Processing complete!")
    print(f"Total processed: {len(ifd_records)}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main()
