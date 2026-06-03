import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="data_new")
parser.add_argument("--input_file", type=str, default="instag_xytext.jsonl")
parser.add_argument("--output_file", type=str, default="analysis/attn_threshold_loss_sweep.jsonl")
parser.add_argument("--summary_file", type=str, default="analysis/attn_threshold_loss_summary.json")
parser.add_argument("--attn_model_name", type=str, default="models/gpt2")
parser.add_argument("--loss_model_name", type=str, default="")
parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
parser.add_argument("--max_samples", type=int, default=10000)
parser.add_argument("--sentence_score_agg", type=str, default="mean", choices=["mean", "sum"])
args = parser.parse_args()


BASE_DIR = args.dir
INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file
SUMMARY_FILE = args.summary_file
ATTN_MODEL_NAME = args.attn_model_name
LOSS_MODEL_NAME = args.loss_model_name if args.loss_model_name else args.attn_model_name
MAX_SAMPLES = args.max_samples
THRESHOLDS = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
SENTENCE_SCORE_AGG = args.sentence_score_agg


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


def build_x_for_model(x_text: str) -> Optional[Dict[str, Any]]:
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
    y_ids = tokenizer.encode(y_text, add_special_tokens=False)
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
        "logits_y": logits_y,
        "attn_y_to_x": attn_y_to_x,
    }


def reduce_tensor(values: torch.Tensor, method: str, dim=None) -> torch.Tensor:
    if method == "mean":
        return values.mean(dim=dim)
    if method == "sum":
        return values.sum(dim=dim)
    raise ValueError(f"Unknown sentence_score_agg: {method}")


def compute_sentence_scores(
    tokenizer,
    sentences: List[str],
    x_scores_1d: torch.Tensor,
    score_agg: str,
) -> Tuple[List[float], List[int]]:
    sent_scores = []
    sent_lens = []
    offset = 0
    for i, s in enumerate(sentences):
        piece = s if i == 0 else (" " + s)
        ids = tokenizer.encode(piece, add_special_tokens=False)
        curr_len = len(ids)
        if curr_len > 0:
            score = float(reduce_tensor(x_scores_1d[offset : offset + curr_len], score_agg).item())
            sent_scores.append(score)
        else:
            sent_scores.append(0.0)
        sent_lens.append(curr_len)
        offset += curr_len
    return sent_scores, sent_lens


def select_key_sentences_by_attn_threshold(sent_scores: List[float], threshold: float) -> List[int]:
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    candidates = [(i, score) for i, score in enumerate(sent_scores)]
    candidates.sort(key=lambda x: x[1], reverse=True)

    total_score = sum(max(score, 0.0) for _, score in candidates)
    if total_score <= 0:
        if not candidates:
            return []
        return [max(range(len(sent_scores)), key=lambda i: sent_scores[i])]

    keep_indices = []
    cumulative_score = 0.0
    for i, score in candidates:
        keep_indices.append(i)
        cumulative_score += max(score, 0.0)
        if cumulative_score / total_score >= threshold:
            break
    return sorted(keep_indices)


def rebuild_x_text(sentences: List[str], keep_indices: List[int]) -> str:
    final_x = " ".join([sentences[i] for i in keep_indices])
    if not final_x.strip().endswith(("gpt:", "GPT:")):
        final_x = f"{final_x.strip()} gpt:"
    return final_x


def get_logits_for_target(
    model,
    tokenizer,
    ctx_text: str,
    tgt_text: str,
):
    if ctx_text == "":
        if tokenizer.bos_token_id is not None:
            ctx_ids = [tokenizer.bos_token_id]
        elif tokenizer.eos_token_id is not None:
            ctx_ids = [tokenizer.eos_token_id]
        else:
            ctx_ids = tokenizer.encode("gpt: ", add_special_tokens=False)
    else:
        ctx_ids = tokenizer.encode(ctx_text, add_special_tokens=False)
    if len(ctx_ids) == 0:
        return None, []

    tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=False)
    if len(tgt_ids) == 0:
        return None, []

    all_ids = ctx_ids + tgt_ids
    input_ids = torch.tensor([all_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits_all = outputs.logits[0]
    ctx_len = len(ctx_ids)
    start = ctx_len - 1
    end = start + len(tgt_ids)
    if start < 0 or end > logits_all.size(0):
        return None, []

    logits_tgt = logits_all[start:end, :]
    return logits_tgt, tgt_ids


def compute_nll(model, tokenizer, x_text: str, y_text: str) -> Optional[float]:
    logits, tgt_ids = get_logits_for_target(
        model=model,
        tokenizer=tokenizer,
        ctx_text=x_text,
        tgt_text=y_text,
    )
    if logits is None or len(tgt_ids) == 0:
        return None

    tgt = torch.tensor(tgt_ids, device=logits.device, dtype=torch.long)
    logp = F.log_softmax(logits, dim=-1)
    logp_sel = logp.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(-logp_sel.mean().item())


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    ensure_parent_dir(OUTPUT_FILE)
    ensure_parent_dir(SUMMARY_FILE)

    attn_tokenizer = AutoTokenizer.from_pretrained(ATTN_MODEL_NAME, trust_remote_code=True)
    if attn_tokenizer.pad_token is None:
        attn_tokenizer.pad_token = attn_tokenizer.eos_token

    loss_tokenizer = AutoTokenizer.from_pretrained(LOSS_MODEL_NAME, trust_remote_code=True)
    if loss_tokenizer.pad_token is None:
        loss_tokenizer.pad_token = loss_tokenizer.eos_token

    print("Loading attention model...")
    attn_model = AutoModelForCausalLM.from_pretrained(
        ATTN_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",
    ).eval()

    if LOSS_MODEL_NAME == ATTN_MODEL_NAME:
        loss_model = attn_model
    else:
        print("Loading loss model...")
        loss_model = AutoModelForCausalLM.from_pretrained(
            LOSS_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

    data = []
    with open(os.path.join(BASE_DIR, INPUT_FILE), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if MAX_SAMPLES > 0:
        data = data[:MAX_SAMPLES]

    summary = {
        "attn_model_name": ATTN_MODEL_NAME,
        "loss_model_name": LOSS_MODEL_NAME,
        "input_file": os.path.join(BASE_DIR, INPUT_FILE),
        "sentence_score_agg": SENTENCE_SCORE_AGG,
        "thresholds": THRESHOLDS,
        "num_examples": 0,
        "num_success": 0,
        "original_nll_mean": None,
        "threshold_stats": {},
    }

    original_nll_values = []
    threshold_to_nlls = {str(t): [] for t in THRESHOLDS}
    threshold_to_kept_token_counts = {str(t): [] for t in THRESHOLDS}
    threshold_to_original_token_counts = {str(t): [] for t in THRESHOLDS}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for ex in tqdm(data, desc="Sweeping attention thresholds"):
            idx = ex["idx"]
            x_text = ex["x_text"]
            y_text = ex["y_text"]

            x_pack = build_x_for_model(x_text)
            if x_pack is None:
                continue

            fw = forward_with_attn_for_xy(attn_model, attn_tokenizer, x_pack["x_text_for_model"], y_text)
            if fw is None:
                continue

            x_scores_1d = fw["attn_y_to_x"].mean(dim=0)
            sent_scores, sent_lens = compute_sentence_scores(
                attn_tokenizer,
                x_pack["sentences"],
                x_scores_1d.cpu(),
                SENTENCE_SCORE_AGG,
            )

            original_nll = compute_nll(loss_model, loss_tokenizer, x_text, y_text)
            if original_nll is None:
                continue
            original_x_token_count = len(loss_tokenizer.encode(x_text, add_special_tokens=False))

            threshold_results = []
            success = True
            for threshold in THRESHOLDS:
                keep_indices = select_key_sentences_by_attn_threshold(sent_scores, threshold)
                rebuilt_x = rebuild_x_text(x_pack["sentences"], keep_indices)
                rebuilt_nll = compute_nll(loss_model, loss_tokenizer, rebuilt_x, y_text)
                if rebuilt_nll is None:
                    success = False
                    break
                rebuilt_x_token_count = len(loss_tokenizer.encode(rebuilt_x, add_special_tokens=False))
                token_keep_ratio = (
                    rebuilt_x_token_count / original_x_token_count
                    if original_x_token_count > 0
                    else None
                )

                threshold_results.append({
                    "threshold": threshold,
                    "keep_indices": keep_indices,
                    "num_kept_sentences": len(keep_indices),
                    "rebuilt_x_text": rebuilt_x,
                    "x_token_count_loss_model": rebuilt_x_token_count,
                    "token_keep_ratio": token_keep_ratio,
                    "token_keep_pct": token_keep_ratio * 100.0 if token_keep_ratio is not None else None,
                    "nll": rebuilt_nll,
                    "delta_vs_original": rebuilt_nll - original_nll,
                })

            if not success:
                continue

            original_nll_values.append(original_nll)
            for item in threshold_results:
                key = str(item["threshold"])
                threshold_to_nlls[key].append(item["nll"])
                threshold_to_kept_token_counts[key].append(item["x_token_count_loss_model"])
                threshold_to_original_token_counts[key].append(original_x_token_count)

            out_obj = {
                "idx": idx,
                "original_x_text": x_text,
                "y_text": y_text,
                "original_nll": original_nll,
                "original_x_token_count_loss_model": original_x_token_count,
                "sentences": x_pack["sentences"],
                "sent_scores": sent_scores,
                "sent_lens": sent_lens,
                "sentence_score_agg": SENTENCE_SCORE_AGG,
                "threshold_results": threshold_results,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            summary["num_success"] += 1

    summary["num_examples"] = len(data)
    if original_nll_values:
        summary["original_nll_mean"] = sum(original_nll_values) / len(original_nll_values)

    for threshold in THRESHOLDS:
        key = str(threshold)
        nll_values = threshold_to_nlls[key]
        kept_token_counts = threshold_to_kept_token_counts[key]
        original_token_counts = threshold_to_original_token_counts[key]
        keep_ratios = [
            kept / original
            for kept, original in zip(kept_token_counts, original_token_counts)
            if original > 0
        ]
        if nll_values:
            total_original_tokens = sum(original_token_counts)
            total_kept_tokens = sum(kept_token_counts)
            total_keep_ratio = (
                total_kept_tokens / total_original_tokens
                if total_original_tokens > 0
                else None
            )
            summary["threshold_stats"][key] = {
                "mean_nll": sum(nll_values) / len(nll_values),
                "count": len(nll_values),
                "mean_token_keep_ratio": (
                    sum(keep_ratios) / len(keep_ratios) if keep_ratios else None
                ),
                "mean_token_keep_pct": (
                    (sum(keep_ratios) / len(keep_ratios)) * 100.0 if keep_ratios else None
                ),
                "total_original_tokens": total_original_tokens,
                "total_kept_tokens": total_kept_tokens,
                "total_token_keep_ratio": total_keep_ratio,
                "total_token_keep_pct": (
                    total_keep_ratio * 100.0 if total_keep_ratio is not None else None
                ),
            }
        else:
            summary["threshold_stats"][key] = {
                "mean_nll": None,
                "count": 0,
                "mean_token_keep_ratio": None,
                "mean_token_keep_pct": None,
                "total_original_tokens": 0,
                "total_kept_tokens": 0,
                "total_token_keep_ratio": None,
                "total_token_keep_pct": None,
            }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote per-example results to: {OUTPUT_FILE}")
    print(f"Wrote summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
