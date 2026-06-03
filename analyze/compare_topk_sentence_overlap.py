import os
import json
import re
import argparse
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def is_punctuation_only(s: str) -> bool:
    return re.fullmatch(r"[\s\W]+", s) is not None


def split_into_sentences(text: str) -> List[str]:
    """
    Keep the same sentence splitting style as the original script.
    """
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


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_model_device(model) -> torch.device:
    """
    Works for both normal and device_map='auto' loaded models.
    """
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def forward_with_attn_for_xy(
    model,
    tokenizer,
    x_text: str,
    y_text: str,
    y_max_tokens: int,
):
    """
    Forward on x + truncated y, then extract:
      - x token ids
      - y token ids
      - last-layer averaged attention from y tokens to x tokens
    """
    x_ids = tokenizer.encode(x_text, add_special_tokens=False)
    y_ids = tokenizer.encode(y_text, add_special_tokens=False)[:y_max_tokens]

    if not x_ids or not y_ids:
        return None

    input_ids = x_ids + y_ids
    device = get_model_device(model)
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_tensor,
            attention_mask=torch.ones_like(input_tensor),
            output_attentions=True,
        )

    # outputs.attentions[-1]: [batch, num_heads, seq_len, seq_len]
    attn = outputs.attentions[-1][0].mean(dim=0)  # [seq_len, seq_len]

    x_len = len(x_ids)
    y_len = len(y_ids)

    # Rows: y tokens, Cols: x tokens
    attn_y_to_x = attn[x_len : x_len + y_len, :x_len]

    return {
        "x_ids": x_ids,
        "y_ids": y_ids,
        "attn_y_to_x": attn_y_to_x.detach().cpu(),
    }


def compute_sentence_scores(
    tokenizer,
    sentences: List[str],
    x_scores_1d: torch.Tensor,
    score_agg: str,
) -> Dict[str, Any]:
    """
    Aggregate token scores into sentence scores.
    This follows the original script's tokenization alignment style:
      piece = s if i == 0 else (" " + s)
    """
    sent_scores = []
    sent_lens = []
    offset = 0

    for i, s in enumerate(sentences):
        piece = s if i == 0 else (" " + s)
        ids = tokenizer.encode(piece, add_special_tokens=False)
        curr_len = len(ids)

        if curr_len > 0:
            score = float(reduce_tensor(x_scores_1d[offset : offset + curr_len], score_agg).item())
        else:
            score = 0.0

        sent_scores.append(score)
        sent_lens.append(curr_len)
        offset += curr_len

    return {
        "sent_scores": sent_scores,
        "sent_lens": sent_lens,
    }


def reduce_tensor(values: torch.Tensor, method: str, dim=None) -> torch.Tensor:
    if method == "mean":
        return values.mean(dim=dim)
    if method == "sum":
        return values.sum(dim=dim)
    raise ValueError(f"Unknown sentence_score_agg: {method}")


def get_topk_sentence_indices(sent_scores: List[float], topk: int) -> List[int]:
    topk = min(topk, len(sent_scores))
    ranked = sorted(range(len(sent_scores)), key=lambda i: sent_scores[i], reverse=True)
    return ranked[:topk]


def jaccard_overlap(a: List[int], b: List[int]) -> float:
    sa = set(a)
    sb = set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def recall_target_covered_by_proxy(proxy_topk: List[int], target_topk: List[int]) -> float:
    target_set = set(target_topk)
    if not target_set:
        return 1.0
    return len(set(proxy_topk) & target_set) / len(target_set)


def precision_proxy_against_target(proxy_topk: List[int], target_topk: List[int]) -> float:
    proxy_set = set(proxy_topk)
    if not proxy_set:
        return 1.0
    return len(proxy_set & set(target_topk)) / len(proxy_set)


def exact_match(a: List[int], b: List[int]) -> float:
    return 1.0 if set(a) == set(b) else 0.0


def safe_get_idx(ex: Dict[str, Any], fallback: int) -> Any:
    if "idx" in ex:
        return ex["idx"]
    if "id" in ex:
        return ex["id"]
    return fallback


def load_model_and_tokenizer(model_name: str, use_bfloat16: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = torch.bfloat16 if use_bfloat16 and torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="eager",
    ).eval()

    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/instag_xytext.jsonl")
    parser.add_argument("--small_model", type=str, default="models/gpt2")
    parser.add_argument("--large_model", type=str, default="models/Meta-Llama-3-8B")
    parser.add_argument("--output_file", type=str, default="analyze/results/top3_sentence_overlap_results.json")
    parser.add_argument("--detail_jsonl", type=str, default="analyze/results/top3_sentence_overlap_details.jsonl")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--y_max_tokens", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--sentence_score_agg", type=str, default="mean", choices=["mean", "sum"])
    args = parser.parse_args()

    print("Loading data...")
    data = load_jsonl(args.input_file)
    if args.max_samples > 0:
        data = data[: args.max_samples]
    print(f"Loaded {len(data)} samples.")

    print(f"Loading small model from: {args.small_model}")
    small_tokenizer, small_model = load_model_and_tokenizer(args.small_model)

    print(f"Loading large model from: {args.large_model}")
    large_tokenizer, large_model = load_model_and_tokenizer(args.large_model)

    detail_f = open(args.detail_jsonl, "w", encoding="utf-8")

    num_valid = 0
    sum_jaccard = 0.0
    sum_recall = 0.0
    sum_precision = 0.0
    sum_exact = 0.0

    for row_id, ex in enumerate(tqdm(data, desc="Comparing top-k sentence overlap")):
        idx = safe_get_idx(ex, row_id)

        x_text = ex.get("x_text", "")
        y_text = ex.get("y_text", "")

        if not x_text.strip() or not y_text.strip():
            continue

        x_pack = build_x_for_model(x_text)
        if x_pack is None:
            continue

        try:
            small_fw = forward_with_attn_for_xy(
                model=small_model,
                tokenizer=small_tokenizer,
                x_text=x_pack["x_text_for_model"],
                y_text=y_text,
                y_max_tokens=args.y_max_tokens,
            )

            large_fw = forward_with_attn_for_xy(
                model=large_model,
                tokenizer=large_tokenizer,
                x_text=x_pack["x_text_for_model"],
                y_text=y_text,
                y_max_tokens=args.y_max_tokens,
            )

            if small_fw is None or large_fw is None:
                continue

            # Aggregate y->x attention into x token scores
            small_x_scores_1d = small_fw["attn_y_to_x"].mean(dim=0)
            large_x_scores_1d = large_fw["attn_y_to_x"].mean(dim=0)

            # Sentence scores for each model
            small_sent_info = compute_sentence_scores(
                tokenizer=small_tokenizer,
                sentences=x_pack["sentences"],
                x_scores_1d=small_x_scores_1d,
                score_agg=args.sentence_score_agg,
            )
            large_sent_info = compute_sentence_scores(
                tokenizer=large_tokenizer,
                sentences=x_pack["sentences"],
                x_scores_1d=large_x_scores_1d,
                score_agg=args.sentence_score_agg,
            )

            small_topk = get_topk_sentence_indices(small_sent_info["sent_scores"], args.topk)
            large_topk = get_topk_sentence_indices(large_sent_info["sent_scores"], args.topk)

            jac = jaccard_overlap(small_topk, large_topk)
            rec = recall_target_covered_by_proxy(small_topk, large_topk)
            prec = precision_proxy_against_target(small_topk, large_topk)
            em = exact_match(small_topk, large_topk)

            sum_jaccard += jac
            sum_recall += rec
            sum_precision += prec
            sum_exact += em
            num_valid += 1

            detail_record = {
                "idx": idx,
                "topk": args.topk,
                "num_sentences": len(x_pack["sentences"]),
                "sentences": x_pack["sentences"],
                "small_sent_scores": small_sent_info["sent_scores"],
                "large_sent_scores": large_sent_info["sent_scores"],
                "small_topk_indices": small_topk,
                "large_topk_indices": large_topk,
                "small_topk_sentences": [x_pack["sentences"][i] for i in small_topk],
                "large_topk_sentences": [x_pack["sentences"][i] for i in large_topk],
                "jaccard_overlap": jac,
                "recall_target_covered_by_small": rec,
                "precision_small_against_target": prec,
                "exact_match": em,
                "sentence_score_agg": args.sentence_score_agg,
            }
            detail_f.write(json.dumps(detail_record, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[Warning] Failed on sample idx={idx}: {e}")
            continue

    detail_f.close()

    if num_valid == 0:
        raise RuntimeError("No valid samples were processed.")

    summary = {
        "input_file": args.input_file,
        "small_model": args.small_model,
        "large_model": args.large_model,
        "topk": args.topk,
        "y_max_tokens": args.y_max_tokens,
        "sentence_score_agg": args.sentence_score_agg,
        "num_valid_samples": num_valid,
        "avg_jaccard_overlap": sum_jaccard / num_valid,
        "avg_recall_target_covered_by_small": sum_recall / num_valid,
        "avg_precision_small_against_target": sum_precision / num_valid,
        "avg_exact_match_rate": sum_exact / num_valid,
        "detail_jsonl": args.detail_jsonl,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved summary to: {args.output_file}")
    print(f"Saved per-sample details to: {args.detail_jsonl}")


if __name__ == "__main__":
    main()
