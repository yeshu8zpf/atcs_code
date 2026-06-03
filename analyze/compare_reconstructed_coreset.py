import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ranking consistency between full-text scoring and key-sentence reconstructed scoring."
    )
    parser.add_argument("--input_file", type=str, default="data_new/tulu3_xytext.jsonl")
    parser.add_argument("--model_name", type=str, default="models/Qwen2.5-7B")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--sentence_select_method", type=str, default="attn_threshold", choices=["attn_threshold", "max_tokens"])
    parser.add_argument("--attn_threshold", type=float, default=0.6)
    parser.add_argument("--sentence_score_agg", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--max_select_tokens", type=int, default=200)
    parser.add_argument("--y_max_tokens", type=int, default=-1)
    parser.add_argument("--dc_pdd_cap", type=float, default=float("inf"))
    parser.add_argument("--detail_jsonl", type=str, default="analysis/results/qwen_tulu3_reconstructed_vs_full_details.jsonl")
    parser.add_argument("--summary_json", type=str, default="analysis/results/qwen_tulu3_reconstructed_vs_full_summary.json")
    parser.add_argument("--summary_csv", type=str, default="analysis/results/qwen_tulu3_reconstructed_vs_full_summary.csv")
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_jsonl(path: str, max_samples: int) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
            if max_samples > 0 and len(data) >= max_samples:
                break
    return data


def get_freq_path(model_name: str) -> str:
    lower = model_name.lower()
    if "qwen" in lower:
        return "freqs/c4_Qwen_freq.pt"
    if "llama" in lower:
        return "freqs/c4_Llama_freq.pt"
    if "gpt" in lower:
        return "freqs/c4_gpt2_freq.pt"
    raise ValueError(f"No predefined freq file for model: {model_name}")


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
    sents = re.split(r"(?<=[.!?。！？])\s+", main_text)

    clean = []
    for s in sents:
        s = s.strip()
        if not s or is_punctuation_only(s):
            continue
        clean.append(s)

    if gpt_prefix:
        clean.append(gpt_prefix)
    return clean


def build_x_for_model(x_text: str) -> Optional[Dict[str, Any]]:
    sentences = split_into_sentences(x_text)
    if not sentences:
        return None
    return {
        "x_text_raw": x_text,
        "x_text_for_model": " ".join(sentences),
        "sentences": sentences,
    }


def maybe_truncate_ids(ids: List[int], max_tokens: int) -> List[int]:
    if max_tokens is None or max_tokens < 0:
        return ids
    return ids[:max_tokens]


def compute_token_unfamiliarity_score(
    logits_y: torch.Tensor,
    y_ids: List[int],
    log_freq: torch.Tensor,
    cap: float,
) -> float:
    probs = torch.softmax(logits_y, dim=-1)
    scores = []
    seen = set()
    vocab_size = log_freq.size(0)

    for i, tid in enumerate(y_ids):
        if tid in seen:
            continue
        seen.add(tid)
        if tid >= vocab_size:
            continue
        alpha = -probs[i, tid].item() * log_freq[tid].item()
        scores.append(min(alpha, cap))

    return sum(scores) / len(scores) if scores else 0.0


def get_logits_for_target(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ctx_text: str,
    tgt_text: str,
    y_max_tokens: int,
) -> Tuple[Optional[torch.Tensor], List[int]]:
    if ctx_text == "":
        if tokenizer.bos_token_id is not None:
            ctx_ids = [tokenizer.bos_token_id]
        elif tokenizer.eos_token_id is not None:
            ctx_ids = [tokenizer.eos_token_id]
        else:
            ctx_ids = tokenizer.encode("gpt: ", add_special_tokens=False)
    else:
        ctx_ids = tokenizer.encode(ctx_text, add_special_tokens=False)

    tgt_ids = maybe_truncate_ids(tokenizer.encode(tgt_text, add_special_tokens=False), y_max_tokens)
    if not ctx_ids or not tgt_ids:
        return None, []

    all_ids = ctx_ids + tgt_ids
    input_ids = torch.tensor([all_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits_all = outputs.logits[0]
    start = len(ctx_ids) - 1
    end = start + len(tgt_ids)
    if start < 0 or end > logits_all.size(0):
        return None, []
    return logits_all[start:end, :], tgt_ids


def compute_nll_and_ufs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    x_text: str,
    y_text: str,
    log_freq: torch.Tensor,
    y_max_tokens: int,
    dc_pdd_cap: float,
) -> Optional[Dict[str, Any]]:
    logits, y_ids = get_logits_for_target(model, tokenizer, x_text, y_text, y_max_tokens)
    if logits is None or not y_ids:
        return None

    tgt = torch.tensor(y_ids, device=logits.device, dtype=torch.long)
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
    nll = -float(selected_log_probs.mean().item())
    ufs = compute_token_unfamiliarity_score(logits, y_ids, log_freq, dc_pdd_cap)

    return {
        "nll": nll,
        "ufs": ufs,
        "y_ids": y_ids,
    }


def forward_with_attn_for_xy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    x_text: str,
    y_text: str,
    y_max_tokens: int,
) -> Optional[Dict[str, Any]]:
    x_ids = tokenizer.encode(x_text, add_special_tokens=False)
    y_ids = maybe_truncate_ids(tokenizer.encode(y_text, add_special_tokens=False), y_max_tokens)
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
        "logits_y": logits_y.detach().cpu(),
        "attn_y_to_x": attn_y_to_x.detach().cpu(),
    }


def reduce_tensor(values: torch.Tensor, method: str, dim=None) -> torch.Tensor:
    if method == "mean":
        return values.mean(dim=dim)
    if method == "sum":
        return values.sum(dim=dim)
    raise ValueError(f"Unknown sentence_score_agg: {method}")


def compute_sentence_scores(
    tokenizer: AutoTokenizer,
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
        else:
            score = 0.0
        sent_scores.append(score)
        sent_lens.append(curr_len)
        offset += curr_len
    return sent_scores, sent_lens


def select_key_sentences(sent_scores: List[float], sent_lens: List[int], method: str, attn_threshold: float, max_tokens: int) -> List[int]:
    if method == "max_tokens":
        candidates = [(i, sent_scores[i], sent_lens[i]) for i in range(len(sent_scores))]
        candidates.sort(key=lambda x: x[1], reverse=True)
        keep = []
        token_budget = 0
        for i, _, length in candidates:
            if token_budget + length <= max_tokens:
                keep.append(i)
                token_budget += length
        return sorted(keep)

    if method == "attn_threshold":
        candidates = [(i, score) for i, score in enumerate(sent_scores)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        total_score = sum(max(score, 0.0) for _, score in candidates)
        if total_score <= 0:
            return [max(range(len(sent_scores)), key=lambda i: sent_scores[i])] if sent_scores else []
        keep = []
        cumulative = 0.0
        for i, score in candidates:
            keep.append(i)
            cumulative += max(score, 0.0)
            if cumulative / total_score >= attn_threshold:
                break
        return sorted(keep)

    raise ValueError(f"Unknown sentence_select_method: {method}")


def reconstruct_x_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    x_text: str,
    y_text: str,
    sentence_select_method: str,
    sentence_score_agg: str,
    attn_threshold: float,
    max_select_tokens: int,
    y_max_tokens: int,
) -> Optional[Dict[str, Any]]:
    x_pack = build_x_for_model(x_text)
    if x_pack is None:
        return None

    fw = forward_with_attn_for_xy(model, tokenizer, x_pack["x_text_for_model"], y_text, y_max_tokens)
    if fw is None:
        return None

    x_scores_1d = fw["attn_y_to_x"].mean(dim=0)
    sent_scores, sent_lens = compute_sentence_scores(
        tokenizer,
        x_pack["sentences"],
        x_scores_1d,
        sentence_score_agg,
    )
    keep_idx = select_key_sentences(sent_scores, sent_lens, sentence_select_method, attn_threshold, max_select_tokens)
    final_x = " ".join([x_pack["sentences"][i] for i in keep_idx])

    if not final_x.strip().endswith(("gpt:", "GPT:")):
        final_x = f"{final_x.strip()} gpt:"

    return {
        "reconstructed_x_text": final_x,
        "sent_scores": sent_scores,
        "sent_lens": sent_lens,
        "keep_indices": keep_idx,
        "num_sentences": len(x_pack["sentences"]),
        "num_kept_sentences": len(keep_idx),
    }


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = math.sqrt(float((x_center ** 2).sum() * (y_center ** 2).sum()))
    if denom == 0:
        return float("nan")
    return float((x_center * y_center).sum() / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata_average(x), rankdata_average(y))


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0
    for i in range(n):
        dx = x[i] - x[i + 1 :]
        dy = y[i] - y[i + 1 :]
        sx = np.sign(dx)
        sy = np.sign(dy)
        both_nonzero = (sx != 0) & (sy != 0)
        concordant += int(np.sum(sx[both_nonzero] == sy[both_nonzero]))
        discordant += int(np.sum(sx[both_nonzero] != sy[both_nonzero]))
        tie_x += int(np.sum((sx == 0) & (sy != 0)))
        tie_y += int(np.sum((sx != 0) & (sy == 0)))
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def pairwise_agreement(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    agree = 0
    valid = 0
    for i in range(n):
        dx = x[i] - x[i + 1 :]
        dy = y[i] - y[i + 1 :]
        sx = np.sign(dx)
        sy = np.sign(dy)
        mask = (sx != 0) & (sy != 0)
        valid += int(np.sum(mask))
        agree += int(np.sum(sx[mask] == sy[mask]))
    if valid == 0:
        return float("nan")
    return agree / valid


def summarize_metric(details: List[Dict[str, Any]], metric: str, larger_is_better: bool, topk_ratios: List[float]) -> Dict[str, Any]:
    full_scores = np.array([row[f"full_{metric}"] for row in details], dtype=float)
    recon_scores = np.array([row[f"reconstructed_{metric}"] for row in details], dtype=float)
    ids = [row["idx"] for row in details]

    comp_full = -full_scores if larger_is_better else full_scores.copy()
    comp_recon = -recon_scores if larger_is_better else recon_scores.copy()

    ranked_full = sorted(zip(ids, comp_full.tolist()), key=lambda x: x[1])
    ranked_recon = sorted(zip(ids, comp_recon.tolist()), key=lambda x: x[1])

    overlap = {}
    for ratio in topk_ratios:
        k = max(1, int(len(details) * ratio))
        top_full = {idx for idx, _ in ranked_full[:k]}
        top_recon = {idx for idx, _ in ranked_recon[:k]}
        inter = len(top_full & top_recon)
        union = len(top_full | top_recon)
        overlap[f"top_{int(ratio * 100)}pct"] = {
            "k": k,
            "overlap_ratio": inter / k,
            "jaccard": inter / union if union else 1.0,
        }

    delta = recon_scores - full_scores
    return {
        "count": len(details),
        "spearman": spearman_corr(comp_full, comp_recon),
        "kendall_tau_b": kendall_tau_b(comp_full, comp_recon),
        "pairwise_agreement": pairwise_agreement(comp_full, comp_recon),
        "full_mean": float(full_scores.mean()),
        "reconstructed_mean": float(recon_scores.mean()),
        "delta_mean": float(delta.mean()),
        "delta_abs_mean": float(np.abs(delta).mean()),
        "overlap": overlap,
    }


def write_summary_csv(path: str, summary: Dict[str, Any]) -> None:
    ensure_parent(path)
    rows = []
    for metric, metric_summary in summary["metrics"].items():
        row = {
            "metric": metric,
            "count": metric_summary["count"],
            "spearman": metric_summary["spearman"],
            "kendall_tau_b": metric_summary["kendall_tau_b"],
            "pairwise_agreement": metric_summary["pairwise_agreement"],
            "full_mean": metric_summary["full_mean"],
            "reconstructed_mean": metric_summary["reconstructed_mean"],
            "delta_mean": metric_summary["delta_mean"],
            "delta_abs_mean": metric_summary["delta_abs_mean"],
        }
        for key, value in metric_summary["overlap"].items():
            row[f"{key}_k"] = value["k"]
            row[f"{key}_overlap_ratio"] = value["overlap_ratio"]
            row[f"{key}_jaccard"] = value["jaccard"]
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ensure_parent(args.detail_jsonl)
    ensure_parent(args.summary_json)
    ensure_parent(args.summary_csv)

    dataset = load_jsonl(args.input_file, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",
    ).eval()

    freq_data = torch.load(get_freq_path(args.model_name), map_location="cpu")
    log_freq = torch.log(freq_data["freq"] + 1e-12)

    details = []

    with open(args.detail_jsonl, "w", encoding="utf-8") as fout:
        for ex in tqdm(dataset, desc="Comparing full vs reconstructed scoring"):
            idx = ex["idx"]
            x_text = ex["x_text"]
            y_text = ex["y_text"]

            full_with = compute_nll_and_ufs(model, tokenizer, x_text, y_text, log_freq, args.y_max_tokens, args.dc_pdd_cap)
            full_no = compute_nll_and_ufs(model, tokenizer, "", y_text, log_freq, args.y_max_tokens, args.dc_pdd_cap)
            recon_pack = reconstruct_x_text(
                model,
                tokenizer,
                x_text,
                y_text,
                args.sentence_select_method,
                args.sentence_score_agg,
                args.attn_threshold,
                args.max_select_tokens,
                args.y_max_tokens,
            )

            if full_with is None or full_no is None or recon_pack is None:
                continue

            recon_with = compute_nll_and_ufs(
                model,
                tokenizer,
                recon_pack["reconstructed_x_text"],
                y_text,
                log_freq,
                args.y_max_tokens,
                args.dc_pdd_cap,
            )
            if recon_with is None:
                continue

            full_ifd = full_with["nll"] / (full_no["nll"] + 1e-8)
            recon_ifd = recon_with["nll"] / (full_no["nll"] + 1e-8)

            row = {
                "idx": idx,
                "full_nll": full_with["nll"],
                "reconstructed_nll": recon_with["nll"],
                "full_ifd": full_ifd,
                "reconstructed_ifd": recon_ifd,
                "full_ufs": full_with["ufs"],
                "reconstructed_ufs": recon_with["ufs"],
                "full_x_text": x_text,
                "reconstructed_x_text": recon_pack["reconstructed_x_text"],
                "y_text": y_text,
                "num_sentences": recon_pack["num_sentences"],
                "num_kept_sentences": recon_pack["num_kept_sentences"],
                "keep_indices": recon_pack["keep_indices"],
                "sentence_score_agg": args.sentence_score_agg,
            }
            details.append(row)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    topk_ratios = [0.01, 0.05, 0.10]
    summary = {
        "input_file": args.input_file,
        "model_name": args.model_name,
        "max_samples": args.max_samples,
        "processed_samples": len(details),
        "sentence_select_method": args.sentence_select_method,
        "sentence_score_agg": args.sentence_score_agg,
        "attn_threshold": args.attn_threshold,
        "max_select_tokens": args.max_select_tokens,
        "metrics": {
            "nll": summarize_metric(details, "nll", larger_is_better=True, topk_ratios=topk_ratios),
            "ifd": summarize_metric(details, "ifd", larger_is_better=True, topk_ratios=topk_ratios),
            "ufs": summarize_metric(details, "ufs", larger_is_better=True, topk_ratios=topk_ratios),
        },
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_summary_csv(args.summary_csv, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
