import os
import json
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default="coarse_results/ifd_top10.jsonl")
parser.add_argument('--output_file', type=str, default='score/qwen/tulu3/ifd_all/score.jsonl')
parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B")
parser.add_argument('--x_max_tokens', type=int, default=1000)
parser.add_argument('--y_max_tokens', type=int, default=70)
args = parser.parse_args()

# ================ Config ================
data_path = args.input_file
out_path = args.output_file
MODEL_NAME = args.model_name
# Truncation lengths for x / y tokens
X_MAX_TOKENS = args.x_max_tokens   # x: keep the last X_MAX_TOKENS tokens
Y_MAX_TOKENS = args.y_max_tokens   # y: keep only the first Y_MAX_TOKENS tokens

# ====== UFS freq ======
FREQ_PATH = "freqs/c4_Qwen_freq.pt"


# ================ Utilities ================

def compute_token_unfamiliarity_score(
    logits_y: torch.Tensor,   # [T, V]
    y_ids: List[int],
    log_freq: torch.Tensor,   # [V]
) -> float:
    """
    UFS = avg_t [ P_theta(y_t) * log_freq(y_t) ]
    """
    if len(y_ids) == 0:
        return 0.0

    probs = torch.softmax(logits_y, dim=-1)  # [T, V]
    scores = []

    V = log_freq.size(0)
    for t, tid in enumerate(y_ids):
        if tid >= V:
            continue
        p = probs[t, tid].item()
        scores.append(p * log_freq[tid].item())

    return sum(scores) / len(scores) if scores else 0.0


def get_logits_for_target(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ctx_text: str,
    tgt_text: str,
    x_max_tokens: int = X_MAX_TOKENS,
    y_max_tokens: int = Y_MAX_TOKENS,
    uncond_ctx_text: str = "gpt: ",   # Use this as x for the unconditional case
):
    """
    Return aligned logits for computing p(y|x).

    Rules:
      - x (ctx_text): keep at most the last x_max_tokens tokens
      - y (tgt_text): keep only the first y_max_tokens tokens
      - If ctx_text is empty, use uncond_ctx_text as an "unconditional context" to avoid the issue
        that the probability of the first y token is undefined with an empty context
        (i.e., compute p(y | uncond_ctx_text)).

    Alignment for an autoregressive LM:
      - logits[pos] predicts input_ids[pos+1]
      - Therefore, y's first token is predicted by the logits at the last x token position
        start = ctx_len - 1
        logits_tgt = logits_all[start : start+T]
    """
    device = model.device

    # 0) Unconditional handling: replace empty ctx with a fixed prefix
    if ctx_text == "" and uncond_ctx_text:
        ctx_text = uncond_ctx_text

    # 1) tokenize x
    ctx_ids = tokenizer.encode(ctx_text, add_special_tokens=False)
    if len(ctx_ids) == 0:
        # If even uncond_ctx_text becomes empty after tokenization, we cannot proceed
        return None, []

    if x_max_tokens is not None and x_max_tokens >= 0 and len(ctx_ids) > x_max_tokens:
        print(f"[warn] length of x_text exceeds max_token! {len(ctx_ids)} -> keep last {x_max_tokens}")
        ctx_ids = ctx_ids[:x_max_tokens]

    # 2) tokenize y
    tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=False)[:y_max_tokens]
    T = len(tgt_ids)
    if T == 0:
        return None, []

    # 3) concatenate input: [x] + [y]
    all_ids = ctx_ids + tgt_ids
    input_ids = torch.tensor([all_ids], device=device)

    # 4) forward
    with torch.no_grad():
        out = model(input_ids=input_ids)

    logits_all = out.logits[0]  # [S, V]

    # 5) aligned logits: y[0] is predicted by position ctx_len-1
    ctx_len = len(ctx_ids)
    start = ctx_len - 1
    end = start + T
    if start < 0 or end > logits_all.size(0):
        return None, []

    logits_tgt = logits_all[start:end, :]   # [T, V] aligned per-token logits for y
    return logits_tgt, tgt_ids


def mean_ll_from_list(logprob_list: List[float]) -> float:
    return float(sum(logprob_list) / len(logprob_list)) if logprob_list else 0.0


def get_target_logprobs_and_logits(
    logits: torch.Tensor,  # [T, V]
    tgt_ids: List[int],
):
    if len(tgt_ids) == 0:
        return [], []

    tgt = torch.tensor(tgt_ids, device=logits.device, dtype=torch.long)  # [T]

    # raw logits of gold token
    logits_sel = logits.gather(1, tgt.unsqueeze(-1)).squeeze(-1)  # [T]

    # log prob of gold token
    logp = F.log_softmax(logits, dim=-1)
    logp_sel = logp.gather(1, tgt.unsqueeze(-1)).squeeze(-1)      # [T]

    return logp_sel.detach().cpu().tolist(), logits_sel.detach().cpu().tolist()


def compute_nll_and_label_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    x_text: str,
    y_text: str,
    log_freq: Optional[torch.Tensor] = None,
    x_max_tokens: int = X_MAX_TOKENS,
    y_max_tokens: int = Y_MAX_TOKENS,
) -> Optional[Dict[str, Any]]:
    """
    Returns:
      - nll: mean negative log-likelihood
      - ufs: Token Unfamiliarity Score (if log_freq is provided)
      - tgt_ids / logprob / label_logits
    """
    logits, tgt_ids = get_logits_for_target(
        model,
        tokenizer,
        x_text,
        y_text,
        x_max_tokens=x_max_tokens,
        y_max_tokens=y_max_tokens,
    )
    if logits is None or len(tgt_ids) == 0:
        return None

    logprob_list, logit_list = get_target_logprobs_and_logits(logits, tgt_ids)
    ll = mean_ll_from_list(logprob_list)
    nll = -ll

    ufs = None
    if log_freq is not None and len(x_text) > 0:
        ufs = compute_token_unfamiliarity_score(logits, tgt_ids, log_freq)

    return {
        "nll": nll,
        "ufs": ufs,
        "tgt_ids": tgt_ids,
        "logprob": logprob_list,
        "label_logits": logit_list,
    }


# ================ Main ================

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    # ---- load freq for UFS ----
    # freq_data["freq"] should be [vocab_size] aligned with this tokenizer's ids
    freq_data = torch.load(FREQ_PATH, map_location="cpu")
    freq_tensor = freq_data["freq"]
    log_freq = torch.log(freq_tensor + 1e-12)

    dataset = [json.loads(l) for l in open(data_path, "r", encoding="utf-8")]

    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for l in f:
                done.add(json.loads(l)["idx"])
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as fout:
        for ex in tqdm(dataset, desc="Scoring with IFD+UFS (Qwen2.5-7B-Instruct)"):
            idx = ex["idx"] if "idx" in ex else ex["id"]
            if idx in done:
                continue

            x_text = ex["x_text"]
            y_text = ex["y_text"]

            # ---------- with instruction ----------
            res_with_x = compute_nll_and_label_logits(
                model,
                tokenizer,
                x_text,
                y_text,
                log_freq=log_freq,
                x_max_tokens=X_MAX_TOKENS,
                y_max_tokens=Y_MAX_TOKENS,
            )
            if res_with_x is None:
                continue

            # ---------- without instruction ----------
            res_no_x = compute_nll_and_label_logits(
                model,
                tokenizer,
                "",          # no instruction
                y_text,
                log_freq=log_freq,
                x_max_tokens=500,
                y_max_tokens=Y_MAX_TOKENS,
            )
            if res_no_x is None:
                continue

            nll_with_x = res_with_x["nll"]
            nll_no_x = res_no_x["nll"]
            ifd = nll_with_x / (nll_no_x + 1e-8)

            ufs_with_x = res_with_x["ufs"]

            out_obj = {
                "idx": idx,
                "ifd": ifd,

                # NLL
                "nll": nll_with_x,
                "nll_no_x": nll_no_x,

                # UFS
                "ufs": ufs_with_x,

                # token-level info (from the instructed run)
                "tgt_ids": res_with_x["tgt_ids"],
                "logprob": res_with_x["logprob"],
                "label_logits": res_with_x["label_logits"],
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[IFD+UFS Score] Written to: {out_path}")


if __name__ == "__main__":
    main()
