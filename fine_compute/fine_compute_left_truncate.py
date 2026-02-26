import os
import json
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


# ---------------- args ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--raw_file', type=str, default="data/tulu3_gpt2.jsonl",
                    help="Raw data jsonl: contains idx and the original x_text")
parser.add_argument('--score_file', type=str, default="coarse_results/ifd_top10.jsonl",
                    help="Coarse filtering results jsonl: contains idx and y_text (its x_text will be ignored)")
parser.add_argument('--output_file', type=str, default='score/qwen/tulu3/ifd_all/score_trunc_control.jsonl')

parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B")
parser.add_argument('--x_max_tokens', type=int, default=200)
parser.add_argument('--y_max_tokens', type=int, default=50)
parser.add_argument('--x_trunc_side', type=str, default="left", choices=["left", "right"],
                    help='left=keep the last N tokens; right=keep the first N tokens')

parser.add_argument('--freq_path', type=str, default="freqs/c4_Llama_freq.pt")
parser.add_argument('--uncond_ctx_text', type=str, default="gpt: ",
                    help='Fixed context used for the no-instruction case to avoid undefined probability for the first y token')
parser.add_argument('--return_x_text_used', action='store_true',
                    help="Whether to save the decoded x_text_used in the output (audit only; adds small overhead)")

args = parser.parse_args()

RAW_FILE = args.raw_file
SCORE_FILE = args.score_file
OUT_PATH = args.output_file

MODEL_NAME = args.model_name
X_MAX_TOKENS = args.x_max_tokens
Y_MAX_TOKENS = args.y_max_tokens
X_TRUNC_SIDE = args.x_trunc_side

FREQ_PATH = args.freq_path
UNCOND_CTX_TEXT = args.uncond_ctx_text
RETURN_X_TEXT_USED = args.return_x_text_used


# ---------------- utils ----------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l:
                out.append(json.loads(l))
    return out


def build_raw_x_map(raw_file: str) -> Dict[Any, str]:
    """
    Build idx -> x_text from the raw file.
    """
    m = {}
    with open(raw_file, "r", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            ex = json.loads(l)
            idx = ex.get("idx", ex.get("id", None))
            if idx is None:
                continue
            x = ex.get("x_text", None)
            if x is None:
                continue
            m[idx] = x
    return m


# ---------------- scoring building blocks ----------------
def truncate_x_ids(tokenizer: AutoTokenizer, text: str, x_max_tokens: int, side: str) -> Tuple[List[int], int]:
    """
    Returns: truncated x_ids, and the original token length.
    side:
      left:  keep the last N tokens
      right: keep the first N tokens
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    orig_len = len(ids)

    if x_max_tokens is None or x_max_tokens < 0 or orig_len <= x_max_tokens:
        return ids, orig_len

    if side == "left":
        return ids[-x_max_tokens:], orig_len
    elif side == "right":
        return ids[:x_max_tokens], orig_len
    else:
        raise ValueError(f"unknown side: {side}")


def truncate_y_ids(tokenizer: AutoTokenizer, text: str, y_max_tokens: int) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if y_max_tokens is None or y_max_tokens < 0:
        return ids
    return ids[:y_max_tokens]


def compute_token_unfamiliarity_score(
    logits_y: torch.Tensor,   # [T, V]
    y_ids: List[int],
    log_freq: torch.Tensor,   # [V]
) -> float:
    if len(y_ids) == 0:
        return 0.0
    probs = torch.softmax(logits_y, dim=-1)  # [T, V]
    scores = []
    V = log_freq.size(0)
    for t, tid in enumerate(y_ids):
        if tid >= V:
            raise ValueError("4")
        scores.append(probs[t, tid].item() * log_freq[tid].item())
    return sum(scores) / len(scores) if scores else 0.0


def get_target_logprobs_and_logits(logits: torch.Tensor, tgt_ids: List[int]) -> Tuple[List[float], List[float]]:
    if len(tgt_ids) == 0:
        return [], []
    tgt = torch.tensor(tgt_ids, device=logits.device, dtype=torch.long)  # [T]
    logits_sel = logits.gather(1, tgt.unsqueeze(-1)).squeeze(-1)  # [T]
    logp = F.log_softmax(logits, dim=-1)
    logp_sel = logp.gather(1, tgt.unsqueeze(-1)).squeeze(-1)      # [T]
    return logp_sel.detach().cpu().tolist(), logits_sel.detach().cpu().tolist()


def mean_ll(logprob_list: List[float]) -> float:
    return float(sum(logprob_list) / len(logprob_list)) if logprob_list else 0.0


def get_logits_for_target_ids(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ctx_text: str,
    tgt_text: str,
    x_max_tokens: int,
    y_max_tokens: int,
    x_trunc_side: str,
    uncond_ctx_text: str,
    return_x_text_used: bool = False,
) -> Tuple[Optional[torch.Tensor], List[int], Dict[str, Any]]:
    """
    Returns:
      logits_tgt: [T, V] aligned logits
      y_ids: truncated target token ids (first y_max_tokens)
      debug: truncation/debug info
    """
    device = model.device

    # Unconditional context
    used_uncond = False
    if (ctx_text is None or ctx_text == "") and uncond_ctx_text:
        ctx_text = uncond_ctx_text
        used_uncond = True

    # x -> ids + truncate (no decode, no second encode)
    x_ids, x_orig_len = truncate_x_ids(tokenizer, ctx_text, x_max_tokens, x_trunc_side)
    if len(x_ids) == 0:
        return None, [], {}

    # y -> ids + right truncate
    y_ids = truncate_y_ids(tokenizer, tgt_text, y_max_tokens)
    T = len(y_ids)
    if T == 0:
        return None, [], {}

    all_ids = x_ids + y_ids
    input_ids = torch.tensor([all_ids], device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids)

    logits_all = out.logits[0]  # [S, V]

    # Alignment: y[0] is predicted by position len(x_ids)-1
    ctx_len = len(x_ids)
    start = ctx_len - 1
    end = start + T
    if start < 0 or end > logits_all.size(0):
        return None, [], {}

    logits_tgt = logits_all[start:end, :]  # [T, V]

    debug = {
        "x_tokens_orig": x_orig_len,
        "x_tokens_used": len(x_ids),
        "x_trunc_side": x_trunc_side,
        "used_uncond_ctx": used_uncond,
    }
    if return_x_text_used:
        debug["x_text_used"] = tokenizer.decode(x_ids, skip_special_tokens=True)

    return logits_tgt, y_ids, debug


def compute_nll_and_label_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    x_text: str,
    y_text: str,
    log_freq: Optional[torch.Tensor],
    x_max_tokens: int,
    y_max_tokens: int,
    x_trunc_side: str,
    uncond_ctx_text: str,
    return_x_text_used: bool = False,
) -> Optional[Dict[str, Any]]:
    logits, y_ids, debug = get_logits_for_target_ids(
        model=model,
        tokenizer=tokenizer,
        ctx_text=x_text,
        tgt_text=y_text,
        x_max_tokens=x_max_tokens,
        y_max_tokens=y_max_tokens,
        x_trunc_side=x_trunc_side,
        uncond_ctx_text=uncond_ctx_text,
        return_x_text_used=return_x_text_used,
    )
    if logits is None or len(y_ids) == 0:
        return None

    logprob_list, logit_list = get_target_logprobs_and_logits(logits, y_ids)
    nll = -mean_ll(logprob_list)

    ufs = None
    if log_freq is not None:
        ufs = compute_token_unfamiliarity_score(logits, y_ids, log_freq)

    return {
        "nll": nll,
        "ufs": ufs,
        "tgt_ids": y_ids,
        "logprob": logprob_list,
        "label_logits": logit_list,
        **debug,
    }


# ---------------- main ----------------
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

    # load freq
    log_freq = None
    if FREQ_PATH and os.path.exists(FREQ_PATH):
        freq_data = torch.load(FREQ_PATH, map_location="cpu")
        freq_tensor = freq_data["freq"]
        log_freq = torch.log(freq_tensor + 1e-12)
    else:
        print(f"[warn] freq_path not found: {FREQ_PATH}. UFS will be None.")

    # load data
    raw_x_map = build_raw_x_map(RAW_FILE)
    score_ds = load_jsonl(SCORE_FILE)

    # resume
    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if l:
                    done.add(json.loads(l)["idx"])
    else:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    missing_raw = 0
    written = 0

    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        for ex in tqdm(score_ds, desc=f"Scoring TRUNC control ({MODEL_NAME})"):
            idx = ex.get("idx", ex.get("id", None))
            if idx is None or idx in done:
                continue

            y_text = ex["y_text"]

            x_text_raw = raw_x_map.get(idx, None)
            if x_text_raw is None:
                missing_raw += 1
                continue

            # ----- with x (control: truncate raw x) -----
            res_with_x = compute_nll_and_label_logits(
                model, tokenizer,
                x_text_raw, y_text,
                log_freq=log_freq,
                x_max_tokens=X_MAX_TOKENS,
                y_max_tokens=Y_MAX_TOKENS,
                x_trunc_side=X_TRUNC_SIDE,
                uncond_ctx_text=UNCOND_CTX_TEXT,
                return_x_text_used=RETURN_X_TEXT_USED,
            )
            if res_with_x is None:
                continue

            # ----- no x -----
            res_no_x = compute_nll_and_label_logits(
                model, tokenizer,
                "", y_text,
                log_freq=log_freq,
                x_max_tokens=500,              # will be replaced by uncond_ctx_text; not important
                y_max_tokens=Y_MAX_TOKENS,
                x_trunc_side="right",          # not important
                uncond_ctx_text=UNCOND_CTX_TEXT,
                return_x_text_used=False,
            )
            if res_no_x is None:
                continue

            nll_with_x = res_with_x["nll"]
            nll_no_x = res_no_x["nll"]
            ifd = nll_with_x / (nll_no_x + 1e-8)

            out_obj = {
                "idx": idx,
                "ifd": ifd,

                "nll": nll_with_x,
                "nll_no_x": nll_no_x,
                "ufs": res_with_x["ufs"],

                # token-level (with x)
                "tgt_ids": res_with_x["tgt_ids"],
                "logprob": res_with_x["logprob"],
                "label_logits": res_with_x["label_logits"],

                # audit
                "x_trunc_side": res_with_x["x_trunc_side"],
                "x_tokens_orig": res_with_x["x_tokens_orig"],
                "x_tokens_used": res_with_x["x_tokens_used"],
                "used_uncond_ctx": res_with_x["used_uncond_ctx"],
            }
            if RETURN_X_TEXT_USED and "x_text_used" in res_with_x:
                out_obj["x_text_used"] = res_with_x["x_text_used"]

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

    print(f"[done] wrote: {OUT_PATH}")
    print(f"[stats] written={written}, missing_raw={missing_raw}, raw_map_size={len(raw_x_map)}, score_size={len(score_ds)}")


if __name__ == "__main__":
    main()
