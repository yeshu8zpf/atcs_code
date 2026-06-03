import os
import json
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


parser = argparse.ArgumentParser()

# Existing stage2 score file (contains idx)
parser.add_argument(
    "--score_file",
    type=str,
    default="score/llama/xsota/ifd_all/score.jsonl",
    help="Existing stage2 score jsonl file. Must contain idx."
)

# Full dataset with original x_text / y_text
parser.add_argument(
    "--full_data_file",
    type=str,
    default="data/instag_xytext_fine.jsonl",
    help="Full dataset jsonl containing idx, x_text, y_text."
)

# Output file for rescored full-sequence results
parser.add_argument(
    "--output_file",
    type=str,
    default="score/llama/xsota_fine_for_topk/ifd_score.jsonl",
    help="Output jsonl path."
)

parser.add_argument(
    "--model_name",
    type=str,
    default="models/Meta-Llama-3-8B",
    help="Target model used for rescoring."
)

# For full-sequence rescoring, set these large enough if you want near-full evaluation.
# If your original texts are very long, you can still cap them for practicality.
parser.add_argument(
    "--x_max_tokens",
    type=int,
    default=3000,
    help="Max tokens for x_text in rescoring."
)
parser.add_argument(
    "--y_max_tokens",
    type=int,
    default=3000,
    help="Max tokens for y_text in rescoring."
)
parser.add_argument(
    "--dc_pdd_cap",
    type=float,
    default=float("inf"),
    help="Upper bound a for DC-PDD token scores."
)



args = parser.parse_args()

SCORE_FILE = args.score_file
FULL_DATA_FILE = args.full_data_file
OUTPUT_FILE = args.output_file
MODEL_NAME = args.model_name
X_MAX_TOKENS = args.x_max_tokens
Y_MAX_TOKENS = args.y_max_tokens
DCPDD_CAP = args.dc_pdd_cap


# ====== UFS freq path ======
if "qwen" in MODEL_NAME.lower():
    FREQ_PATH = "freqs/c4_Qwen_freq.pt"
elif "llama" in MODEL_NAME.lower():
    FREQ_PATH = "freqs/c4_Llama_freq.pt"
elif "gpt" in MODEL_NAME.lower():
    FREQ_PATH = "freqs/c4_gpt2_freq.pt"
else:
    raise ValueError(f"Model not predefined for freq file: {MODEL_NAME}")


def compute_token_unfamiliarity_score(
    logits_y: torch.Tensor,   # [T, V]
    y_ids: List[int],
    log_freq: torch.Tensor,   # [V]
    cap: float = DCPDD_CAP,
) -> float:
    if len(y_ids) == 0:
        return 0.0

    probs = torch.softmax(logits_y, dim=-1)
    scores = []
    seen = set()

    vocab_size = log_freq.size(0)
    for t, tid in enumerate(y_ids):
        if tid in seen:
            continue
        seen.add(tid)
        if tid >= vocab_size:
            continue
        p = probs[t, tid].item()
        alpha = -p * log_freq[tid].item()
        scores.append(min(alpha, cap))

    return sum(scores) / len(scores) if scores else 0.0


def get_logits_for_target(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ctx_text: str,
    tgt_text: str,
    x_max_tokens: int,
    y_max_tokens: int,
    uncond_ctx_text: str = "gpt: ",
):
    """
    Return aligned logits for computing p(y|x).

    Alignment:
      logits[pos] predicts input_ids[pos+1]
      so y[0] is predicted by logits at position ctx_len - 1
    """
    device = model.device

    # Avoid empty-context issue with a true sequence-start token when available.
    if ctx_text == "":
        if tokenizer.bos_token_id is not None:
            ctx_ids = [tokenizer.bos_token_id]
        elif tokenizer.eos_token_id is not None:
            ctx_ids = [tokenizer.eos_token_id]
        else:
            ctx_ids = tokenizer.encode(uncond_ctx_text, add_special_tokens=False)
    else:
        ctx_ids = tokenizer.encode(ctx_text, add_special_tokens=False)
    if len(ctx_ids) == 0:
        return None, []

    # Keep at most the last x_max_tokens to preserve the response prompt tail.
    if x_max_tokens is not None and x_max_tokens >= 0 and len(ctx_ids) > x_max_tokens:
        ctx_ids = ctx_ids[-x_max_tokens:]

    tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=False)[:y_max_tokens]
    if len(tgt_ids) == 0:
        return None, []

    all_ids = ctx_ids + tgt_ids
    input_ids = torch.tensor([all_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits_all = outputs.logits[0]  # [S, V]

    ctx_len = len(ctx_ids)
    start = ctx_len - 1
    end = start + len(tgt_ids)

    if start < 0 or end > logits_all.size(0):
        return None, []

    logits_tgt = logits_all[start:end, :]
    return logits_tgt, tgt_ids


def mean_ll_from_list(logprob_list: List[float]) -> float:
    return float(sum(logprob_list) / len(logprob_list)) if logprob_list else 0.0


def get_target_logprobs_and_logits(
    logits: torch.Tensor,
    tgt_ids: List[int],
):
    if len(tgt_ids) == 0:
        return [], []

    tgt = torch.tensor(tgt_ids, device=logits.device, dtype=torch.long)

    logits_sel = logits.gather(1, tgt.unsqueeze(-1)).squeeze(-1)

    logp = F.log_softmax(logits, dim=-1)
    logp_sel = logp.gather(1, tgt.unsqueeze(-1)).squeeze(-1)

    return logp_sel.detach().cpu().tolist(), logits_sel.detach().cpu().tolist()


def compute_nll_and_label_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    x_text: str,
    y_text: str,
    log_freq: Optional[torch.Tensor],
    x_max_tokens: int,
    y_max_tokens: int,
) -> Optional[Dict[str, Any]]:
    logits, tgt_ids = get_logits_for_target(
        model=model,
        tokenizer=tokenizer,
        ctx_text=x_text,
        tgt_text=y_text,
        x_max_tokens=x_max_tokens,
        y_max_tokens=y_max_tokens,
    )
    if logits is None or len(tgt_ids) == 0:
        return None

    logprob_list, logit_list = get_target_logprobs_and_logits(logits, tgt_ids)
    ll = mean_ll_from_list(logprob_list)
    nll = -ll

    ufs = None
    if log_freq is not None:
        ufs = compute_token_unfamiliarity_score(logits, tgt_ids, log_freq)

    return {
        "nll": nll,
        "ufs": ufs,
        "tgt_ids": tgt_ids,
        "logprob": logprob_list,
        "label_logits": logit_list,
    }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_idx_to_text(full_data: List[Dict[str, Any]]) -> Dict[Any, Dict[str, str]]:
    idx_to_text = {}
    for ex in full_data:
        idx = ex["idx"] if "idx" in ex else ex.get("id")
        if idx is None:
            continue
        if "x_text" not in ex or "y_text" not in ex:
            continue
        idx_to_text[idx] = {
            "x_text": ex["x_text"],
            "y_text": ex["y_text"],
        }
    return idx_to_text


def main():
    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    print("Loading frequency file...")
    freq_data = torch.load(FREQ_PATH, map_location="cpu")
    freq_tensor = freq_data["freq"]
    log_freq = torch.log(freq_tensor + 1e-12)

    print(f"Loading existing score file: {SCORE_FILE}")
    score_data = load_jsonl(SCORE_FILE)

    print(f"Loading full dataset: {FULL_DATA_FILE}")
    full_data = load_jsonl(FULL_DATA_FILE)
    idx_to_text = build_idx_to_text(full_data)

    print(f"Full dataset indexed: {len(idx_to_text)} samples")

    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done.add(obj["idx"])
    else:
        out_dir = os.path.dirname(OUTPUT_FILE)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    missing_count = 0
    skipped_count = 0
    written_count = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for ex in tqdm(score_data, desc="Rescoring full-sequence samples"):
            idx = ex["idx"] if "idx" in ex else ex.get("id")
            if idx is None:
                skipped_count += 1
                continue
            if idx in done:
                continue

            if idx not in idx_to_text:
                missing_count += 1
                continue

            x_text = idx_to_text[idx]["x_text"]
            y_text = idx_to_text[idx]["y_text"]

            # With instruction
            res_with_x = compute_nll_and_label_logits(
                model=model,
                tokenizer=tokenizer,
                x_text=x_text,
                y_text=y_text,
                log_freq=log_freq,
                x_max_tokens=X_MAX_TOKENS,
                y_max_tokens=Y_MAX_TOKENS,
            )
            if res_with_x is None:
                skipped_count += 1
                continue

            # Without instruction
            res_no_x = compute_nll_and_label_logits(
                model=model,
                tokenizer=tokenizer,
                x_text="",
                y_text=y_text,
                log_freq=log_freq,
                x_max_tokens=500,
                y_max_tokens=Y_MAX_TOKENS,
            )
            if res_no_x is None:
                skipped_count += 1
                continue

            nll_with_x = res_with_x["nll"]
            nll_no_x = res_no_x["nll"]
            ifd = nll_with_x / (nll_no_x + 1e-8)
            ufs_with_x = res_no_x["ufs"]

            out_obj = {
                "idx": idx,

                # full-sequence rescored metrics
                "ifd": ifd,
                "nll": nll_with_x,
                "nll_no_x": nll_no_x,
                "ufs": ufs_with_x,

                # token-level info
                "tgt_ids": res_with_x["tgt_ids"],
                "logprob": res_with_x["logprob"],
                "label_logits": res_with_x["label_logits"],

                # original full texts
                "x_text": x_text,
                "y_text": y_text,
            }

            # Optionally preserve old stage2 scores if they exist
            for key in ["ifd", "nll", "nll_no_x", "ufs", "tgt_ids", "logprob", "label_logits"]:
                if key in ex:
                    out_obj[f"stage2_{key}"] = ex[key]

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            fout.flush()
            written_count += 1

    print(f"Done. Written: {written_count}")
    print(f"Missing idx in full dataset: {missing_count}")
    print(f"Skipped due to invalid/empty sample: {skipped_count}")


if __name__ == "__main__":
    main()
