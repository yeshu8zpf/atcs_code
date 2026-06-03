import os
import json
from typing import List, Tuple, Dict, Any

from tqdm import tqdm
from transformers import AutoTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="data")
parser.add_argument('--input_file', type=str, default="instag_cleaned.jsonl")
parser.add_argument('--output_file', type=str, default='instag_xytext.jsonl')
parser.add_argument('--sharegpt_output_file', type=str, default="")
parser.add_argument('--x_max_token', type=int, default=500)
parser.add_argument('--y_max_token', type=int, default=100)
parser.add_argument('--model_name', type=str, default="models/gpt2")
args = parser.parse_args()
# ===== Config =====
BASE_DIR = args.dir
RAW_FILE = args.input_file
OUTPUT_FILE = args.output_file
SHAREGPT_OUTPUT_FILE = args.sharegpt_output_file

MODEL_NAME = args.model_name
X_MAX_TOKENS = args.x_max_token   # Token limit for x_text (all turns concatenated)
Y_MAX_TOKENS = args.y_max_token   # Token limit for y_text (last GPT turn)

# Conversation role prefixes (customizable)
HUMAN_PREFIX = "human: "
GPT_PREFIX = "gpt: "
# Fixed GPT prefix appended at end of x_text (to prompt response)
X_END_GPT_PREFIX = f"\n{GPT_PREFIX}"


def truncate_text_by_tokens(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> str:
    """
    Truncate text by token count, returning truncated string.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return ""
    truncated_ids = ids[:max_tokens]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return truncated_text.strip()


def build_sample(
    conversations: List[dict],
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> Tuple[str, str, List[Dict[str, str]], bool]:
    """
    Build a truncated training/scoring sample following these rules:
    1. Keep a conversation prefix in chronological order.
    2. Use the last kept human turn + X_END_GPT_PREFIX as x_text.
    3. Use the last kept gpt turn as y_text.
    4. Return (x_text, y_text, truncated_conversations, is_too_long), where
       is_too_long=True means even the first turn prompt exceeded max_tokens.
    """
    # Filter valid turns with human and gpt both present and non-empty content
    valid_turns = []
    for i in range(0, len(conversations), 2):
        if i + 1 >= len(conversations):
            break
        human_turn = conversations[i]
        gpt_turn = conversations[i + 1]

        if (human_turn.get("from") != "human" or gpt_turn.get("from") != "gpt" or
                not human_turn.get("value", "").strip() or not gpt_turn.get("value", "").strip()):
            continue

        valid_turns.append({
            "human": human_turn["value"].strip(),
            "gpt": gpt_turn["value"].strip()
        })

    if not valid_turns:
        return "", "", [], False

    prefix_token_len = 0
    last_kept_idx = -1

    for i, turn in enumerate(valid_turns):
        prompt_text = f"{HUMAN_PREFIX}{turn['human']}{X_END_GPT_PREFIX}"
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))

        if prefix_token_len + prompt_len <= max_tokens:
            last_kept_idx = i
            full_turn_text = f"{HUMAN_PREFIX}{turn['human']}\n{GPT_PREFIX}{turn['gpt']}\n"
            prefix_token_len += len(tokenizer.encode(full_turn_text, add_special_tokens=False))
        else:
            break

    if last_kept_idx < 0:
        return "", "", [], True

    final_x = ""
    truncated_conversations = []
    for turn in valid_turns[:last_kept_idx]:
        final_x += f"{HUMAN_PREFIX}{turn['human']}\n{GPT_PREFIX}{turn['gpt']}\n"
        truncated_conversations.extend([
            {"from": "human", "value": turn["human"]},
            {"from": "gpt", "value": turn["gpt"]},
        ])

    target_turn = valid_turns[last_kept_idx]
    final_x += f"{HUMAN_PREFIX}{target_turn['human']}{X_END_GPT_PREFIX}"
    truncated_conversations.append({"from": "human", "value": target_turn["human"]})

    return final_x, target_turn["gpt"], truncated_conversations, False


def process_conversations():
    base_dir = BASE_DIR
    data_file = RAW_FILE
    output_file = OUTPUT_FILE
    sharegpt_output_file = SHAREGPT_OUTPUT_FILE

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    in_path = os.path.join(base_dir, data_file)
    with open(in_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    results = []
    sharegpt_results = []
    too_long_count = 0
    empty_count = 0

    for d in tqdm(data, desc="Processing conversations ..."):
        raw_id = str(d.get("id", "")).strip()
        source = str(d.get("source", "")).strip()

        if source and raw_id:
            sample_id = f"{source}_{raw_id}"
        elif raw_id:
            sample_id = raw_id
        else:
            sample_id = "unknown"

        conversations = d.get("conversations", [])
        if not conversations:
            empty_count += 1
            continue

        x_text, target_gpt_content, truncated_conversations, is_too_long = build_sample(
            conversations,
            tokenizer,
            X_MAX_TOKENS,
        )
        if is_too_long:
            too_long_count += 1
            continue

        if not target_gpt_content or not x_text or not truncated_conversations:
            empty_count += 1
            continue

        y_text = truncate_text_by_tokens(target_gpt_content, tokenizer, Y_MAX_TOKENS)
        if not y_text:
            empty_count += 1
            continue

        new_dict = {
            "idx": sample_id,
            "x_text": x_text,
            "y_text": y_text,
        }
        results.append(new_dict)

        sharegpt_results.append({
            "id": sample_id,
            "conversations": truncated_conversations + [
                {"from": "gpt", "value": y_text},
            ],
        })

    out_path = os.path.join(base_dir, output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in results:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    if sharegpt_output_file:
        sharegpt_out_path = os.path.join(base_dir, sharegpt_output_file)
    else:
        sharegpt_name, sharegpt_ext = os.path.splitext(output_file)
        sharegpt_out_path = os.path.join(base_dir, f"{sharegpt_name}_sharegpt{sharegpt_ext or '.jsonl'}")

    with open(sharegpt_out_path, "w", encoding="utf-8") as f:
        for d in sharegpt_results:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Total original samples: {len(data)}")
    print(f"Valid samples processed: {len(results)}")
    print(f"Discarded due to last turn too long: {too_long_count}")
    print(f"Discarded empty/invalid samples: {empty_count}")
    print(f"Processed data saved to: {out_path}")
    print(f"Truncated ShareGPT data saved to: {sharegpt_out_path}")


if __name__ == "__main__":
    process_conversations()
