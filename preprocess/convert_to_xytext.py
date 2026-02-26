import os
import json
from typing import List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="data")
parser.add_argument('--input_file', type=str, default="instag_cleaned.jsonl")
parser.add_argument('--output_file', type=str, default='instag_xytext.jsonl')
parser.add_argument('--x_max_token', type=int, default=500)
parser.add_argument('--y_max_token', type=int, default=50)
parser.add_argument('--model_name', type=str, default="models/gpt2")
args = parser.parse_args()
# ===== Config =====
BASE_DIR = args.dir
RAW_FILE = args.input_file
OUTPUT_FILE = args.output_file

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


def build_x_text(
    conversations: List[dict],
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> Tuple[str, bool]:
    """
    Build x_text following these rules:
    1. Start from last human turn + X_END_GPT_PREFIX as base.
    2. If under max_tokens, add history turns in order: first turn → second to last turn → third to last turn...
    3. Return (x_text, is_too_long), where is_too_long=True means last turn exceeded token limit and sample should be discarded.
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
        return "", False

    last_turn = valid_turns[-1]
    base_x = f"{HUMAN_PREFIX}{last_turn['human']}{X_END_GPT_PREFIX}"
    base_ids = tokenizer.encode(base_x, add_special_tokens=False)
    base_token_len = len(base_ids)

    if base_token_len > max_tokens:
        return "", True  # Too long, discard sample

    history_turns = valid_turns[:-1]
    if not history_turns:
        return base_x, False

    # Reorder history: first turn → reversed remaining turns
    if len(history_turns) <= 1:
        add_turns = history_turns
    else:
        first_turn = [history_turns[0]]
        remaining_turns_reversed = history_turns[1:][::-1]
        add_turns = first_turn + remaining_turns_reversed

    selected_turns = []
    current_token_len = base_token_len

    for turn in add_turns:
        turn_text = f"\n{HUMAN_PREFIX}{turn['human']}\n{GPT_PREFIX}{turn['gpt']}"
        turn_ids = tokenizer.encode(turn_text, add_special_tokens=False)
        turn_token_len = len(turn_ids)

        if current_token_len + turn_token_len <= max_tokens:
            selected_turns.append(turn)
            current_token_len += turn_token_len
        else:
            break

    final_x = ""
    for turn in valid_turns:
        if turn in selected_turns:
            final_x += f"{HUMAN_PREFIX}{turn['human']}\n{GPT_PREFIX}{turn['gpt']}\n"

    final_x += base_x

    return final_x, False


def process_conversations():
    base_dir = BASE_DIR
    data_file = RAW_FILE
    output_file = OUTPUT_FILE

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    in_path = os.path.join(base_dir, data_file)
    with open(in_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    results = []
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

        x_text, is_too_long = build_x_text(conversations, tokenizer, X_MAX_TOKENS)
        if is_too_long:
            too_long_count += 1
            continue

        last_gpt_content = ""
        for turn in reversed(conversations):
            if turn.get("from") == "gpt" and turn.get("value", "").strip():
                last_gpt_content = turn["value"].strip()
                break

        if not last_gpt_content or not x_text:
            empty_count += 1
            continue

        y_text = truncate_text_by_tokens(last_gpt_content, tokenizer, Y_MAX_TOKENS)
        if not y_text:
            empty_count += 1
            continue

        new_dict = {
            "idx": sample_id,
            "x_text": x_text,
            "y_text": y_text,
        }
        results.append(new_dict)

    out_path = os.path.join(base_dir, output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in results:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Total original samples: {len(data)}")
    print(f"Valid samples processed: {len(results)}")
    print(f"Discarded due to last turn too long: {too_long_count}")
    print(f"Discarded empty/invalid samples: {empty_count}")
    print(f"Processed data saved to: {out_path}")


if __name__ == "__main__":
    process_conversations()
