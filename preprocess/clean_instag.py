import json, argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="data")
parser.add_argument('--input_file', type=str, default="instag_mix.json")
parser.add_argument('--output_file', type=str, default='instag_cleaned.jsonl')
args = parser.parse_args()

# ========== Config ==========
INPUT_FILE = f"{args.dir}/{args.input_file}"
OUTPUT_FILE = f"{args.dir}/{args.output_file}"
# ============================

# Role mapping
ROLE_MAP = {
    "user": "human",
    "chatgpt": "gpt",
    "bing": "gpt",
    "bard": "gpt",
}
STANDARD_ROLES = {"human", "gpt", "system"}

def normalize_role(role):
    if not isinstance(role, str):
        return "human"
    r = role.lower().strip()
    if r in STANDARD_ROLES:
        return r
    elif r in ROLE_MAP:
        return ROLE_MAP.get(r, r)
    else:
        return None

def is_valid_conversation(convs):
    """Validate format: human -> gpt -> human -> gpt ... -> gpt (must end with gpt)."""
    if not convs:
        return False
    # Must start with human
    if convs[0]["from"] != "human":
        return False
    # Must end with gpt
    if convs[-1]["from"] != "gpt":
        return False
    # Must have an even number of turns (human-gpt pairs)
    if len(convs) % 2 != 0:
        return False
    # Strict alternation check
    for i, msg in enumerate(convs):
        expected = "human" if i % 2 == 0 else "gpt"
        if msg["from"] != expected:
            return False
    return True

def process_conversations(convs):
    # 1) Normalize roles and filter
    cleaned = []
    for msg in convs:
        if not (isinstance(msg, dict) and "from" in msg and "value" in msg):
            continue
        new_role = normalize_role(msg["from"])
        if new_role in STANDARD_ROLES:
            cleaned.append({"from": new_role, "value": str(msg["value"])})
        else:
            return None

    # 2) Split system vs non-system
    system_parts = []
    non_system_msgs = []
    for msg in cleaned:
        if msg["from"] == "system":
            if msg["value"].strip():
                system_parts.append(msg["value"].strip())
        else:
            non_system_msgs.append(msg)

    # 3) Merge system into the first human message
    system_prefix = "\n\n".join(system_parts).strip()
    if system_prefix and non_system_msgs:
        first_human_idx = None
        for i, msg in enumerate(non_system_msgs):
            if msg["from"] == "human":
                first_human_idx = i
                break
        if first_human_idx is not None:
            original = non_system_msgs[first_human_idx]["value"]
            non_system_msgs[first_human_idx]["value"] = system_prefix + "\n\n" + original
        # If there is no human, keep non_system_msgs unchanged (validation will fail later)

    return non_system_msgs

def main():
    total_read = 0
    kept = 0
    discarded = 0
    not_align = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue
            total_read += 1

            try:
                item = json.loads(line)
            except Exception:
                discarded += 1
                continue

            if not isinstance(item, dict) or "conversations" not in item:
                discarded += 1
                continue

            if not isinstance(item["conversations"], list):
                discarded += 1
                continue

            processed = process_conversations(item["conversations"])
            if processed is None:
                print("Abnormal sample!")
                continue

            if not is_valid_conversation(processed):
                not_align += 1
                discarded += 1
                continue

            item["conversations"] = processed
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print("Data cleaning completed!")
    print(f"Total input samples: {total_read}")
    print(f"Kept samples: {kept}")
    print(f"Discarded samples: {discarded}")


if __name__ == "__main__":
    main()
