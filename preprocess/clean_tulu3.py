import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="data")
parser.add_argument('--input', type=str, default="tulu3.jsonl")
parser.add_argument('--output', type=str, default="tulu3_cleaned.jsonl")
args = parser.parse_args()

# ========== Config ==========
dir_path = args.dir
INPUT_FILE = f"{dir_path}/{args.input}"
OUTPUT_FILE = f"{dir_path}/{args.output}"
# ============================

# tulu3 role mapping: allow user / assistant / system
ROLE_MAP = {
    "user": "human",
    "assistant": "gpt",
    "system": "system",
}
STANDARD_ROLES = {"human", "gpt", "system"}
ALLOWED_ROLES = set(ROLE_MAP.keys())


def normalize_role(role):
    """Return normalized role ('human'/'gpt'/'system') or None if invalid."""
    if not isinstance(role, str):
        return None
    r = role.strip().lower()
    if r not in ALLOWED_ROLES:
        return None
    return ROLE_MAP[r]


def is_valid_conversation(convs):
    """Validate format: human -> gpt -> human -> gpt ... -> gpt (must end with gpt)."""
    if not convs:
        return False
    if convs[0].get("from") != "human":
        return False
    if convs[-1].get("from") != "gpt":
        return False
    if len(convs) % 2 != 0:
        return False
    for i, msg in enumerate(convs):
        expected = "human" if i % 2 == 0 else "gpt"
        if msg.get("from") != expected:
            return False
    return True


def process_messages(messages):
    """
    Convert:
      messages: [{role: user/assistant/system, content: ...}, ...]
    To:
      conversations: [{from: human/gpt, value: ...}, ...]
    System contents are merged into the first human turn (as a prefix).
    Return None if any message has an invalid schema/role.
    """
    # 1) Normalize roles + rename fields (role->from, content->value)
    cleaned = []
    for msg in messages:
        if not (isinstance(msg, dict) and "role" in msg and "content" in msg):
            return None

        new_role = normalize_role(msg["role"])
        if new_role is None:
            return None

        cleaned.append({
            "from": new_role,
            "value": str(msg["content"])
        })

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
        # If there is no human, keep as-is; validation will fail and the sample will be discarded.

    return non_system_msgs


def main():
    total_read = 0
    kept = 0
    discarded = 0
    not_align = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="cleaning"):
            line = line.strip()
            if not line:
                continue
            total_read += 1

            try:
                item = json.loads(line)
            except Exception:
                discarded += 1
                continue

            if not isinstance(item, dict) or "messages" not in item:
                discarded += 1
                continue
            if not isinstance(item["messages"], list):
                discarded += 1
                continue

            processed = process_messages(item["messages"])
            if processed is None:
                discarded += 1
                continue

            if not is_valid_conversation(processed):
                not_align += 1
                discarded += 1
                continue

            # Unify key naming to instag style
            item.pop("messages", None)
            item["conversations"] = processed
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print("Data cleaning completed!")
    print(f"Total input samples: {total_read}")
    print(f"Kept samples: {kept}")
    print(f"Discarded samples: {discarded}")


if __name__ == "__main__":
    main()
