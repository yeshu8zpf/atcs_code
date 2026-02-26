import os
import json
import argparse
from collections import Counter
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='models/Llama3.1-8B'
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        default=15,
        help="Number of C4 documents used as reference corpus (paper uses 15)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="freqs/c4_Llama_freq.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True
    )

    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")

    print("Loading C4 dataset (English)...")
    dataset = load_dataset("json", data_files="/SSD/zpf/LLM/sft_85/C4/*.json.gz", split="train")

    print(f"Using first {args.num_docs} documents from C4 as reference corpus")

    token_counter = Counter()
    total_tokens = 0

    for idx in tqdm(range(args.num_docs), desc="Processing C4 documents"):
        text = dataset[idx]["text"]

        # Tokenize WITHOUT adding special tokens
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        token_counter.update(token_ids)
        total_tokens += len(token_ids)

    print(f"Total tokens counted: {total_tokens}")

    # === Laplace smoothing ===
    # f(x) = (count(x) + 1) / (total_tokens + vocab_size)
    print("Applying Laplace (+1) smoothing...")

    freq_dict = torch.zeros(vocab_size, dtype=torch.float64)

    for token_id, count in token_counter.items():
        freq_dict[token_id] = count + 1

    # Tokens never appearing still get +1
    freq_dict += 1
    freq_dict /= (total_tokens + vocab_size)

    print("Saving token frequency tensor...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    torch.save(
        {
            "freq": freq_dict,                 # shape: [vocab_size]
            "total_tokens": total_tokens,
            "vocab_size": vocab_size,
            "num_docs": args.num_docs,
            "model_name": args.model_name_or_path,
        },
        args.output_path
    )

    print(f"Saved to {args.output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
