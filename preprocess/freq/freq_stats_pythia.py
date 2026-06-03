import os
import json
import argparse
from collections import Counter
from itertools import islice
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='models/pythia-70m'
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        default=15,
        help="Number of C4 documents used as reference corpus (paper uses 15)"
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default="freqs/C4/*.json.gz",
        help="Glob pattern or file path for the reference C4 JSON/JSON.GZ files"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="freqs/.hf_datasets_cache",
        help="Cache directory used by datasets.load_dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="freqs/c4_pythia_freq.pt",
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

    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")

    print(f"Loading reference corpus from: {args.data_files}")
    dataset = load_dataset(
        "json",
        data_files=args.data_files,
        split="train",
        cache_dir=args.cache_dir,
        streaming=True,
    )

    docs = list(islice(dataset, args.num_docs))
    usable_docs = len(docs)
    print(f"Using first {usable_docs} documents from reference corpus")

    token_counter = Counter()
    total_tokens = 0

    for doc in tqdm(docs, desc="Processing C4 documents"):
        text = doc["text"]

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
            "used_docs": usable_docs,
            "data_files": args.data_files,
            "cache_dir": args.cache_dir,
            "model_name": args.model_name_or_path,
        },
        args.output_path
    )

    print(f"Saved to {args.output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
