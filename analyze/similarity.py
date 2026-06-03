import os
import json
import argparse
from typing import List

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_instructions(jsonl_path: str) -> List[str]:
    """
    Load instruction text from a coreset jsonl file.

    Expected format per line:
    {
      "id": "...",
      "conversations": [
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}
      ],
      ...
    }

    We use the first human turn as the instruction text.
    """
    instructions = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skip line {line_idx}: JSON decode error: {e}")
                continue

            convs = obj.get("conversations", [])
            instruction = None

            for turn in convs:
                if turn.get("from") == "human":
                    instruction = turn.get("value", "").strip()
                    break

            if instruction:
                instructions.append(instruction)

    return instructions


def encode_texts(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """
    Encode texts into sentence embeddings.

    If normalize_embeddings=True, cosine similarity between two vectors u, v
    is simply dot(u, v).
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings


def compute_nn_redundancy_metrics(
    embeddings: np.ndarray,
    thresholds: List[float],
    block_size: int = 2048,
):
    """
    Compute:
      A. Mean nearest-neighbor similarity
      B. Redundancy@tau for multiple thresholds

    Since embeddings are normalized, cosine similarity matrix is:
        sim_matrix = embeddings @ embeddings.T

    To avoid self-match, we set diagonal similarities to -inf (or a very small number).
    For memory efficiency, we compute row blocks against the full matrix.

    Args:
        embeddings: np.ndarray of shape [N, D], normalized
        thresholds: list of tau values, e.g. [0.8, 0.85, 0.9]
        block_size: number of rows per block when computing similarities

    Returns:
        nn_sims: np.ndarray of shape [N], nearest-neighbor similarity for each sample
        mean_nn_similarity: float
        redundancy_at_tau: dict {tau: ratio}
    """
    num_items = embeddings.shape[0]
    if num_items < 2:
        raise ValueError("Need at least 2 samples to compute nearest-neighbor redundancy.")

    nn_sims = np.empty(num_items, dtype=np.float32)

    for start in tqdm(range(0, num_items, block_size), desc="Computing nearest-neighbor similarities"):
        end = min(start + block_size, num_items)
        block = embeddings[start:end]  # [B, D]

        # Similarity between current block and all embeddings
        sim = np.matmul(block, embeddings.T)  # [B, N]

        # Remove self-similarity
        row_indices = np.arange(end - start)
        col_indices = np.arange(start, end)
        sim[row_indices, col_indices] = -1.0  # cosine similarity lower than any valid self-match after normalization

        # Nearest-neighbor similarity for each row in this block
        nn_sims[start:end] = sim.max(axis=1)

    mean_nn_similarity = float(nn_sims.mean())

    redundancy_at_tau = {}
    for tau in thresholds:
        ratio = float((nn_sims > tau).mean())
        redundancy_at_tau[tau] = ratio

    return nn_sims, mean_nn_similarity, redundancy_at_tau


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean nearest-neighbor similarity and Redundancy@tau from coreset.jsonl"
    )
    parser.add_argument(
        "--input",
        type=str,
        default='/SSD/zpf/atcs/coreset/llama/instag_id/key/ufs_ufs/coreset.jsonl',
        help="Path to coreset jsonl file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local model path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Encoding batch size",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
        help="Block size for nearest-neighbor similarity computation",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.8],
        help="Threshold(s) for Redundancy@tau, e.g. --thresholds 0.8 0.85 0.9",
    )
    parser.add_argument(
        "--save_embeddings",
        type=str,
        default="",
        help="Optional path to save embeddings as .npy",
    )
    parser.add_argument(
        "--save_nn_sims",
        type=str,
        default="",
        help="Optional path to save nearest-neighbor similarities as .npy",
    )

    args = parser.parse_args()

    print(f"Loading instructions from: {args.input}")
    instructions = load_instructions(args.input)
    print(f"Loaded {len(instructions)} instructions")

    if len(instructions) < 2:
        raise ValueError("Not enough valid instructions found in input file.")

    print(f"Encoding with model: {args.model}")
    embeddings = encode_texts(
        texts=instructions,
        model_name=args.model,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    if args.save_embeddings:
        if os.path.dirname(args.save_embeddings):
            os.makedirs(os.path.dirname(args.save_embeddings), exist_ok=True)
        np.save(args.save_embeddings, embeddings)
        print(f"Saved embeddings to: {args.save_embeddings}")

    nn_sims, mean_nn_similarity, redundancy_at_tau = compute_nn_redundancy_metrics(
        embeddings=embeddings,
        thresholds=args.thresholds,
        block_size=args.block_size,
    )

    if args.save_nn_sims:
        if os.path.dirname(args.save_nn_sims):
            os.makedirs(os.path.dirname(args.save_nn_sims), exist_ok=True)
        np.save(args.save_nn_sims, nn_sims)
        print(f"Saved nearest-neighbor similarities to: {args.save_nn_sims}")

    print("\n===== Redundancy Metrics =====")
    print(f"A. Mean nearest-neighbor similarity: {mean_nn_similarity:.6f}  (higher = more redundant)")

    print("B. Redundancy@τ")
    for tau in args.thresholds:
        print(f"   Redundancy@{tau}: {redundancy_at_tau[tau]:.6f}  (higher = more redundant)")


if __name__ == "__main__":
    main()