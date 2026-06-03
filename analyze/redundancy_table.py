import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


UTILITY_ORDER = ["NLL", "IFD", "UFS"]
METHOD_ORDER = ["ATCS", "Fine", "Coarse"]
METHOD_TO_DIR = {
    "ATCS": ("key", True),
    "Fine": ("fine", False),
    "Coarse": ("coarse", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean nearest-neighbor similarity and Redundancy@tau "
            "for the standard NLL/IFD/UFS x ATCS/Fine/Coarse coreset grid."
        )
    )
    parser.add_argument("--group_model", type=str, default="pythia_qwen")
    parser.add_argument("--group_dataset", type=str, default="tulu3")
    parser.add_argument(
        "--input_root",
        type=str,
        default="",
        help="Optional override for the coreset root directory.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load the embedding model only from local files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Redundancy@tau threshold.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument(
        "--output_json",
        type=str,
        default="analysis/results/redundancy_table.json",
    )
    parser.add_argument(
        "--output_markdown",
        type=str,
        default="analysis/results/redundancy_table.md",
    )
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_input_root(args: argparse.Namespace) -> Path:
    if args.input_root:
        return Path(args.input_root)
    return Path("coreset") / args.group_model / args.group_dataset


def resolve_grid_paths(input_root: Path) -> Dict[str, Dict[str, Path]]:
    grid: Dict[str, Dict[str, Path]] = {}
    for utility in UTILITY_ORDER:
        utility_lower = utility.lower()
        grid[utility] = {}
        for method in METHOD_ORDER:
            method_dir, duplicated_name = METHOD_TO_DIR[method]
            if duplicated_name:
                path = input_root / method_dir / f"{utility_lower}_{utility_lower}" / "coreset.jsonl"
            else:
                path = input_root / method_dir / utility_lower / "coreset.jsonl"
            grid[utility][method] = path
    return grid


def load_instructions(jsonl_path: Path) -> List[str]:
    instructions: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_idx}: {exc}") from exc

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
    batch_size: int,
    local_files_only: bool,
) -> np.ndarray:
    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def compute_nn_redundancy_metrics(
    embeddings: np.ndarray,
    threshold: float,
    block_size: int,
) -> Tuple[np.ndarray, float, float]:
    num_items = embeddings.shape[0]
    if num_items < 2:
        raise ValueError("Need at least 2 samples to compute nearest-neighbor redundancy.")

    nn_sims = np.empty(num_items, dtype=np.float32)

    for start in tqdm(range(0, num_items, block_size), desc="Computing nearest-neighbor similarities"):
        end = min(start + block_size, num_items)
        block = embeddings[start:end]
        sim = np.matmul(block, embeddings.T)
        row_indices = np.arange(end - start)
        col_indices = np.arange(start, end)
        sim[row_indices, col_indices] = -1.0
        nn_sims[start:end] = sim.max(axis=1)

    mean_nn_similarity = float(nn_sims.mean())
    redundancy = float((nn_sims > threshold).mean())
    return nn_sims, mean_nn_similarity, redundancy


def build_markdown(
    results: Dict[str, Dict[str, Dict[str, float]]],
    threshold: float,
    group_model: str,
    group_dataset: str,
) -> str:
    lines = []
    lines.append(f"## {group_model} / {group_dataset} Redundancy")
    lines.append("")
    lines.append(f"| Utility | Metric | {' | '.join(METHOD_ORDER)} |")
    lines.append("|---|---|---:|---:|---:|")
    for utility in UTILITY_ORDER:
        vals = results[utility]
        lines.append(
            "| {utility} | Mean NN Sim. | {atcs:.3f} | {fine:.3f} | {coarse:.3f} |".format(
                utility=utility,
                atcs=vals["ATCS"]["mean_nn_similarity"],
                fine=vals["Fine"]["mean_nn_similarity"],
                coarse=vals["Coarse"]["mean_nn_similarity"],
            )
        )
        lines.append(
            "| {utility} | Redundancy@{tau:.1f} | {atcs:.1f}% | {fine:.1f}% | {coarse:.1f}% |".format(
                utility=utility,
                tau=threshold,
                atcs=vals["ATCS"]["redundancy"] * 100.0,
                fine=vals["Fine"]["redundancy"] * 100.0,
                coarse=vals["Coarse"]["redundancy"] * 100.0,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_root = resolve_input_root(args)
    grid_paths = resolve_grid_paths(input_root)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for utility in UTILITY_ORDER:
        results[utility] = {}
        for method in METHOD_ORDER:
            path = grid_paths[utility][method]
            if not path.exists():
                raise FileNotFoundError(f"Missing coreset file: {path}")

            print(f"\n=== {utility} / {method} ===")
            print(f"Input: {path}")
            instructions = load_instructions(path)
            print(f"Loaded {len(instructions)} instructions")

            embeddings = encode_texts(
                texts=instructions,
                model_name=args.embedding_model,
                batch_size=args.batch_size,
                local_files_only=args.local_files_only,
            )
            _, mean_nn_similarity, redundancy = compute_nn_redundancy_metrics(
                embeddings=embeddings,
                threshold=args.threshold,
                block_size=args.block_size,
            )

            results[utility][method] = {
                "num_items": len(instructions),
                "mean_nn_similarity": mean_nn_similarity,
                "redundancy": redundancy,
            }

            print(f"Mean NN Sim.: {mean_nn_similarity:.6f}")
            print(f"Redundancy@{args.threshold}: {redundancy:.6f}")

    payload = {
        "scope": {
            "group_model": args.group_model,
            "group_dataset": args.group_dataset,
            "input_root": str(input_root),
        },
        "embedding_model": args.embedding_model,
        "local_files_only": args.local_files_only,
        "threshold": args.threshold,
        "results": results,
    }

    ensure_parent(args.output_json)
    ensure_parent(args.output_markdown)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    markdown = build_markdown(
        results=results,
        threshold=args.threshold,
        group_model=args.group_model,
        group_dataset=args.group_dataset,
    )
    with open(args.output_markdown, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\nSaved JSON to: {args.output_json}")
    print(f"Saved Markdown to: {args.output_markdown}")
    print()
    print(markdown)


if __name__ == "__main__":
    main()
