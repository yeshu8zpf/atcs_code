import os
import json
import math
import random
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np


def load_jsonl_as_dict(path: str, idx_key: str = "idx") -> Dict[Any, Dict[str, Any]]:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] skip invalid json line {line_no} in {path}")
                continue

            idx = obj.get(idx_key, obj.get("id"))
            if idx is None:
                print(f"[warn] skip line {line_no} in {path}: missing idx/id")
                continue

            data[idx] = obj
    return data


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """
    Similar to scipy.stats.rankdata(method='average').
    Smaller value -> smaller rank.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)

    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1

        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    x_center = x - x_mean
    y_center = y - y_mean
    denom = np.sqrt((x_center ** 2).sum() * (y_center ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((x_center * y_center).sum() / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    return pearson_corr(rx, ry)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """
    O(n^2) implementation.
    Fine for a few thousand points; if your overlap is huge, consider sampling.
    """
    n = len(x)
    if n < 2:
        return float("nan")

    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0

    for i in range(n):
        dx = x[i] - x[i + 1 :]
        dy = y[i] - y[i + 1 :]

        sx = np.sign(dx)
        sy = np.sign(dy)

        both_nonzero = (sx != 0) & (sy != 0)
        concordant += int(np.sum(sx[both_nonzero] == sy[both_nonzero]))
        discordant += int(np.sum(sx[both_nonzero] != sy[both_nonzero]))

        tie_x += int(np.sum((sx == 0) & (sy != 0)))
        tie_y += int(np.sum((sx != 0) & (sy == 0)))
        # ties in both are ignored in tau-b numerator and denominator adjustments

    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def get_topk_ids(
    items: List[Tuple[Any, float]],
    k: int,
    larger_is_better: bool,
) -> List[Any]:
    items_sorted = sorted(items, key=lambda t: t[1], reverse=larger_is_better)
    return [idx for idx, _ in items_sorted[:k]]


def pairwise_agreement(
    x: np.ndarray,
    y: np.ndarray,
    num_pairs: int = 100000,
    seed: int = 42,
) -> float:
    n = len(x)
    if n < 2:
        return float("nan")

    rng = random.Random(seed)
    agree = 0
    valid = 0

    max_pairs = n * (n - 1) // 2
    if max_pairs <= num_pairs:
        for i in range(n):
            for j in range(i + 1, n):
                sx = np.sign(x[i] - x[j])
                sy = np.sign(y[i] - y[j])
                if sx == 0 or sy == 0:
                    continue
                valid += 1
                if sx == sy:
                    agree += 1
    else:
        for _ in range(num_pairs):
            i, j = rng.sample(range(n), 2)
            if i == j:
                continue
            sx = np.sign(x[i] - x[j])
            sy = np.sign(y[i] - y[j])
            if sx == 0 or sy == 0:
                continue
            valid += 1
            if sx == sy:
                agree += 1

    if valid == 0:
        return float("nan")
    return agree / valid


def compare_scores(
    file1: str,
    file2: str,
    score_key1: str,
    score_key2: str,
    larger_is_better1: bool,
    larger_is_better2: bool,
    topk_list: List[int],
    pairwise_num_pairs: int,
    export_aligned_path: str = "",
    metrics_output_path: str = "",
):
    data1 = load_jsonl_as_dict(file1)
    data2 = load_jsonl_as_dict(file2)

    ids1 = set(data1.keys())
    ids2 = set(data2.keys())
    common_ids = sorted(ids1 & ids2)

    print(f"file1 samples: {len(ids1)}")
    print(f"file2 samples: {len(ids2)}")
    print(f"common samples: {len(common_ids)}")
    print(f"only in file1: {len(ids1 - ids2)}")
    print(f"only in file2: {len(ids2 - ids1)}")

    aligned = []
    missing_key1 = 0
    missing_key2 = 0

    for idx in common_ids:
        obj1 = data1[idx]
        obj2 = data2[idx]

        if score_key1 not in obj1:
            missing_key1 += 1
            continue
        if score_key2 not in obj2:
            missing_key2 += 1
            continue

        try:
            s1 = float(obj1[score_key1])
            s2 = float(obj2[score_key2])
        except Exception:
            continue

        aligned.append((idx, s1, s2))

    print(f"usable aligned samples: {len(aligned)}")
    if missing_key1 > 0:
        raise KeyError(f"[warn] common samples missing '{score_key1}' in file1: {missing_key1}")
    if missing_key2 > 0:
        raise KeyError(f"[warn] common samples missing '{score_key2}' in file2: {missing_key2}")

    if len(aligned) < 2:
        print("Not enough aligned samples to compare.")
        return

    ids = [x[0] for x in aligned]
    scores1 = np.array([x[1] for x in aligned], dtype=float)
    scores2 = np.array([x[2] for x in aligned], dtype=float)

    comp1 = scores1.copy()
    comp2 = scores2.copy()

    if larger_is_better1:
        comp1 = -comp1
    if larger_is_better2:
        comp2 = -comp2

    sp = spearman_corr(comp1, comp2)
    print(f"Spearman: {sp:.6f}")

    kt = None
    if len(aligned) <= 5000:
        kt = kendall_tau_b(comp1, comp2)
        print(f"Kendall tau-b: {kt:.6f}")
    else:
        print("Kendall tau-b: skipped because aligned sample size > 5000")

    p_agree = pairwise_agreement(comp1, comp2, num_pairs=pairwise_num_pairs)
    print(f"Pairwise agreement: {p_agree:.6f}")

    items1 = list(zip(ids, comp1.tolist()))
    items2 = list(zip(ids, comp2.tolist()))

    topk_overlap = {}
    for k in topk_list:
        if k <= 0:
            continue
        kk = min(k, len(aligned))
        top1 = set(get_topk_ids(items1, kk, larger_is_better=False))
        top2 = set(get_topk_ids(items2, kk, larger_is_better=False))
        overlap = len(top1 & top2) / kk
        topk_overlap[str(kk)] = overlap
        print(f"Top-{kk} overlap: {overlap:.6f}")

    if export_aligned_path:
        os.makedirs(os.path.dirname(export_aligned_path) or ".", exist_ok=True)
        with open(export_aligned_path, "w", encoding="utf-8") as f:
            for idx, s1, s2 in aligned:
                f.write(json.dumps({
                    "idx": idx,
                    score_key1: s1,
                    score_key2: s2,
                }, ensure_ascii=False) + "\n")
        print(f"Aligned scores written to: {export_aligned_path}")

    metrics = {
        "file1": file1,
        "file2": file2,
        "score_key1": score_key1,
        "score_key2": score_key2,
        "larger_is_better1": larger_is_better1,
        "larger_is_better2": larger_is_better2,
        "file1_samples": len(ids1),
        "file2_samples": len(ids2),
        "common_samples": len(common_ids),
        "only_in_file1": len(ids1 - ids2),
        "only_in_file2": len(ids2 - ids1),
        "usable_aligned_samples": len(aligned),
        "missing_key1": missing_key1,
        "missing_key2": missing_key2,
        "spearman": sp,
        "kendall_tau_b": kt,
        "pairwise_agreement": p_agree,
        "pairwise_num_pairs": pairwise_num_pairs,
        "topk_overlap": topk_overlap,
    }

    if metrics_output_path:
        os.makedirs(os.path.dirname(metrics_output_path) or ".", exist_ok=True)
        with open(metrics_output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Metrics written to: {metrics_output_path}")

    return metrics


def parse_topk_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, default="score/llama/xsota/ifd_all/score.jsonl")
    parser.add_argument("--file2", type=str, required="score/llama/xsota_fine_for_topk/ifd_score.jsonl")

    parser.add_argument("--score_key", type=str, default="ifd")
    # For NLL: usually smaller is better -> set False
    # For IFD/UFS: usually larger or smaller depends on your selection convention
    parser.add_argument("--larger_is_better1", action="store_true")
    parser.add_argument("--larger_is_better2", action="store_true")

    parser.add_argument("--topk", type=str, default="1000,2000,5000")
    parser.add_argument("--pairwise_num_pairs", type=int, default=100000)
    parser.add_argument("--export_aligned_path", type=str, default="")
    parser.add_argument("--metrics_output_path", type=str, default="analyze/results/ifd_sort_compare.json")
    parser.set_defaults(larger_is_better1=True,
                        larger_is_better2=True)

    args = parser.parse_args()
    args.score_key1 = args.score_key
    args.score_key2 = args.score_key

    compare_scores(
        file1=args.file1,
        file2=args.file2,
        score_key1=args.score_key1,
        score_key2=args.score_key2,
        larger_is_better1=args.larger_is_better1,
        larger_is_better2=args.larger_is_better2,
        topk_list=parse_topk_list(args.topk),
        pairwise_num_pairs=args.pairwise_num_pairs,
        export_aligned_path=args.export_aligned_path,
    )


if __name__ == "__main__":
    main()