import argparse
import csv
import json
import os
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize NLL loss and token retention across attention-threshold rebuilding."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="analysis/attn_threshold_loss_sweep.jsonl",
        help="JSONL produced by find_key_sentence/attn_threshold_loss_sweep.py.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="analysis/attn_threshold_token_diff_summary.json",
        help="Where to write aggregate JSON summary.",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="analysis/attn_threshold_token_diff_summary.csv",
        help="Where to write aggregate CSV summary. Use an empty string to disable.",
    )
    parser.add_argument(
        "--markdown_file",
        type=str,
        default="analysis/attn_threshold_nll_token_table.md",
        help="Where to write a markdown table. Use an empty string to disable.",
    )
    parser.add_argument(
        "--per_example_file",
        type=str,
        default="",
        help="Optional JSONL path for per-example token differences.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Expected threshold list and output order.",
    )
    parser.add_argument(
        "--baseline",
        choices=["sent_lens", "threshold_1", "max_threshold"],
        default="sent_lens",
        help=(
            "How to estimate the before-rebuild token count when the sweep file does not "
            "store original_x_token_count_loss_model. sent_lens is the tokenized sentence "
            "length sum used by the attention sweep."
        ),
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def basic_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "p10": None,
            "p90": None,
            "max": None,
        }
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def threshold_key(value: Any) -> str:
    return f"{float(value):.1f}"


def parse_threshold_list(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [threshold_key(x.strip()) for x in raw.split(",") if x.strip()]


def get_baseline_count(record: Dict[str, Any], baseline: str) -> Optional[int]:
    if "original_x_token_count_loss_model" in record:
        return int(record["original_x_token_count_loss_model"])

    threshold_results = record.get("threshold_results", [])

    if baseline == "sent_lens" and record.get("sent_lens"):
        return int(sum(record["sent_lens"]))

    if baseline == "threshold_1":
        for item in threshold_results:
            if float(item["threshold"]) == 1.0:
                return int(item["x_token_count_loss_model"])

    if baseline == "max_threshold" and threshold_results:
        item = max(threshold_results, key=lambda x: float(x["threshold"]))
        return int(item["x_token_count_loss_model"])

    return None


def summarize(per_example_rows: List[Dict[str, Any]], threshold_order: List[str]) -> Dict[str, Any]:
    thresholds = threshold_order or sorted({row["threshold"] for row in per_example_rows}, key=float)
    threshold_stats: Dict[str, Dict[str, Any]] = {}

    for threshold in thresholds:
        rows = [row for row in per_example_rows if row["threshold"] == threshold]
        before_counts = [row["before_tokens"] for row in rows]
        after_counts = [row["after_tokens"] for row in rows]
        token_reductions = [row["token_reduction"] for row in rows]
        keep_ratios = [row["keep_ratio"] for row in rows]
        keep_pcts = [row["keep_pct"] for row in rows]
        kept_sentences = [row["num_kept_sentences"] for row in rows]
        original_nlls = [row["original_nll"] for row in rows if row["original_nll"] is not None]
        rebuilt_nlls = [row["rebuilt_nll"] for row in rows if row["rebuilt_nll"] is not None]
        nll_deltas = [row["nll_delta"] for row in rows if row["nll_delta"] is not None]

        threshold_stats[threshold] = {
            "count": len(rows),
            "before_tokens": basic_stats(before_counts),
            "after_tokens": basic_stats(after_counts),
            "token_reduction": basic_stats(token_reductions),
            "keep_ratio": basic_stats(keep_ratios),
            "keep_pct": basic_stats(keep_pcts),
            "num_kept_sentences": basic_stats(kept_sentences),
            "original_nll": basic_stats(original_nlls),
            "rebuilt_nll": basic_stats(rebuilt_nlls),
            "nll_delta": basic_stats(nll_deltas),
            "total_before_tokens": sum(before_counts),
            "total_after_tokens": sum(after_counts),
            "total_token_reduction": sum(token_reductions),
            "total_keep_ratio": sum(after_counts) / sum(before_counts) if sum(before_counts) else None,
            "total_keep_pct": (sum(after_counts) / sum(before_counts) * 100.0) if sum(before_counts) else None,
        }

    return {
        "num_rows": len(per_example_rows),
        "num_examples": len({row["idx"] for row in per_example_rows}),
        "thresholds": thresholds,
        "threshold_stats": threshold_stats,
    }


def write_summary_csv(path: str, summary: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    fields = [
        "threshold",
        "count",
        "original_nll_mean",
        "rebuilt_nll_mean",
        "nll_delta_mean",
        "before_tokens_mean",
        "after_tokens_mean",
        "token_reduction_mean",
        "keep_ratio_mean",
        "keep_pct_mean",
        "total_before_tokens",
        "total_after_tokens",
        "total_token_reduction",
        "total_keep_ratio",
        "total_keep_pct",
        "num_kept_sentences_mean",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for threshold in summary["thresholds"]:
            stats = summary["threshold_stats"][threshold]
            writer.writerow(
                {
                    "threshold": threshold,
                    "count": stats["count"],
                    "original_nll_mean": stats["original_nll"]["mean"],
                    "rebuilt_nll_mean": stats["rebuilt_nll"]["mean"],
                    "nll_delta_mean": stats["nll_delta"]["mean"],
                    "before_tokens_mean": stats["before_tokens"]["mean"],
                    "after_tokens_mean": stats["after_tokens"]["mean"],
                    "token_reduction_mean": stats["token_reduction"]["mean"],
                    "keep_ratio_mean": stats["keep_ratio"]["mean"],
                    "keep_pct_mean": stats["keep_pct"]["mean"],
                    "total_before_tokens": stats["total_before_tokens"],
                    "total_after_tokens": stats["total_after_tokens"],
                    "total_token_reduction": stats["total_token_reduction"],
                    "total_keep_ratio": stats["total_keep_ratio"],
                    "total_keep_pct": stats["total_keep_pct"],
                    "num_kept_sentences_mean": stats["num_kept_sentences"]["mean"],
                }
            )


def fmt(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}{suffix}"


def build_markdown_lines(summary: Dict[str, Any]) -> List[str]:
    lines = [
        "| threshold | count | original_nll_mean | rebuilt_nll_mean | nll_delta_mean | token_keep_pct_mean | token_keep_pct_total |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in summary["thresholds"]:
        stats = summary["threshold_stats"][threshold]
        lines.append(
            "| {threshold} | {count} | {original_nll} | {rebuilt_nll} | {delta_nll} | {keep_pct_mean} | {keep_pct_total} |".format(
                threshold=threshold,
                count=stats["count"],
                original_nll=fmt(stats["original_nll"]["mean"], digits=4),
                rebuilt_nll=fmt(stats["rebuilt_nll"]["mean"], digits=4),
                delta_nll=fmt(stats["nll_delta"]["mean"], digits=4),
                keep_pct_mean=fmt(stats["keep_pct"]["mean"], digits=2, suffix="%"),
                keep_pct_total=fmt(stats["total_keep_pct"], digits=2, suffix="%"),
            )
        )
    return lines


def write_markdown_table(path: str, summary: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(build_markdown_lines(summary)) + "\n")


def print_table(summary: Dict[str, Any]) -> None:
    for line in build_markdown_lines(summary):
        print(line)


def main() -> None:
    args = parse_args()
    threshold_order = parse_threshold_list(args.thresholds)
    per_example_rows: List[Dict[str, Any]] = []
    skipped = 0

    for record in iter_jsonl(args.input_file):
        before_tokens = get_baseline_count(record, args.baseline)
        if before_tokens is None or before_tokens <= 0:
            skipped += 1
            continue

        for item in record.get("threshold_results", []):
            after_tokens = int(item["x_token_count_loss_model"])
            threshold = threshold_key(item["threshold"])
            token_reduction = before_tokens - after_tokens
            original_nll = record.get("original_nll")
            rebuilt_nll = item.get("nll")
            per_example_rows.append(
                {
                    "idx": record["idx"],
                    "threshold": threshold,
                    "before_tokens": before_tokens,
                    "after_tokens": after_tokens,
                    "token_reduction": token_reduction,
                    "token_delta_after_minus_before": after_tokens - before_tokens,
                    "keep_ratio": after_tokens / before_tokens,
                    "keep_pct": after_tokens / before_tokens * 100.0,
                    "num_kept_sentences": int(item.get("num_kept_sentences", len(item.get("keep_indices", [])))),
                    "original_nll": None if original_nll is None else float(original_nll),
                    "rebuilt_nll": None if rebuilt_nll is None else float(rebuilt_nll),
                    "nll_delta": None if original_nll is None or rebuilt_nll is None else float(rebuilt_nll) - float(original_nll),
                }
            )

    summary = summarize(per_example_rows, threshold_order)
    summary["input_file"] = args.input_file
    summary["baseline"] = args.baseline
    summary["requested_thresholds"] = threshold_order
    summary["skipped_examples"] = skipped

    ensure_parent_dir(args.output_file)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.csv_file:
        write_summary_csv(args.csv_file, summary)

    if args.markdown_file:
        write_markdown_table(args.markdown_file, summary)

    if args.per_example_file:
        ensure_parent_dir(args.per_example_file)
        with open(args.per_example_file, "w", encoding="utf-8") as f:
            for row in per_example_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_table(summary)
    print(f"\nWrote summary JSON to: {args.output_file}")
    if args.csv_file:
        print(f"Wrote summary CSV to: {args.csv_file}")
    if args.markdown_file:
        print(f"Wrote markdown table to: {args.markdown_file}")
    if args.per_example_file:
        print(f"Wrote per-example JSONL to: {args.per_example_file}")


if __name__ == "__main__":
    main()
