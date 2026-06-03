import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = AXIS_LABEL_FONTSIZE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot threshold-vs-NLL summary from attn_threshold_loss_summary.json."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="analysis/attn_threshold_loss_summary.json",
        help="Path to the summary JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/attn_threshold_loss_plot.png",
        help="Path to the output image.",
    )
    parser.add_argument(
        "--pdf-output",
        type=str,
        default="analysis/attn_threshold_loss_plot.pdf",
        help="Optional PDF output path.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_summary(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def padded_limits(values, pad_ratio: float = 0.2):
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        pad = max(abs(vmin) * pad_ratio, 1.0)
        return vmin - pad, vmax + pad
    span = vmax - vmin
    pad = span * pad_ratio
    return vmin - pad, vmax + pad


def main():
    args = parse_args()
    summary = load_summary(args.input)

    stats = summary["threshold_stats"]
    thresholds = [float(x) for x in summary["thresholds"] if 0.5 <= float(x) <= 1.0]
    if not thresholds:
        raise ValueError("No thresholds found in the range [0.5, 1.0].")
    mean_nlls = [float(stats[str(t)]["mean_nll"]) for t in thresholds]
    mean_token_keep_pcts = []
    for t in thresholds:
        item = stats[str(t)]
        keep_pct = item.get("mean_token_keep_pct")
        if keep_pct is None:
            keep_pct = item.get("total_token_keep_pct")
        if keep_pct is None:
            raise ValueError(
                "Missing token_keep_pct in summary. Re-run "
                "find_key_sentence/attn_threshold_loss_sweep.py to regenerate the summary."
            )
        mean_token_keep_pcts.append(float(keep_pct))
    ensure_parent_dir(args.output)
    ensure_parent_dir(args.pdf_output)

    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    ax2 = ax1.twinx()

    loss_line = ax1.plot(
        thresholds,
        mean_nlls,
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="loss",
    )[0]
    token_line = ax2.plot(
        thresholds,
        mean_token_keep_pcts,
        color="#59a14f",
        marker="s",
        linewidth=2,
        label="token_keep_pct",
    )[0]

    ax1.set_xlabel("Attention threshold", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_ylabel("loss", fontsize=AXIS_LABEL_FONTSIZE, color=loss_line.get_color())
    ax2.set_ylabel("token_keep_pct", fontsize=AXIS_LABEL_FONTSIZE, color=token_line.get_color())
    ax1.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    ax1.tick_params(axis="y", labelcolor=loss_line.get_color(), labelsize=TICK_LABEL_FONTSIZE)
    ax2.tick_params(axis="y", labelcolor=token_line.get_color(), labelsize=TICK_LABEL_FONTSIZE)
    ax1.set_ylim(*padded_limits(mean_nlls))
    ax2.set_ylim(*padded_limits(mean_token_keep_pcts))
    ax1.grid(True, axis="both", alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    tick_labels = [f"{t:.1f}" for t in thresholds]
    ax1.set_xticks(thresholds)
    ax1.set_xticklabels(tick_labels)

    handles = [loss_line, token_line]
    labels = [h.get_label() for h in handles]
    ax1.legend(
        handles,
        labels,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    if args.pdf_output:
        plt.savefig(args.pdf_output, bbox_inches="tight")

    print(f"Saved plot to: {args.output}")
    if args.pdf_output:
        print(f"Saved plot to: {args.pdf_output}")


if __name__ == "__main__":
    main()
