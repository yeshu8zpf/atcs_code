import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

METRIC_ORDER = ["nll", "ifd", "ufs"]
METHOD_ORDER = ["coarse", "fine", "key"]
METHOD_LABELS = {
    "coarse": "Coarse",
    "fine": "Fine",
    "key": "ATCS",
}
TASK_ORDER = ["arc", "bbh", "ifeval", "mmlu"]
TASK_LABELS = {
    "arc": "ARC",
    "bbh": "BBH",
    "ifeval": "IFEVAL",
    "mmlu": "MMLU",
}
MIX_TABLE_HOMOGENEOUS = ["nll", "ifd", "ufs"]
MIX_TABLE_MIXED = ["ifd_nll", "ifd_ufs", "nll_ifd", "nll_ufs", "ufs_ifd", "ufs_nll"]
MIX_TABLE_LABELS = {
    "nll": "NLL",
    "ifd": "IFD",
    "ufs": "UFS",
    "ifd_nll": "IFD-NLL",
    "ifd_ufs": "IFD-UFS",
    "nll_ifd": "NLL-IFD",
    "nll_ufs": "NLL-UFS",
    "ufs_ifd": "UFS-IFD",
    "ufs_nll": "UFS-NLL",
}


def parse_key(raw_key: str) -> dict:
    parts = [part.strip() for part in re.split(r"\s{2,}", raw_key.strip()) if part.strip()]
    if len(parts) < 5:
        parts = raw_key.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Unable to parse key: {raw_key}")

    model, dataset, task, method = parts[:4]
    run_name = " ".join(parts[4:])

    return {
        "raw_key": raw_key,
        "model": model,
        "dataset": dataset,
        "task": task,
        "method": method,
        "run_name": run_name,
    }


def shorten_run_name(run_name: str) -> str:
    return run_name.split("_lr", 1)[0]


def normalize_metric(run_name: str) -> str:
    short_name = shorten_run_name(run_name).lower()
    if short_name in METRIC_ORDER:
        return short_name

    parts = [part for part in short_name.split("_") if part]
    if len(parts) == 2 and parts[0] == parts[1] and parts[0] in METRIC_ORDER:
        return parts[0]

    return short_name


def format_score(score: float) -> str:
    return f"{float(score):.2f}"


def format_title(text: str) -> str:
    return text.replace("_", " ").title()


def order_tasks(tasks) -> list:
    known = [task for task in TASK_ORDER if task in tasks]
    extra = sorted(task for task in tasks if task not in TASK_ORDER)
    return known + extra


def order_metrics(metrics) -> list:
    known = list(METRIC_ORDER)
    extra = sorted(metric for metric in metrics if metric not in METRIC_ORDER)
    return known + extra


def order_groups(groups) -> list:
    return sorted(groups, key=lambda item: (item[0], item[1]))


def build_grouped_scores(records) -> dict:
    grouped = defaultdict(lambda: defaultdict(dict))
    for record in records:
        group_key = (record["model"], record["dataset"])
        task = record["task"]
        metric = record["metric"]
        method = record["method"]
        grouped[group_key][task][(metric, method)] = record["score_value"]
    return grouped


def build_mix_table_scores(records) -> dict:
    grouped = defaultdict(lambda: defaultdict(dict))
    allowed_metrics = set(MIX_TABLE_HOMOGENEOUS + MIX_TABLE_MIXED)

    for record in records:
        if record["method"] != "key":
            continue
        if record["metric"] not in allowed_metrics:
            continue

        group_key = (record["model"], record["dataset"])
        grouped[group_key][record["task"]][record["metric"]] = record["score_value"]

    return grouped


def filter_groups_with_mixed_scores(grouped: dict) -> dict:
    filtered = {}
    mixed_metrics = set(MIX_TABLE_MIXED)

    for group_key, task_map in grouped.items():
        has_mixed = any(any(metric in mixed_metrics for metric in values) for values in task_map.values())
        if has_mixed:
            filtered[group_key] = task_map

    return filtered


def write_detail_csv(records, output_path: Path) -> None:
    fieldnames = [
        "model",
        "dataset",
        "task",
        "method",
        "display_method",
        "metric",
        "run_name",
        "short_run_name",
        "score",
        "raw_key",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "model": record["model"],
                    "dataset": record["dataset"],
                    "task": record["task"],
                    "method": record["method"],
                    "display_method": METHOD_LABELS.get(record["method"], record["method"]),
                    "metric": record["metric"],
                    "run_name": record["run_name"],
                    "short_run_name": shorten_run_name(record["run_name"]),
                    "score": format_score(record["score_value"]),
                    "raw_key": record["raw_key"],
                }
            )


def write_pivot_csv(records, output_path: Path) -> None:
    grouped = build_grouped_scores(records)
    metrics = list(METRIC_ORDER)

    columns = []
    for metric in metrics:
        for method in METHOD_ORDER:
            columns.append(f"{metric}|{METHOD_LABELS[method]}")

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dataset", "benchmark", *columns])

        for group_key in order_groups(grouped):
            task_map = grouped[group_key]
            tasks = order_tasks(task_map.keys())

            for task in tasks:
                row = [group_key[0], group_key[1], TASK_LABELS.get(task, task.upper())]
                values = task_map[task]
                for metric in metrics:
                    for method in METHOD_ORDER:
                        score = values.get((metric, method))
                        row.append("" if score is None else format_score(score))
                writer.writerow(row)


def collect_avg(values: dict, metric: str, method: str) -> float | None:
    scores = []
    for task_values in values.values():
        score = task_values.get((metric, method))
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return sum(scores) / len(scores)


def collect_mix_avg(values: dict, metric: str) -> float | None:
    scores = []
    for task_values in values.values():
        score = task_values.get(metric)
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return sum(scores) / len(scores)


def render_score_cell(score: float | None, highlight: bool = False) -> str:
    if score is None:
        return '<td style="padding:6px 10px;text-align:center;">&nbsp;</td>'

    content = format_score(score)
    if highlight:
        content = f"<strong>{content}</strong>"
    return f'<td style="padding:6px 10px;text-align:center;">{content}</td>'


def build_mix_table_rows(values: dict, tasks: list[str]) -> list[list[float | None]]:
    metrics = MIX_TABLE_HOMOGENEOUS + MIX_TABLE_MIXED
    rows = []

    for task in tasks:
        task_values = values[task]
        rows.append([task_values.get(metric) for metric in metrics])

    rows.append([collect_mix_avg(values, metric) for metric in metrics])
    return rows


def render_standard_table_text(values: dict, metrics: list[str]) -> str:
    rows = []

    for task in order_tasks(values.keys()):
        task_values = values[task]
        row = [TASK_LABELS.get(task, task.upper())]
        for metric in metrics:
            for method in METHOD_ORDER:
                score = task_values.get((metric, method))
                row.append("" if score is None else format_score(score))
        rows.append(row)

    avg_row = ["AVG"]
    for metric in metrics:
        for method in METHOD_ORDER:
            score = collect_avg(values, metric, method)
            avg_row.append("" if score is None else format_score(score))
    rows.append(avg_row)

    benchmark_width = max(len("Benchmark"), *(len(str(row[0])) for row in rows))
    subcol_widths = []
    for metric_idx, _metric in enumerate(metrics):
        for method_idx, method in enumerate(METHOD_ORDER):
            col_values = [METHOD_LABELS[method]]
            for row in rows:
                col_values.append(str(row[1 + metric_idx * len(METHOD_ORDER) + method_idx]))
            subcol_widths.append(max(len(value) for value in col_values))

    def make_border(left: str, mid: str, right: str, benchmark_join: str, metric_join: str) -> str:
        parts = [left, "-" * (benchmark_width + 2), benchmark_join]
        for metric_idx, _metric in enumerate(metrics):
            metric_widths = subcol_widths[metric_idx * len(METHOD_ORDER):(metric_idx + 1) * len(METHOD_ORDER)]
            for method_idx, width in enumerate(metric_widths):
                parts.append("-" * (width + 2))
                if method_idx < len(metric_widths) - 1:
                    parts.append(mid)
            parts.append(metric_join if metric_idx < len(metrics) - 1 else right)
        return "".join(parts)

    top_border = make_border("+", "+", "+", "+", "+")
    row_separator = make_border("+", "+", "+", "+", "+")

    def center_metric(metric: str, metric_idx: int) -> str:
        metric_widths = subcol_widths[metric_idx * len(METHOD_ORDER):(metric_idx + 1) * len(METHOD_ORDER)]
        span_width = sum(metric_widths) + 3 * len(METHOD_ORDER) - 1
        return metric.upper().center(span_width)

    def make_group_border() -> str:
        parts = ["+", "-" * (benchmark_width + 2), "+"]
        for metric_idx, _metric in enumerate(metrics):
            metric_widths = subcol_widths[metric_idx * len(METHOD_ORDER):(metric_idx + 1) * len(METHOD_ORDER)]
            span_width = sum(metric_widths) + 3 * len(METHOD_ORDER) - 1
            parts.append("-" * span_width)
            parts.append("+")
        return "".join(parts)

    group_separator = make_group_border()

    def format_data_row(label: str, data_values: list[str]) -> str:
        parts = [f"| {label.ljust(benchmark_width)} |"]
        for idx, value in enumerate(data_values):
            parts.append(f" {value.ljust(subcol_widths[idx])} ")
            parts.append("|")
        return "".join(parts)

    lines = [top_border]
    metric_header = [f"| {'Benchmark'.ljust(benchmark_width)} |"]
    for metric_idx, metric in enumerate(metrics):
        metric_header.append(center_metric(metric, metric_idx))
        metric_header.append("|")
    lines.append("".join(metric_header))
    lines.append(group_separator)
    subheader = [f"| {' '.ljust(benchmark_width)} |"]
    for metric_idx, _metric in enumerate(metrics):
        for method_idx, method in enumerate(METHOD_ORDER):
            width = subcol_widths[metric_idx * len(METHOD_ORDER) + method_idx]
            subheader.append(f" {METHOD_LABELS[method].ljust(width)} ")
            subheader.append("|")
    lines.append("".join(subheader))
    lines.append(row_separator)
    for row in rows[:-1]:
        lines.append(format_data_row(row[0], row[1:]))
    lines.append(row_separator)
    lines.append(format_data_row(rows[-1][0], rows[-1][1:]))
    lines.append(top_border)

    return "\n".join(lines)


def render_mix_table_text(values: dict, tasks: list[str]) -> str:
    metric_order = MIX_TABLE_HOMOGENEOUS + MIX_TABLE_MIXED
    rows = build_mix_table_rows(values, tasks)
    row_labels = [TASK_LABELS.get(task, task.upper()) for task in tasks] + ["AVG"]

    highlighted_rows = []
    for row_scores in rows:
        available = [score for score in row_scores if score is not None]
        max_score = max(available) if available else None
        highlighted_rows.append(
            [
                (
                    ""
                    if score is None
                    else f"{format_score(score)}*" if max_score is not None and abs(score - max_score) < 1e-9 else format_score(score)
                )
                for score in row_scores
            ]
        )

    benchmark_width = max(len("Benchmark"), *(len(label) for label in row_labels))
    col_widths = []
    for idx, metric in enumerate(metric_order):
        values_for_col = [MIX_TABLE_LABELS[metric]]
        values_for_col.extend(row[idx] for row in highlighted_rows)
        col_widths.append(max(len(value) for value in values_for_col))

    left_metrics = MIX_TABLE_HOMOGENEOUS
    right_metrics = MIX_TABLE_MIXED
    left_width = sum(col_widths[: len(left_metrics)]) + 3 * (len(left_metrics) - 1)
    right_width = sum(col_widths[len(left_metrics):]) + 3 * (len(right_metrics) - 1)

    def format_segment(values: list[str], widths: list[int]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    border = (
        f"{'-' * benchmark_width}-+-"
        f"{'-' * left_width}-+-"
        f"{'-' * right_width}"
    )

    lines = []
    lines.append(
        f"{'Benchmark'.ljust(benchmark_width)} | "
        f"{'Homogeneous'.center(left_width)} | "
        f"{'Mixed Utility Combinations (Stage 1 - Stage 2)'.center(right_width)}"
    )
    lines.append(
        f"{' '.ljust(benchmark_width)} | "
        f"{format_segment([MIX_TABLE_LABELS[m] for m in left_metrics], col_widths[: len(left_metrics)])} | "
        f"{format_segment([MIX_TABLE_LABELS[m] for m in right_metrics], col_widths[len(left_metrics):])}"
    )
    lines.append(border)

    for label, row in zip(row_labels[:-1], highlighted_rows[:-1]):
        lines.append(
            f"{label.ljust(benchmark_width)} | "
            f"{format_segment(row[: len(left_metrics)], col_widths[: len(left_metrics)])} | "
            f"{format_segment(row[len(left_metrics):], col_widths[len(left_metrics):])}"
        )

    lines.append(border)
    avg_row = highlighted_rows[-1]
    lines.append(
        f"{row_labels[-1].ljust(benchmark_width)} | "
        f"{format_segment(avg_row[: len(left_metrics)], col_widths[: len(left_metrics)])} | "
        f"{format_segment(avg_row[len(left_metrics):], col_widths[len(left_metrics):])}"
    )

    return "\n".join(lines)


def render_mix_table_markup(values: dict, tasks: list[str]) -> str:
    rows = build_mix_table_rows(values, tasks)
    metric_order = MIX_TABLE_HOMOGENEOUS + MIX_TABLE_MIXED

    parts = ['<table class="mix-table">', "  <thead>"]
    parts.append("    <tr>")
    parts.append('      <th rowspan="2" class="bench">Benchmark</th>')
    parts.append('      <th colspan="3" class="group split-right">Homogeneous</th>')
    parts.append('      <th colspan="6" class="group">Mixed Utility Combinations (Stage 1 - Stage 2)</th>')
    parts.append("    </tr>")
    parts.append("    <tr>")

    for idx, metric in enumerate(metric_order):
        classes = []
        if idx == len(MIX_TABLE_HOMOGENEOUS) - 1:
            classes.append("split-right")
        class_attr = f' class="{" ".join(classes)}"' if classes else ""
        parts.append(f'      <th{class_attr}>{MIX_TABLE_LABELS[metric]}</th>')

    parts.append("    </tr>")
    parts.append("  </thead>")
    parts.append("  <tbody>")

    for task, row_scores in zip(tasks, rows[:-1]):
        available = [score for score in row_scores if score is not None]
        max_score = max(available) if available else None

        parts.append("    <tr>")
        parts.append(f'      <td class="bench">{TASK_LABELS.get(task, task.upper())}</td>')
        for idx, score in enumerate(row_scores):
            classes = []
            if idx == len(MIX_TABLE_HOMOGENEOUS) - 1:
                classes.append("split-right")
            if score is not None and max_score is not None and abs(score - max_score) < 1e-9:
                classes.append("best")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            content = "&nbsp;" if score is None else format_score(score)
            parts.append(f"      <td{class_attr}>{content}</td>")
        parts.append("    </tr>")

    avg_scores = rows[-1]
    available = [score for score in avg_scores if score is not None]
    max_score = max(available) if available else None

    parts.append('    <tr class="avg-row">')
    parts.append('      <td class="bench"><strong>AVG</strong></td>')
    for idx, score in enumerate(avg_scores):
        classes = []
        if idx == len(MIX_TABLE_HOMOGENEOUS) - 1:
            classes.append("split-right")
        if score is not None and max_score is not None and abs(score - max_score) < 1e-9:
            classes.append("best")
        class_attr = f' class="{" ".join(classes)}"' if classes else ""
        content = "&nbsp;" if score is None else format_score(score)
        parts.append(f"      <td{class_attr}>{content}</td>")
    parts.append("    </tr>")
    parts.append("  </tbody>")
    parts.append("</table>")

    return "\n".join(parts)


def mix_table_style_block() -> str:
    return "\n".join(
        [
            "<style>",
            "  .mix-table { border-collapse: collapse; margin: 12px 0 28px; font-family: 'Times New Roman', serif; border-top: 3px solid #111; border-bottom: 3px solid #111; }",
            "  .mix-table th, .mix-table td { padding: 6px 12px; text-align: center; white-space: nowrap; }",
            "  .mix-table thead tr:first-child th { border-bottom: 1px solid #999; font-size: 16px; }",
            "  .mix-table thead tr:last-child th { border-bottom: 1px solid #111; font-weight: 400; }",
            "  .mix-table .bench { text-align: left; border-right: 1px solid #111; }",
            "  .mix-table .split-right { border-right: 1px solid #111; }",
            "  .mix-table tbody .avg-row td { border-top: 1px solid #111; padding-top: 8px; }",
            "  .mix-table .best { font-weight: 700; }",
            "</style>",
        ]
    )


def write_markdown_table(records, output_path: Path) -> None:
    standard_grouped = build_grouped_scores(records)
    standard_metrics = list(METRIC_ORDER)
    mix_grouped = filter_groups_with_mixed_scores(build_mix_table_scores(records))

    lines = []
    for model, dataset in order_groups(standard_grouped):
        values = standard_grouped[(model, dataset)]
        lines.append(f"## {format_title(model)} / {format_title(dataset)}")
        lines.append("")
        lines.append("```text")
        lines.append(render_standard_table_text(values, standard_metrics))
        lines.append("```")
        lines.append("")

    if mix_grouped:
        lines.append("## Mix Metric")
        lines.append("")
        lines.append("`*` marks the best score in each row.")
        lines.append("")

        for model, dataset in order_groups(mix_grouped):
            values = mix_grouped[(model, dataset)]
            tasks = order_tasks(values.keys())
            lines.append(f"### {format_title(model)} / {format_title(dataset)}")
            lines.append("")
            lines.append("```text")
            lines.append(render_mix_table_text(values, tasks))
            lines.append("```")
            lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_html_table(records, output_path: Path) -> None:
    standard_grouped = build_grouped_scores(records)
    standard_metrics = list(METRIC_ORDER)
    mix_grouped = filter_groups_with_mixed_scores(build_mix_table_scores(records))

    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        "  <title>Recode Tables</title>",
        mix_table_style_block(),
        "  <style>",
        "    body { margin: 24px; color: #111; }",
        "    h2 { margin: 28px 0 12px; font-family: 'Times New Roman', serif; }",
        "    h3 { margin: 20px 0 10px; font-family: 'Times New Roman', serif; }",
        "    table.std-table { border-collapse: collapse; margin: 12px 0 28px; font-family: 'Times New Roman', serif; }",
        "    .std-table th, .std-table td { padding: 6px 12px; text-align: center; }",
        "    .std-table thead th { border-bottom: 1px solid #999; }",
        "    .std-table tbody tr.avg-row td { border-top: 1px solid #999; }",
        "    .std-table td:first-child, .std-table th:first-child { text-align: left; }",
        "  </style>",
        "</head>",
        "<body>",
    ]

    for model, dataset in order_groups(standard_grouped):
        values = standard_grouped[(model, dataset)]
        tasks = order_tasks(values.keys())
        lines.append(f"  <h2>{format_title(model)} / {format_title(dataset)}</h2>")
        lines.append('  <table class="std-table">')
        lines.append("    <thead>")
        lines.append("      <tr>")
        lines.append('        <th rowspan="2">Benchmark</th>')
        for metric in standard_metrics:
            lines.append(f"        <th colspan=\"3\">{metric.upper()}</th>")
        lines.append("      </tr>")
        lines.append("      <tr>")
        for _metric in standard_metrics:
            for method in METHOD_ORDER:
                lines.append(f"        <th>{METHOD_LABELS[method]}</th>")
        lines.append("      </tr>")
        lines.append("    </thead>")
        lines.append("    <tbody>")

        for task in tasks:
            task_values = values[task]
            lines.append("      <tr>")
            lines.append(f"        <td>{TASK_LABELS.get(task, task.upper())}</td>")
            for metric in standard_metrics:
                row_scores = [task_values.get((metric, method)) for method in METHOD_ORDER]
                available = [score for score in row_scores if score is not None]
                max_score = max(available) if available else None

                for score in row_scores:
                    highlight = score is not None and max_score is not None and abs(score - max_score) < 1e-9
                    lines.append("        " + render_score_cell(score, highlight))
            lines.append("      </tr>")

        lines.append('      <tr class="avg-row">')
        lines.append("        <td><strong>AVG</strong></td>")
        for metric in standard_metrics:
            avg_scores = [collect_avg(values, metric, method) for method in METHOD_ORDER]
            available = [score for score in avg_scores if score is not None]
            max_score = max(available) if available else None
            for score in avg_scores:
                highlight = score is not None and max_score is not None and abs(score - max_score) < 1e-9
                lines.append("        " + render_score_cell(score, highlight))
        lines.append("      </tr>")

        lines.append("    </tbody>")
        lines.append("  </table>")

    if mix_grouped:
        lines.append("  <h2>Mix Metric</h2>")
        for model, dataset in order_groups(mix_grouped):
            values = mix_grouped[(model, dataset)]
            tasks = order_tasks(values.keys())
            lines.append(f"  <h3>{format_title(model)} / {format_title(dataset)}</h3>")
            lines.append(render_mix_table_markup(values, tasks))

    lines.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export recode.json to detail and grouped tables.")
    parser.add_argument("--input", default="recode.json", help="Path to recode.json")
    parser.add_argument(
        "--output-dir",
        default="analysis/results",
        help="Directory to write exported tables",
    )
    parser.add_argument(
        "--prefix",
        default="recode_table",
        help="Prefix for output filenames",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        recode = json.load(f)

    records = []
    for raw_key, score in recode.items():
        parsed = parse_key(raw_key)
        parsed["metric"] = normalize_metric(parsed["run_name"])
        parsed["score_value"] = float(score)
        records.append(parsed)

    records.sort(key=lambda item: (item["model"], item["dataset"], item["task"], item["metric"], item["method"]))

    detail_csv = output_dir / f"{args.prefix}_detail.csv"
    pivot_csv = output_dir / f"{args.prefix}_pivot.csv"
    pivot_md = output_dir / f"{args.prefix}_pivot.md"
    pivot_html = output_dir / f"{args.prefix}_pivot.html"

    write_detail_csv(records, detail_csv)
    write_pivot_csv(records, pivot_csv)
    write_markdown_table(records, pivot_md)
    write_html_table(records, pivot_html)

    print(f"Wrote detail table: {detail_csv}")
    print(f"Wrote pivot table:  {pivot_csv}")
    print(f"Wrote markdown:     {pivot_md}")
    print(f"Wrote html:         {pivot_html}")


if __name__ == "__main__":
    main()
