import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare three selected core sets and report 3-way overlap counts."
    )
    parser.add_argument("--coarse_file", type=str, default="")
    parser.add_argument("--fine_file", type=str, default="")
    parser.add_argument("--key_file", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--metric", type=str, default="")
    parser.add_argument("--topk", type=int, default=10000)
    parser.add_argument("--coarse_label", type=str, default="Coarse")
    parser.add_argument("--fine_label", type=str, default="Fine")
    parser.add_argument("--key_label", type=str, default="ATCS")
    parser.add_argument("--summary_json", type=str, default="")
    parser.add_argument("--region_ids_dir", type=str, default="")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
    return data


def infer_id(obj: Dict[str, Any], line_no: int, path: Path) -> Any:
    if "idx" in obj:
        return obj["idx"]
    if "source" in obj and "id" in obj:
        return f"{obj['source']}_{obj['id']}"
    if "id" in obj:
        return obj["id"]
    raise KeyError(f"Missing idx/id on line {line_no} in {path}")


def load_ids_from_coreset(path: Path) -> Set[Any]:
    ids: Set[Any] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.add(infer_id(obj, line_no, path))
    return ids


def load_ids_from_score(path: Path, metric: str, topk: int) -> Set[Any]:
    records: List[Tuple[Any, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = infer_id(obj, line_no, path)
            if metric not in obj:
                raise KeyError(f"Missing metric '{metric}' on line {line_no} in {path}")
            records.append((idx, float(obj[metric])))

    records.sort(key=lambda item: item[1], reverse=True)
    return {idx for idx, _score in records[:topk]}


def infer_input_kind(sample: Dict[str, Any], metric: str) -> str:
    if "conversations" in sample:
        return "coreset"
    if metric and metric in sample:
        return "score"
    if "x_text" in sample and "y_text" in sample:
        return "score"
    return "coreset"


def load_selected_ids(path_str: str, metric: str, topk: int) -> Set[Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = load_jsonl(path)
    if not data:
        return set()

    kind = infer_input_kind(data[0], metric)
    if kind == "score":
        if not metric:
            raise ValueError(f"Metric is required when loading score file: {path}")
        return load_ids_from_score(path, metric, topk)
    return load_ids_from_coreset(path)


def resolve_default_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    if args.coarse_file and args.fine_file and args.key_file:
        return args.coarse_file, args.fine_file, args.key_file

    missing = [name for name in ("model", "dataset", "metric") if not getattr(args, name)]
    if missing:
        raise ValueError(
            "Either pass --coarse_file/--fine_file/--key_file, or provide "
            "--model/--dataset/--metric."
        )

    base = Path("coreset") / args.model / args.dataset
    coarse_file = str(base / "coarse" / args.metric / "coreset.jsonl")
    fine_file = str(base / "fine" / args.metric / "coreset.jsonl")
    key_file = str(base / "key" / f"{args.metric}_{args.metric}" / "coreset.jsonl")
    return coarse_file, fine_file, key_file


def compute_regions(a: Set[Any], b: Set[Any], c: Set[Any]) -> Dict[str, Set[Any]]:
    return {
        "a_only": a - b - c,
        "b_only": b - a - c,
        "c_only": c - a - b,
        "ab_only": (a & b) - c,
        "ac_only": (a & c) - b,
        "bc_only": (b & c) - a,
        "abc": a & b & c,
    }


def sort_ids(values: Iterable[Any]) -> List[Any]:
    return sorted(values, key=lambda x: (str(type(x)), str(x)))


def dump_region_ids(region_ids_dir: str, regions: Dict[str, Set[Any]]) -> None:
    out_dir = Path(region_ids_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for region_name, ids in regions.items():
        out_path = out_dir / f"{region_name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(sort_ids(ids), f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    coarse_file, fine_file, key_file = resolve_default_paths(args)

    coarse_ids = load_selected_ids(coarse_file, args.metric, args.topk)
    fine_ids = load_selected_ids(fine_file, args.metric, args.topk)
    key_ids = load_selected_ids(key_file, args.metric, args.topk)

    regions = compute_regions(coarse_ids, fine_ids, key_ids)

    summary = {
        "files": {
            "coarse": coarse_file,
            "fine": fine_file,
            "key": key_file,
        },
        "labels": {
            "coarse": args.coarse_label,
            "fine": args.fine_label,
            "key": args.key_label,
        },
        "set_sizes": {
            "coarse": len(coarse_ids),
            "fine": len(fine_ids),
            "key": len(key_ids),
            "union": len(coarse_ids | fine_ids | key_ids),
        },
        "pairwise_intersections": {
            "coarse_fine": len(coarse_ids & fine_ids),
            "coarse_key": len(coarse_ids & key_ids),
            "fine_key": len(fine_ids & key_ids),
        },
        "regions": {name: len(ids) for name, ids in regions.items()},
    }

    print(f"{args.coarse_label} file: {coarse_file}")
    print(f"{args.fine_label} file: {fine_file}")
    print(f"{args.key_label} file: {key_file}")
    print()
    print("Venn regions:")
    print(f"  {args.coarse_label} only: {summary['regions']['a_only']}")
    print(f"  {args.coarse_label} & {args.fine_label} only: {summary['regions']['ab_only']}")
    print(f"  {args.fine_label} only: {summary['regions']['b_only']}")
    print(f"  {args.coarse_label} & {args.key_label} only: {summary['regions']['ac_only']}")
    print(f"  {args.coarse_label} & {args.fine_label} & {args.key_label}: {summary['regions']['abc']}")
    print(f"  {args.fine_label} & {args.key_label} only: {summary['regions']['bc_only']}")
    print(f"  {args.key_label} only: {summary['regions']['c_only']}")
    print()
    print("Set sizes:")
    print(f"  {args.coarse_label}: {summary['set_sizes']['coarse']}")
    print(f"  {args.fine_label}: {summary['set_sizes']['fine']}")
    print(f"  {args.key_label}: {summary['set_sizes']['key']}")
    print(f"  Union: {summary['set_sizes']['union']}")
    print()
    print("Pairwise intersections (inclusive of triple-overlap):")
    print(f"  {args.coarse_label} & {args.fine_label}: {summary['pairwise_intersections']['coarse_fine']}")
    print(f"  {args.coarse_label} & {args.key_label}: {summary['pairwise_intersections']['coarse_key']}")
    print(f"  {args.fine_label} & {args.key_label}: {summary['pairwise_intersections']['fine_key']}")

    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print()
        print(f"Saved summary to: {out_path}")

    if args.region_ids_dir:
        dump_region_ids(args.region_ids_dir, regions)
        print(f"Saved region id lists to: {args.region_ids_dir}")


if __name__ == "__main__":
    main()
