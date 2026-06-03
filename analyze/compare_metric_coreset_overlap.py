import argparse
import json
from pathlib import Path
from typing import Tuple

try:
    from analyze.compare_coreset_overlap import compute_regions, dump_region_ids, load_selected_ids
except ModuleNotFoundError:
    from compare_coreset_overlap import compute_regions, dump_region_ids, load_selected_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare three metric-specific core sets under the same method."
    )
    parser.add_argument("--nll_file", type=str, default="")
    parser.add_argument("--ifd_file", type=str, default="")
    parser.add_argument("--ufs_file", type=str, default="")
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--dataset", type=str, default="xsota")
    parser.add_argument("--method", type=str, default="key")
    parser.add_argument("--topk", type=int, default=10000)
    parser.add_argument("--summary_json", type=str, default="")
    parser.add_argument("--region_ids_dir", type=str, default="")
    return parser.parse_args()


def resolve_metric_path(model: str, dataset: str, method: str, metric: str) -> str:
    base = Path("coreset") / model / dataset / method
    if method == "key":
        return str(base / f"{metric}_{metric}" / "coreset.jsonl")
    return str(base / metric / "coreset.jsonl")


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    nll_file = args.nll_file or resolve_metric_path(args.model, args.dataset, args.method, "nll")
    ifd_file = args.ifd_file or resolve_metric_path(args.model, args.dataset, args.method, "ifd")
    ufs_file = args.ufs_file or resolve_metric_path(args.model, args.dataset, args.method, "ufs")
    return nll_file, ifd_file, ufs_file


def main() -> None:
    args = parse_args()
    nll_file, ifd_file, ufs_file = resolve_paths(args)

    nll_ids = load_selected_ids(nll_file, "nll", args.topk)
    ifd_ids = load_selected_ids(ifd_file, "ifd", args.topk)
    ufs_ids = load_selected_ids(ufs_file, "ufs", args.topk)

    regions = compute_regions(nll_ids, ifd_ids, ufs_ids)
    summary = {
        "scope": {
            "model": args.model,
            "dataset": args.dataset,
            "method": args.method,
        },
        "files": {
            "nll": nll_file,
            "ifd": ifd_file,
            "ufs": ufs_file,
        },
        "set_sizes": {
            "nll": len(nll_ids),
            "ifd": len(ifd_ids),
            "ufs": len(ufs_ids),
            "union": len(nll_ids | ifd_ids | ufs_ids),
        },
        "pairwise_intersections": {
            "nll_ifd": len(nll_ids & ifd_ids),
            "nll_ufs": len(nll_ids & ufs_ids),
            "ifd_ufs": len(ifd_ids & ufs_ids),
        },
        "regions": {name: len(ids) for name, ids in regions.items()},
    }

    print(f"NLL file: {nll_file}")
    print(f"IFD file: {ifd_file}")
    print(f"UFS file: {ufs_file}")
    print()
    print("Venn regions:")
    print(f"  NLL only: {summary['regions']['a_only']}")
    print(f"  NLL & IFD only: {summary['regions']['ab_only']}")
    print(f"  IFD only: {summary['regions']['b_only']}")
    print(f"  NLL & UFS only: {summary['regions']['ac_only']}")
    print(f"  NLL & IFD & UFS: {summary['regions']['abc']}")
    print(f"  IFD & UFS only: {summary['regions']['bc_only']}")
    print(f"  UFS only: {summary['regions']['c_only']}")
    print()
    print("Set sizes:")
    print(f"  NLL: {summary['set_sizes']['nll']}")
    print(f"  IFD: {summary['set_sizes']['ifd']}")
    print(f"  UFS: {summary['set_sizes']['ufs']}")
    print(f"  Union: {summary['set_sizes']['union']}")
    print()
    print("Pairwise intersections (inclusive of triple-overlap):")
    print(f"  NLL & IFD: {summary['pairwise_intersections']['nll_ifd']}")
    print(f"  NLL & UFS: {summary['pairwise_intersections']['nll_ufs']}")
    print(f"  IFD & UFS: {summary['pairwise_intersections']['ifd_ufs']}")

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
