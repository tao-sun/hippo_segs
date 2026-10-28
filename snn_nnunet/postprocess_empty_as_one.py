"""Select nnU-Net post-processing with Dice(empty GT, empty prediction) = 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from snn_nnunet.empty_dice_evaluation import (
    finalize_summary,
    native_empty_dice_evaluation,
    write_voxel_counts_csv,
)


def _print_voxel_totals(summary: dict, label: str) -> None:
    print(f"{label} voxel counts (overlapping BraTS regions):", flush=True)
    if not summary.get("metric_per_case"):
        print("  no cases", flush=True)
        return
    for region in summary["metric_per_case"][0]["metrics"]:
        rows = [case["metrics"][region] for case in summary["metric_per_case"]]
        gt_voxels = sum(int(row["n_ref"]) for row in rows)
        pred_voxels = sum(int(row["n_pred"]) for row in rows)
        dice = summary["mean"][region]["Dice"]
        print(
            f"  {region}: GT={gt_voxels} prediction={pred_voxels} Dice={dice:.6f}",
            flush=True,
        )


def run(
    predictions: Path,
    references: Path,
    plans_json: Path,
    dataset_json: Path,
    num_processes: int,
) -> None:
    from nnunetv2.postprocessing import remove_connected_components

    summary_path = predictions / "summary.json"
    raw_summary = finalize_summary(summary_path)
    _print_voxel_totals(raw_summary, "Raw")

    with native_empty_dice_evaluation():
        remove_connected_components.determine_postprocessing(
            str(predictions),
            str(references),
            str(plans_json),
            str(dataset_json),
            num_processes=num_processes,
            keep_postprocessed_files=True,
        )

    postprocessed_summary_path = predictions / "postprocessed" / "summary.json"
    with postprocessed_summary_path.open(encoding="utf-8") as stream:
        postprocessed_summary = json.load(stream)
    write_voxel_counts_csv(
        postprocessed_summary,
        predictions / "postprocessed" / "voxel_counts_per_case.csv",
    )
    _print_voxel_totals(postprocessed_summary, "Postprocessed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", type=Path, required=True, dest="predictions")
    parser.add_argument("-ref", type=Path, required=True, dest="references")
    parser.add_argument("-plans_json", type=Path, required=True)
    parser.add_argument("-dataset_json", type=Path, required=True)
    parser.add_argument("-np", type=int, default=8, dest="num_processes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_processes < 1:
        raise ValueError("-np must be a positive integer")
    run(
        args.predictions.resolve(),
        args.references.resolve(),
        args.plans_json.resolve(),
        args.dataset_json.resolve(),
        args.num_processes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
