import csv
import json
import math
import os
from copy import deepcopy
from pathlib import Path


REGION_NAMES = {
    "(1,2,3)": "WT",
    "(1,3)": "TC",
    "3": "ET",
}


def _region_name(key: object) -> str:
    normalized = str(key).replace(" ", "")
    return REGION_NAMES.get(normalized, str(key))


def _case_name(path: str) -> str:
    name = Path(path).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def apply_empty_as_one(summary: dict) -> dict:
    converted = deepcopy(summary)
    cases = converted.get("metric_per_case", [])
    if not cases:
        return converted

    region_keys = list(cases[0]["metrics"])
    for case in cases:
        for region in region_keys:
            metrics = case["metrics"][region]
            if metrics["n_ref"] == 0 and metrics["n_pred"] == 0:
                metrics["Dice"] = 1.0

    for region in region_keys:
        dice_values = [case["metrics"][region]["Dice"] for case in cases]
        finite_values = [float(value) for value in dice_values if not math.isnan(float(value))]
        converted["mean"].setdefault(region, {})["Dice"] = (
            sum(finite_values) / len(finite_values) if finite_values else math.nan
        )

    region_means = [converted["mean"][region]["Dice"] for region in region_keys]
    converted["foreground_mean"]["Dice"] = sum(region_means) / len(region_means)
    return converted


def write_voxel_counts_csv(summary: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "case", "region", "gt_voxels", "pred_voxels", "dice",
        ])
        writer.writeheader()
        for case in summary.get("metric_per_case", []):
            case_name = _case_name(case["prediction_file"])
            for region, metrics in case["metrics"].items():
                writer.writerow({
                    "case": case_name,
                    "region": _region_name(region),
                    "gt_voxels": metrics["n_ref"],
                    "pred_voxels": metrics["n_pred"],
                    "dice": metrics["Dice"],
                })


def finalize_summary(summary_path: Path) -> dict:
    summary_path = Path(summary_path)
    with summary_path.open(encoding="utf-8") as stream:
        native = json.load(stream)

    backup_path = summary_path.with_name("summary_nnunet_empty_as_nan.json")
    if not backup_path.exists():
        _write_json(backup_path, native)

    converted = apply_empty_as_one(native)
    _write_json(summary_path, converted)
    write_voxel_counts_csv(converted, summary_path.with_name("voxel_counts_per_case.csv"))
    return converted


def finalize_fold_summary(model_folder: Path, fold: int) -> dict:
    return finalize_summary(
        Path(model_folder) / f"fold_{fold}" / "validation" / "summary.json"
    )


def metric_writer_empty_as_one(native_writer):
    def wrapped(*args, **kwargs):
        from nnunetv2.evaluation.evaluate_predictions import save_summary_json

        metrics = native_writer(*args, **kwargs)
        output_file = kwargs.get("output_file")
        if output_file is None:
            if len(args) < 3:
                raise TypeError("compute_metrics_on_folder output_file argument is missing")
            output_file = args[2]
        converted = apply_empty_as_one(metrics)
        save_summary_json(converted, str(output_file))
        return converted

    return wrapped


def _write_json(destination: Path, payload: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=4, allow_nan=True)
        stream.write("\n")
    os.replace(temporary, destination)
