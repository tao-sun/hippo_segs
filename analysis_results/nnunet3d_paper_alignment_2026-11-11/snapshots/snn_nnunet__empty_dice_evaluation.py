import csv
from contextlib import contextmanager
from functools import wraps
import importlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from threading import RLock


DICE_CONVENTION = {
    "empty_reference_and_prediction": 1.0,
    "empty_reference_with_false_positives": 0.0,
}
_native_metrics_lock = RLock()


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
            if metrics["n_ref"] == 0 or metrics["n_pred"] == 0:
                empty_score = 1.0 if metrics["n_ref"] == metrics["n_pred"] == 0 else 0.0
                metrics["Dice"] = empty_score
                if "IoU" in metrics:
                    metrics["IoU"] = empty_score

    for metric in ("Dice", "IoU"):
        if not all(metric in case["metrics"][region] for case in cases for region in region_keys):
            continue
        for region in region_keys:
            values = [float(case["metrics"][region][metric]) for case in cases]
            finite_values = [value for value in values if not math.isnan(value)]
            converted["mean"].setdefault(region, {})[metric] = (
                sum(finite_values) / len(finite_values) if finite_values else math.nan
            )
        region_means = [converted["mean"][region][metric] for region in region_keys if region not in (0, "0")]
        converted["foreground_mean"][metric] = (
            sum(region_means) / len(region_means) if region_means else math.nan
        )
    converted["dice_convention"] = dict(DICE_CONVENTION)
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
    if getattr(native_writer, "_empty_dice_as_one", False):
        return native_writer

    @wraps(native_writer)
    def wrapped(*args, **kwargs):
        from nnunetv2.evaluation.evaluate_predictions import save_summary_json

        # Normalize before the summary is first written or returned for logging.
        native_args, native_kwargs = list(args), dict(kwargs)
        if len(native_args) >= 3:
            output_file = native_args[2]
            native_args[2] = None
        elif "output_file" in native_kwargs:
            output_file = native_kwargs["output_file"]
            native_kwargs["output_file"] = None
        else:
            raise TypeError("compute_metrics_on_folder output_file argument is missing")
        metrics = native_writer(*native_args, **native_kwargs)
        converted = apply_empty_as_one(metrics)
        if output_file is not None:
            save_summary_json(converted, str(output_file))
        return converted

    wrapped._empty_dice_as_one = True
    return wrapped


@contextmanager
def native_empty_dice_evaluation():
    """Use identical empty-region scores before native saving, logs, and PP selection."""
    module_names = (
        "nnunetv2.evaluation.evaluate_predictions",
        "nnunetv2.training.nnUNetTrainer.nnUNetTrainer",
        "nnunetv2.postprocessing.remove_connected_components",
    )
    with _native_metrics_lock:
        modules = [importlib.import_module(name) for name in module_names]
        originals = [(module, module.compute_metrics_on_folder) for module in modules]
        try:
            for module, original in originals:
                module.compute_metrics_on_folder = metric_writer_empty_as_one(original)
            yield
        finally:
            for module, original in originals:
                module.compute_metrics_on_folder = original


def _write_json(destination: Path, payload: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=4, allow_nan=True)
        stream.write("\n")
    os.replace(temporary, destination)
