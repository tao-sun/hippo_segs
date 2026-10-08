"""Evaluate a BraTS SNN fold with nnU-Net inference and native postprocessing.

Example (from the repository root, with nnU-Net paths configured)::

    python -m snn_nnunet.evaluate_snn_fold \
        --dataset-id 1 --fold 0 --run-dir "$nnUNet_results" --best

Predictions and Dice summaries are written to a timestamped directory under
the fold's results directory. Dice is printed after each case is postprocessed.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from snn_nnunet import cli


def _dataset_id(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("dataset ID must be an integer from 0 to 999") from error
    if not 0 <= number <= 999:
        raise argparse.ArgumentTypeError("dataset ID must be an integer from 0 to 999")
    return number


def _fold(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("fold must be an integer from 0 to 4") from error
    if number not in cli.FIVE_FOLDS:
        raise argparse.ArgumentTypeError("fold must be an integer from 0 to 4")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run native nnU-Net full-volume validation for one SNN fold and "
            "report Dice per case, per region, and averaged across regions."
        )
    )
    parser.add_argument("--dataset-id", type=_dataset_id, required=True)
    parser.add_argument("--fold", type=_fold, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="existing nnU-Net results root used for this training run",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="validate checkpoint_best.pth instead of checkpoint_final.pth",
    )
    return parser


def _region_key(value: Any) -> tuple[int, ...] | int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid region key in nnU-Net summary: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    if not isinstance(value, str):
        raise ValueError(f"Invalid region key in nnU-Net summary: {value!r}")

    key = value.strip()
    if key.startswith("(") and key.endswith(")"):
        items = [item.strip() for item in key[1:-1].split(",") if item.strip()]
        return tuple(int(item) for item in items)
    return int(key)


def _region_name_map(dataset_json_path: Path) -> dict[tuple[int, ...] | int, str]:
    if not dataset_json_path.is_file():
        return {}
    with dataset_json_path.open(encoding="utf-8") as stream:
        dataset_json = json.load(stream)

    names: dict[tuple[int, ...] | int, str] = {}
    for name, value in dataset_json.get("labels", {}).items():
        if name.lower() == "background":
            continue
        if isinstance(value, list):
            key: tuple[int, ...] | int = tuple(int(item) for item in value)
        elif isinstance(value, int) and not isinstance(value, bool):
            key = value
        else:
            continue
        names[key] = name
    return names


def _format_region(region: Any, names: Mapping[tuple[int, ...] | int, str]) -> str:
    key = _region_key(region)
    return names.get(key, str(key))


def _mean(values: list[float]) -> float:
    finite_values = [value for value in values if not math.isnan(value)]
    if not finite_values:
        return float("nan")
    return sum(finite_values) / len(finite_values)


def _print_case(case_id: str, metrics: Mapping[Any, Any], regions, region_names) -> None:
    dice_values = [float(metrics[region]["Dice"]) for region in regions]
    print(
        f"{case_id} | "
        + " ".join(
            f"{_format_region(region, region_names)}="
            f"{float(metrics[region]['Dice']):.4f}"
            f"[GT={int(metrics[region]['n_ref'])},"
            f"Pred={int(metrics[region]['n_pred'])}]"
            for region in regions
        )
        + f" mean={_mean(dice_values):.4f}",
        flush=True,
    )


def print_summary(
    summary: Mapping[str, Any], dataset_json_path: Path, *, print_cases: bool = True
) -> None:
    """Print per-case and fold-level Dice using nnU-Net's native metrics."""

    mean_by_region = summary.get("mean")
    cases = summary.get("metric_per_case")
    if not isinstance(mean_by_region, dict) or not isinstance(cases, list) or not cases:
        raise ValueError("nnU-Net validation summary is missing mean metrics or per-case results")

    region_names = _region_name_map(dataset_json_path)
    region_keys = list(mean_by_region)
    regions = [(key, _format_region(key, region_names)) for key in region_keys]

    print(f"Validation subjects: {len(cases)}")
    if print_cases:
        for case in cases:
            metrics = case.get("metrics", {})
            case_id = Path(
                case.get("prediction_file", case.get("reference_file", "unknown"))
            ).name
            print(
                f"{case_id} | "
                + " ".join(
                    f"{name}={float(metrics[key]['Dice']):.4f}"
                    for key, name in regions
                )
                + f" mean={_mean([float(metrics[key]['Dice']) for key in region_keys]):.4f}",
                flush=True,
            )

    class_means = [float(mean_by_region[key]["Dice"]) for key in region_keys]
    native_macro_mean = summary.get("foreground_mean", {}).get("Dice")
    macro_mean = (
        float(native_macro_mean)
        if native_macro_mean is not None
        else _mean(class_means)
    )
    print(
        f"Mean over {len(cases)} subjects | "
        + " ".join(
            f"{name}={float(mean_by_region[key]['Dice']):.4f}"
            for key, name in regions
        )
        + f" mean={macro_mean:.4f}",
        flush=True,
    )


def _case_metrics(reference_file, prediction_file, reader_writer, regions, ignore_label):
    """Compute nnU-Net metrics with a perfect Dice for two empty masks."""
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics

    metrics = compute_metrics(
        str(reference_file), str(prediction_file), reader_writer, regions, ignore_label
    )
    for region_metrics in metrics["metrics"].values():
        if region_metrics["TP"] + region_metrics["FP"] + region_metrics["FN"] == 0:
            region_metrics["Dice"] = 1.0
            region_metrics["IoU"] = 1.0
    return metrics


def _save_fold_summary(all_case_metrics, regions, summary_path):
    from nnunetv2.evaluation.evaluate_predictions import save_summary_json
    from nnunetv2.utilities.json_export import recursive_fix_for_json_export

    if not all_case_metrics:
        raise RuntimeError("The selected fold contains no validation subjects")
    metric_names = list(all_case_metrics[0]["metrics"][regions[0]])
    means = {
        region: {
            metric_name: float(
                np.nanmean([row["metrics"][region][metric_name] for row in all_case_metrics])
            )
            for metric_name in metric_names
        }
        for region in regions
    }
    foreground_mean = {
        metric_name: float(np.mean([means[region][metric_name] for region in means]))
        for metric_name in metric_names
    }
    summary = {
        "metric_per_case": all_case_metrics,
        "mean": means,
        "foreground_mean": foreground_mean,
    }
    # compute_metrics stores voxel counts as numpy.int64, which JSON cannot serialize.
    recursive_fix_for_json_export(summary)
    save_summary_json(summary, str(summary_path))
    return summary


def _run_native_validation(
    dataset_id: int, fold: int, run_dir: Path, *, best: bool, timestamp: str
) -> Path:
    """Infer the full fold, select nnU-Net postprocessing, and report Dice."""

    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    os.environ["nnUNet_results"] = str(run_dir)
    cli.configure_runtime()
    cli._require_native_paths()

    run_args = SimpleNamespace(dataset_id=dataset_id, run_dir=run_dir)
    cli._restore_run_plans(run_args)
    cli._check_saved_results(dataset_id)

    results_folder = cli._results_folder(dataset_id)
    checkpoint_name = "checkpoint_best.pth" if best else "checkpoint_final.pth"
    checkpoint_path = results_folder / f"fold_{fold}" / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    import torch
    from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.postprocessing.remove_connected_components import determine_postprocessing
    from nnunetv2.run.run_training import get_trainer_from_args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    trainer = get_trainer_from_args(
        str(dataset_id), cli.CONFIGURATION, fold, cli.TRAINER, cli.PLANS,
        continue_training=True, device=device,
    )
    trainer.load_checkpoint(str(checkpoint_path))
    print(f"Loaded checkpoint at epoch {trainer.current_epoch + 1}/{trainer.num_epochs}")
    print("Running native full-volume validation on the held-out fold")

    checkpoint_label = "best" if best else "final"
    evaluation_folder = (
        results_folder / f"fold_{fold}" / f"evaluation_native_{checkpoint_label}_{timestamp}"
    )
    prediction_folder = evaluation_folder / "raw"
    reference_folder = evaluation_folder / "reference"
    summary_path = evaluation_folder / "summary.json"
    raw_root = Path(os.environ["nnUNet_raw"])
    from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

    dataset_json_path = raw_root / convert_id_to_dataset_name(dataset_id) / "dataset.json"

    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=trainer.device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.manual_initialization(
        trainer.network,
        trainer.plans_manager,
        trainer.configuration_manager,
        None,
        trainer.dataset_json,
        trainer.__class__.__name__,
        trainer.inference_allowed_mirroring_axes,
    )

    maybe_mkdir_p(str(prediction_folder))
    maybe_mkdir_p(str(reference_folder))
    _, validation_keys = trainer.do_split()
    dataset_val = trainer.dataset_class(
        trainer.preprocessed_dataset_folder,
        validation_keys,
        folder_with_segs_from_previous_stage=trainer.folder_with_segs_from_previous_stage,
    )
    foreground_regions = (
        trainer.label_manager.foreground_regions
        if trainer.label_manager.has_regions
        else trainer.label_manager.foreground_labels
    )
    reader_writer = trainer.plans_manager.image_reader_writer_class()
    region_names = _region_name_map(dataset_json_path)
    print("Sliding window: 50% overlap, Gaussian weighting enabled")
    print("Postprocessing: native nnU-Net selection on the complete validation fold")
    print("Per-patient Dice will be printed after nnU-Net selects its rules.")
    print(f"Evaluation directory: {evaluation_folder}")

    for index, case_id in enumerate(dataset_val.identifiers, start=1):
        print(f"[{index}/{len(dataset_val.identifiers)}] Predicting {case_id}", flush=True)
        data, _, seg_prev, properties = dataset_val.load_case(case_id)
        data = data[:]
        if trainer.is_cascaded:
            from nnunetv2.utilities.label_handling.label_handling import (
                convert_labelmap_to_one_hot,
            )

            seg_prev = seg_prev[:]
            data = np.vstack(
                (
                    data,
                    convert_labelmap_to_one_hot(
                        seg_prev,
                        trainer.label_manager.foreground_labels,
                        output_dtype=data.dtype,
                    ),
                )
            )

        logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(data)).cpu()
        output_file_truncated = prediction_folder / case_id
        export_prediction_from_logits(
            logits,
            properties,
            trainer.configuration_manager,
            trainer.plans_manager,
            trainer.dataset_json,
            str(output_file_truncated),
        )

        prediction_file = Path(
            str(output_file_truncated) + trainer.dataset_json["file_ending"]
        )
        reference_file = (
            Path(trainer.preprocessed_dataset_folder_base)
            / "gt_segmentations"
            / prediction_file.name
        )
        if not reference_file.is_file():
            raise FileNotFoundError(f"Reference segmentation not found: {reference_file}")
        (reference_folder / reference_file.name).symlink_to(reference_file)
        del logits, data

    print("Selecting nnU-Net postprocessing rules on all validation predictions.", flush=True)
    pp_fns, pp_fn_kwargs = determine_postprocessing(
        str(prediction_folder), str(reference_folder), trainer.plans_manager.plans,
        trainer.dataset_json, num_processes=2, keep_postprocessed_files=True,
    )
    selected_rules = [
        (fn.__name__, kwargs) for fn, kwargs in zip(pp_fns, pp_fn_kwargs)
    ]
    print(f"Selected nnU-Net rules: {selected_rules}", flush=True)
    print(f"Native postprocessing file: {prediction_folder / 'postprocessing.pkl'}")

    postprocessed_folder = prediction_folder / "postprocessed"
    all_case_metrics = []
    for case_id in dataset_val.identifiers:
        filename = case_id + trainer.dataset_json["file_ending"]
        reference_file = reference_folder / filename
        postprocessed_file = postprocessed_folder / filename
        if not postprocessed_file.is_file():
            raise FileNotFoundError(f"Postprocessed prediction not found: {postprocessed_file}")
        case_metrics = _case_metrics(
            reference_file, postprocessed_file, reader_writer, foreground_regions,
            trainer.label_manager.ignore_label,
        )
        all_case_metrics.append(case_metrics)
        print("FINAL ", end="")
        _print_case(case_id, case_metrics["metrics"], foreground_regions, region_names)

    summary = _save_fold_summary(all_case_metrics, foreground_regions, summary_path)
    print("FINAL fold scores (nnU-Net-selected postprocessing):")
    print_summary(summary, dataset_json_path, print_cases=False)
    print(f"Final summary: {summary_path}")
    return summary_path


class _Tee:
    """Write output to the terminal and the run log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def main() -> int:
    args = build_parser().parse_args()
    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_label = "best" if args.best else "final"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = logs_dir / (
        f"evaluate_snn_fold_dataset{args.dataset_id:03d}_fold{args.fold}_"
        f"{checkpoint_label}_{timestamp}.out"
    )

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee = _Tee(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"Evaluation log: {log_path}")
            try:
                _run_native_validation(
                    args.dataset_id, args.fold, args.run_dir,
                    best=args.best, timestamp=timestamp,
                )
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
