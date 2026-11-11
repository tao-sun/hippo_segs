"""Custom full-volume nnU-Net validation pipeline."""

import argparse
import csv
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from postprocessing import morphological_closing_3d, remove_small_components_3d


REGION_ORDER = ("WT", "TC", "ET")


def _three_ints(values):
    converted = tuple(int(value) for value in values)
    if len(converted) != 3:
        raise argparse.ArgumentTypeError("expected exactly three integer values")
    return converted


def build_parser():
    parser = argparse.ArgumentParser(
        description="Full-volume nnU-Net validation with fixed sliding-window and BraTS postprocessing"
    )
    parser.add_argument("--dataset-id", required=True, type=int)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth")
    parser.add_argument("--patch-size", nargs=3, type=int, default=(96, 96, 96))
    parser.add_argument("--tile-step-size", type=float, default=0.5)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-component-sizes", nargs=3, type=int, default=(161, 166, 50),
        metavar=("ET", "TC", "WT"),
    )
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def labelmap_to_regions(labelmap):
    """Convert BraTS labels; padding -1 and resection cavity 4 are outside WT/TC/ET."""
    array = np.asarray(labelmap)
    if array.ndim != 3:
        raise ValueError("labelmap must have shape [X, Y, Z]")
    if not np.isin(array, (-1, 0, 1, 2, 3, 4)).all():
        raise ValueError("labelmap contains values outside nnU-Net padding and BraTS labels")
    return np.stack(
        (
            np.isin(array, (1, 2, 3)),
            np.isin(array, (1, 3)),
            array == 3,
        )
    ).astype(np.uint8)


def threshold_probabilities(probabilities, threshold=0.5):
    array = np.asarray(probabilities)
    if array.ndim != 4 or array.shape[0] != 3:
        raise ValueError("probabilities must have shape [3, X, Y, Z] in WT, TC, ET order")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return (array > threshold).astype(np.uint8)


def dice_metrics(reference, prediction):
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if reference.shape != prediction.shape or reference.ndim != 4 or reference.shape[0] != 3:
        raise ValueError("reference and prediction must have matching [3, X, Y, Z] shapes")
    result = {}
    for index, region in enumerate(REGION_ORDER):
        gt_voxels = int(reference[index].sum())
        pred_voxels = int(prediction[index].sum())
        denominator = gt_voxels + pred_voxels
        intersection = int(np.logical_and(reference[index], prediction[index]).sum())
        result[region] = {
            "dice": 1.0 if denominator == 0 else 2.0 * intersection / denominator,
            "gt_voxels": gt_voxels,
            "pred_voxels": pred_voxels,
        }
    return result


def summarize_cases(cases):
    if not cases:
        raise ValueError("cannot summarize an empty validation fold")
    means = {}
    for stage in ("raw", "postprocessed"):
        region_means = {
            region: float(np.mean([case[stage][region]["dice"] for case in cases]))
            for region in REGION_ORDER
        }
        region_means["mean"] = float(np.mean(list(region_means.values())))
        means[stage] = region_means
    delta = {
        key: means["postprocessed"][key] - means["raw"][key]
        for key in (*REGION_ORDER, "mean")
    }
    return {"cases": cases, "mean": means, "delta": delta}


def postprocess_regions(regions, min_component_sizes_et_tc_wt, closing_radius):
    array = np.asarray(regions)
    if array.ndim != 4 or array.shape[0] != 3:
        raise ValueError("regions must have shape [3, X, Y, Z] in WT, TC, ET order")
    if len(min_component_sizes_et_tc_wt) != 3:
        raise ValueError("component thresholds must contain ET, TC, WT values")
    thresholds = {
        "ET": int(min_component_sizes_et_tc_wt[0]),
        "TC": int(min_component_sizes_et_tc_wt[1]),
        "WT": int(min_component_sizes_et_tc_wt[2]),
    }
    output = np.empty_like(array, dtype=np.uint8)
    for index, region in enumerate(REGION_ORDER):
        filtered = remove_small_components_3d(array[index], thresholds[region])
        output[index] = morphological_closing_3d(filtered, closing_radius)
    return output


@contextmanager
def temporary_patch_size(configuration_manager, patch_size):
    original = deepcopy(configuration_manager.configuration["patch_size"])
    configuration_manager.configuration["patch_size"] = [int(value) for value in patch_size]
    try:
        yield
    finally:
        configuration_manager.configuration["patch_size"] = original


def _regions_to_logits(regions, *, device):
    import torch

    masks = torch.as_tensor(regions, dtype=torch.bool, device=device)
    return torch.where(masks, 20.0, -20.0)


def _canonicalize_regions(regions, label_manager):
    """Resolve overlapping region predictions using nnU-Net's label priority."""
    segmentation = label_manager.convert_logits_to_segmentation(
        _regions_to_logits(regions, device="cpu")
    )
    if hasattr(segmentation, "cpu"):
        segmentation = segmentation.cpu().numpy()
    return labelmap_to_regions(segmentation)


def run_fold_validation(
    trainer,
    *,
    output_dir,
    patch_size=(96, 96, 96),
    tile_step_size=0.5,
    prob_threshold=0.5,
    min_component_sizes=(161, 166, 50),
    closing_radius=1,
    predictor_class=None,
    export_prediction=None,
):
    """Run tiled inference on a trainer's validation split and evaluate before/after PP."""
    import torch

    if predictor_class is None:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor_class = nnUNetPredictor
    if export_prediction is None:
        from nnunetv2.inference.export_prediction import export_prediction_from_logits

        export_prediction = export_prediction_from_logits

    output_dir = Path(output_dir)
    raw_folder = output_dir / "raw"
    postprocessed_folder = output_dir / "postprocessed"
    raw_regions_folder = output_dir / "regions" / "raw"
    pp_regions_folder = output_dir / "regions" / "postprocessed"
    for folder in (raw_folder, postprocessed_folder, raw_regions_folder, pp_regions_folder):
        folder.mkdir(parents=True, exist_ok=True)

    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()
    predictor = predictor_class(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=trainer.device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    with temporary_patch_size(trainer.configuration_manager, patch_size):
        predictor.manual_initialization(
            trainer.network,
            trainer.plans_manager,
            trainer.configuration_manager,
            None,
            trainer.dataset_json,
            trainer.__class__.__name__,
            trainer.inference_allowed_mirroring_axes,
        )

    _, validation_keys = trainer.do_split()
    dataset_val = trainer.dataset_class(
        trainer.preprocessed_dataset_folder,
        validation_keys,
        folder_with_segs_from_previous_stage=trainer.folder_with_segs_from_previous_stage,
    )
    if not dataset_val.identifiers:
        raise ValueError("validation split is empty")

    case_results = []
    with temporary_patch_size(trainer.configuration_manager, patch_size):
        print(
            f"Inference patch: {tuple(patch_size)}; tile step: {tile_step_size}; "
            "Gaussian blending: on; mirroring: on",
            flush=True,
        )
        for index, case_id in enumerate(dataset_val.identifiers, start=1):
            print(f"[{index}/{len(dataset_val.identifiers)}] {case_id}", flush=True)
            data, seg, seg_prev, properties = dataset_val.load_case(case_id)
            data = np.asarray(data)
            if trainer.is_cascaded:
                from nnunetv2.utilities.label_handling.label_handling import (
                    convert_labelmap_to_one_hot,
                )

                data = np.vstack(
                    (
                        data,
                        convert_labelmap_to_one_hot(
                            np.asarray(seg_prev),
                            trainer.label_manager.foreground_labels,
                            output_dtype=data.dtype,
                        ),
                    )
                )
            if seg is None:
                raise ValueError(f"Missing reference segmentation for {case_id}")
            reference_labelmap = np.asarray(seg)
            if reference_labelmap.ndim == 4 and reference_labelmap.shape[0] == 1:
                reference_labelmap = reference_labelmap[0]
            reference_regions = labelmap_to_regions(reference_labelmap)

            with torch.no_grad():
                logits = predictor.predict_sliding_window_return_logits(
                    torch.from_numpy(data)
                ).detach().cpu()
            if logits.ndim != 4 or logits.shape[0] != 3:
                raise ValueError(
                    f"Expected 3 region logits [WT, TC, ET] for {case_id}; got {tuple(logits.shape)}"
                )
            probabilities = torch.sigmoid(logits).numpy()
            raw_prediction = threshold_probabilities(probabilities, prob_threshold)
            raw_prediction = _canonicalize_regions(raw_prediction, trainer.label_manager)
            processed_prediction = postprocess_regions(
                raw_prediction,
                min_component_sizes_et_tc_wt=min_component_sizes,
                closing_radius=closing_radius,
            )
            processed_prediction = _canonicalize_regions(
                processed_prediction, trainer.label_manager
            )

            raw_metrics = dice_metrics(reference_regions, raw_prediction)
            pp_metrics = dice_metrics(reference_regions, processed_prediction)
            case_results.append(
                {"case": str(case_id), "raw": raw_metrics, "postprocessed": pp_metrics}
            )

            for stage, masks, folder in (
                ("raw", raw_prediction, raw_folder),
                ("postprocessed", processed_prediction, postprocessed_folder),
            ):
                output_prefix = folder / str(case_id)
                export_prediction(
                    _regions_to_logits(masks, device="cpu"),
                    properties,
                    trainer.configuration_manager,
                    trainer.plans_manager,
                    trainer.dataset_json,
                    str(output_prefix),
                )
            np.savez_compressed(raw_regions_folder / f"{case_id}.npz", regions=raw_prediction)
            np.savez_compressed(
                pp_regions_folder / f"{case_id}.npz", regions=processed_prediction
            )

    summary = summarize_cases(case_results)
    summary["settings"] = {
        "patch_size": [int(value) for value in patch_size],
        "tile_step_size": float(tile_step_size),
        "gaussian_blending": True,
        "mirroring": True,
        "mirroring_axes": list(trainer.inference_allowed_mirroring_axes),
        "probability_threshold": float(prob_threshold),
        "min_component_sizes_et_tc_wt": [int(value) for value in min_component_sizes],
        "closing_radius": int(closing_radius),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "stage", "region", "dice", "gt_voxels", "pred_voxels"))
        for case in case_results:
            for stage in ("raw", "postprocessed"):
                for region in REGION_ORDER:
                    metrics = case[stage][region]
                    writer.writerow(
                        (
                            case["case"], stage, region, metrics["dice"],
                            metrics["gt_voxels"], metrics["pred_voxels"],
                        )
                    )
    print(
        f"Mean Dice (raw): {summary['mean']['raw']['mean']:.6f}; "
        f"postprocessed: {summary['mean']['postprocessed']['mean']:.6f}",
        flush=True,
    )
    return summary


def main(argv=None):
    import torch
    from nnunetv2.run.run_training import get_trainer_from_args

    args = build_parser().parse_args(argv)
    if not args.results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {args.results_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = get_trainer_from_args(
        str(args.dataset_id),
        args.configuration,
        args.fold,
        args.trainer,
        args.plans,
        continue_training=True,
        device=device,
    )
    checkpoint = Path(trainer.output_folder) / args.checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    trainer.load_checkpoint(str(checkpoint))
    print(f"Validation output: {args.output_dir}", flush=True)
    run_fold_validation(
        trainer,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        tile_step_size=args.tile_step_size,
        prob_threshold=args.prob_threshold,
        min_component_sizes=args.min_component_sizes,
        closing_radius=args.closing_radius,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
