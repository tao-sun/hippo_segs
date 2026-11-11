"""Custom full-volume nnU-Net validation pipeline."""

import argparse
import csv
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from snn_nnunet.empty_dice_evaluation import DICE_CONVENTION

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
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--patch-size", nargs=3, type=int, default=None)
    parser.add_argument("--subset", choices=('val', 'test'), default='val',
                        help='Internal validation, or explicit held-out patient test after training')
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


def _mean_case_dice(cases):
    means = {}
    for stage in ("raw", "postprocessed"):
        region_means = {
            region: float(np.mean([case[stage][region]["dice"] for case in cases]))
            for region in REGION_ORDER
        }
        region_means["mean"] = float(np.mean(list(region_means.values())))
        means[stage] = region_means
    return means


def summarize_cases(cases, *, patient_wise=False):
    """Average volume Dice; optionally also give every base patient ID equal weight."""
    if not cases:
        raise ValueError("cannot summarize an empty validation fold")
    means = _mean_case_dice(cases)
    delta = {
        key: means["postprocessed"][key] - means["raw"][key]
        for key in (*REGION_ORDER, "mean")
    }
    summary = {"cases": cases, "mean": means, "delta": delta, "dice_convention": dict(DICE_CONVENTION)}
    if patient_wise:
        from snn_nnunet.patient_partition import patient_id

        groups, seen = {}, set()
        for case in cases:
            identifier = case['case']
            patient = patient_id(identifier)
            if identifier in seen:
                raise ValueError(f'Duplicate volume in patient Dice aggregation: {identifier}')
            seen.add(identifier)
            groups.setdefault(patient, []).append(case)
        patients = []
        for patient, volumes in sorted(groups.items()):
            volumes = sorted(volumes, key=lambda case: case['case'])
            patients.append({'patient': patient, 'num_volumes': len(volumes),
                             'cases': [case['case'] for case in volumes],
                             **_mean_case_dice(volumes)})
        patient_means = {}
        for stage in ('raw', 'postprocessed'):
            scores = {region: float(np.mean([patient[stage][region] for patient in patients]))
                      for region in REGION_ORDER}
            scores['mean'] = float(np.mean(list(scores.values())))
            patient_means[stage] = scores
        summary.update(
            counts={'volumes': len(cases), 'patients': len(patients)},
            patients=patients, mean_per_volume=means, mean_per_patient=patient_means,
            delta_per_patient={region: patient_means['postprocessed'][region] - patient_means['raw'][region]
                               for region in (*REGION_ORDER, 'mean')},
        )
    return summary


def write_patient_dice_tables(summary, output_dir):
    """Export evaluation aggregates plus auditable volume means for each patient."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'aggregate_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('aggregation', 'stage', 'region', 'dice', 'num_volumes', 'num_patients'))
        for aggregation in ('per_volume', 'per_patient'):
            for stage in ('raw', 'postprocessed'):
                for region in (*REGION_ORDER, 'mean'):
                    writer.writerow((aggregation, stage, region,
                                     summary[f'mean_{aggregation}'][stage][region],
                                     summary['counts']['volumes'], summary['counts']['patients']))
    with (output_dir / 'patient_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('patient', 'num_volumes', 'cases', 'stage', *REGION_ORDER, 'mean'))
        for patient in summary['patients']:
            for stage in ('raw', 'postprocessed'):
                writer.writerow((patient['patient'], patient['num_volumes'],
                                 ';'.join(patient['cases']), stage,
                                 *(patient[stage][region] for region in (*REGION_ORDER, 'mean'))))


def print_dice_summary(summary):
    """Report a directly comparable raw/PP pair and its absolute Dice gain."""
    print("Dice convention (raw and postprocessed): empty-empty=1; empty GT with false positives=0", flush=True)
    print(
        f"Mean Dice (raw): {summary['mean']['raw']['mean']:.6f}; "
        f"postprocessed: {summary['mean']['postprocessed']['mean']:.6f}",
        flush=True,
    )
    gains = " ".join(f"{region}={summary['delta'][region]:+.6f}" for region in REGION_ORDER)
    print(f"Dice gain from postprocessing: {gains} mean={summary['delta']['mean']:+.6f}", flush=True)
    if 'mean_per_patient' in summary:
        print(f"Evaluation aggregation: {summary['counts']['volumes']} volumes; "
              f"{summary['counts']['patients']} patients", flush=True)
        for unit in ('volume', 'patient'):
            for stage in ('raw', 'postprocessed'):
                scores = summary[f'mean_per_{unit}'][stage]
                values = ' '.join(f'{region}={scores[region]:.6f}'
                                  for region in (*REGION_ORDER, 'mean'))
                print(f'Mean Dice per {unit} ({stage}): {values}', flush=True)


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
    patch_size=(128, 128, 128),
    tile_step_size=0.5,
    prob_threshold=0.5,
    min_component_sizes=(161, 166, 50),
    closing_radius=1,
    predictor_class=None,
    export_prediction=None,
    subset='val',
):
    """Run tiled inference on a trainer's validation split and evaluate before/after PP."""
    import torch

    if predictor_class is None:
        from snn_nnunet.inference import FP32nnUNetPredictor

        predictor_class = FP32nnUNetPredictor
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

    from snn_nnunet.partition_runtime import evaluation_case_ids
    validation_keys = evaluation_case_ids(trainer, subset)
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
            "precision: FP32; Gaussian blending: on; mirroring: on",
            flush=True,
        )
        for index, case_id in enumerate(dataset_val.identifiers, start=1):
            case_started = perf_counter()
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
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite logits for {case_id}")
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
            metric_lines = []
            for stage, metrics in (("raw", raw_metrics), ("postprocessed", pp_metrics)):
                scores = " ".join(
                    f"{region}={metrics[region]['dice']:.4f}" for region in REGION_ORDER
                )
                mean_dice = sum(metrics[region]["dice"] for region in REGION_ORDER) / len(REGION_ORDER)
                metric_lines.append(f"  Dice {stage}: {scores} mean={mean_dice:.4f}")
            elapsed = perf_counter() - case_started
            print(
                f"[{index}/{len(dataset_val.identifiers)}] {case_id} completed in {elapsed:.1f}s\n"
                + "\n".join(metric_lines),
                flush=True,
            )

    dataset_base = getattr(trainer, 'preprocessed_dataset_folder_base', None)
    patient_wise = (subset == 'test' or (dataset_base is not None and
                    (Path(dataset_base) / 'patient_partition.json').is_file()))
    summary = summarize_cases(case_results, patient_wise=patient_wise)
    summary["settings"] = {
        "subset": subset,
        "precision": "float32",
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
    if patient_wise:
        write_patient_dice_tables(summary, output_dir)
    print_dice_summary(summary)
    return summary


def main(argv=None):
    import os
    import torch

    args = build_parser().parse_args(argv)
    args.results_root = args.results_root.resolve()
    if not args.results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {args.results_root}")
    from snn_nnunet.partition_runtime import restore_run_partition
    partition = restore_run_partition(args.results_root)
    if args.subset == 'test' and partition is None:
        raise ValueError('Held-out test requires a saved patient partition from a new training run')
    args.checkpoint_name = args.checkpoint_name or (
        'checkpoint_best.pth' if args.subset == 'test' else 'checkpoint_final.pth')

    # Rebuild the model from this run's metadata. The shared preprocessed plans
    # may have been replaced by a later experiment with a different architecture.
    model_name = f"{args.trainer}__{args.plans}__{args.configuration}"
    model_dirs = sorted(
        folder for folder in args.results_root.glob(f"Dataset{args.dataset_id:03d}_*/{model_name}")
        if folder.is_dir()
    )
    if len(model_dirs) != 1:
        raise RuntimeError(
            f"Expected one saved model for dataset {args.dataset_id} and {model_name} "
            f"under {args.results_root}; found {len(model_dirs)}: {model_dirs}"
        )
    model_dir = model_dirs[0]
    checkpoint = model_dir / f"fold_{args.fold}" / args.checkpoint_name
    if args.subset == 'test' and not (checkpoint.parent / 'checkpoint_final.pth').is_file():
        raise ValueError('Complete training before evaluating the held-out patient test')
    for required_file in (model_dir / "plans.json", model_dir / "dataset.json", checkpoint):
        if not required_file.is_file():
            raise FileNotFoundError(f"Required saved model file not found: {required_file}")
    with (model_dir / "plans.json").open(encoding="utf-8") as stream:
        plans = json.load(stream)
    with (model_dir / "dataset.json").open(encoding="utf-8") as stream:
        dataset_json = json.load(stream)
    if plans["dataset_name"] != model_dir.parent.name or plans["plans_name"] != args.plans:
        raise ValueError(f"Saved plans do not match the selected model directory: {model_dir}")
    if args.subset == 'test':
        # An epoch extension retains the earlier final checkpoint. Its existence
        # alone does not establish completion of the currently saved horizon.
        final_state = torch.load(checkpoint.parent / 'checkpoint_final.pth',
                                 map_location='cpu', weights_only=False)
        completed = final_state.get('current_epoch') if isinstance(final_state, dict) else None
        del final_state
        target = plans['snn_config'].get('num_epochs', 300) if 'snn_config' in plans else 1000
        if (type(completed) is not int or type(target) is not int or target < 1
                or completed < target):
            raise ValueError(f'Test requires training to be complete: final checkpoint '
                             f'completed {completed} epochs; current target is {target}')
    plans["continue_training"] = True
    if args.patch_size is None:
        args.patch_size = plans.get('snn_config', {}).get('validation_patch_size', (128, 128, 128))

    # Set the result root before importing nnU-Net so trainer paths and logs
    # belong to the selected run, even when the shell points to another run.
    os.environ["nnUNet_results"] = str(args.results_root)
    from snn_nnunet.cli import configure_runtime
    configure_runtime()
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer_class = recursive_find_trainer_class_by_name(args.trainer)
    print(f"Loading saved plans: {model_dir / 'plans.json'}", flush=True)
    trainer = trainer_class(
        plans=plans,
        configuration=args.configuration,
        fold=args.fold,
        dataset_json=dataset_json,
        device=device,
    )
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    trainer.load_checkpoint(str(checkpoint))
    print(f"Evaluation subset: {args.subset}; output: {args.output_dir}", flush=True)
    run_fold_validation(
        trainer,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        tile_step_size=args.tile_step_size,
        prob_threshold=args.prob_threshold,
        min_component_sizes=args.min_component_sizes,
        closing_radius=args.closing_radius,
        subset=args.subset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
