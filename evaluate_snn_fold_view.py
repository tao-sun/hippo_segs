#!/usr/bin/env python3
"""Evaluate one SNN checkpoint on one BraTS fold and anatomical view."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from postprocessing import (
    postprocess_brats_prediction,
    validate_postprocessing_parameters,
)


VALID_VIEWS = {"sagittal", "coronal", "axial"}
VALID_LABEL_FORMATS = {"auto", "brats17", "brats23", "brats24"}


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: Path
    data_root: Path
    cache_root: Optional[Path]
    cache_required: bool
    label_format: str
    val_fold: int
    view: str
    eval_batch_slices: int
    prob_threshold: float
    loader_workers: int
    loader_prefetch_factor: int
    subjects_per_fold: Optional[int]
    device: str
    evaluate_train: bool = False
    split: Optional[str] = None
    number_patients: Optional[int] = None
    apply_postprocessing: bool = False
    min_component_sizes: Tuple[int, int, int] = (0, 0, 0)
    closing_radius: int = 1


def _path_from_config(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def load_eval_config(config_path: Path) -> EvalConfig:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Evaluation YAML must contain a mapping")

    missing = sorted({"checkpoint", "data_root", "val_fold", "view"} - set(raw))
    if missing:
        raise ValueError(f"Evaluation YAML is missing: {', '.join(missing)}")

    fold = raw["val_fold"]
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(1, 6):
        raise ValueError("val_fold must be an integer from 1 to 5")
    view = raw["view"]
    if view not in VALID_VIEWS:
        raise ValueError(f"view must be one of {sorted(VALID_VIEWS)}")

    label_format = raw.get("label_format", "auto")
    if label_format not in VALID_LABEL_FORMATS:
        raise ValueError(f"label_format must be one of {sorted(VALID_LABEL_FORMATS)}")
    cache_required = raw.get("cache_required", False)
    if not isinstance(cache_required, bool):
        raise ValueError("cache_required must be a boolean")

    eval_batch_slices = raw.get("eval_batch_slices", 16)
    if isinstance(eval_batch_slices, bool) or not isinstance(eval_batch_slices, int) or eval_batch_slices <= 0:
        raise ValueError("eval_batch_slices must be a positive integer")
    threshold = raw.get("prob_threshold", 0.5)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("prob_threshold must be between 0 and 1")
    workers = raw.get("loader_workers", 2)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("loader_workers must be a non-negative integer")
    prefetch = raw.get("loader_prefetch_factor", 1)
    if isinstance(prefetch, bool) or not isinstance(prefetch, int) or prefetch <= 0:
        raise ValueError("loader_prefetch_factor must be a positive integer")
    device = raw.get("device", "auto")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be 'auto' or a valid torch device")
    subject_limit = raw.get("subjects_per_fold")
    if subject_limit is not None and (
        isinstance(subject_limit, bool)
        or not isinstance(subject_limit, int)
        or subject_limit <= 0
    ):
        raise ValueError("subjects_per_fold must be null or a positive integer")

    evaluate_train = raw.get("evaluate_train", False)
    if not isinstance(evaluate_train, bool):
        raise ValueError("evaluate_train must be a boolean")

    split = raw.get("split", "both" if evaluate_train else "testing")
    if split not in ("testing", "training", "both"):
        raise ValueError("split must be 'testing', 'training', or 'both'")
    number_patients = raw.get("number_patients")
    if number_patients is not None and (
        isinstance(number_patients, bool) or not isinstance(number_patients, int)
        or number_patients <= 0
    ):
        raise ValueError("number_patients must be null or a positive integer")
    # Explicit new controls supersede the old evaluation-only fold limit.
    if "number_patients" in raw:
        subject_limit = None

    apply_postprocessing, min_component_sizes, closing_radius = (
        validate_postprocessing_parameters(
            raw.get("apply_postprocessing", False),
            raw.get("min_component_sizes", (0, 0, 0)),
            raw.get("closing_radius", 1),
        )
    )

    config_dir = config_path.parent
    cache_value = raw.get("cache_root")
    cache_root = _path_from_config(cache_value, config_dir) if cache_value else None
    if cache_required and cache_root is None:
        raise ValueError("cache_required=true requires cache_root")

    return EvalConfig(
        checkpoint=_path_from_config(raw["checkpoint"], config_dir),
        data_root=_path_from_config(raw["data_root"], config_dir),
        cache_root=cache_root,
        cache_required=cache_required,
        label_format=label_format,
        val_fold=fold,
        view=view,
        eval_batch_slices=eval_batch_slices,
        prob_threshold=float(threshold),
        loader_workers=workers,
        loader_prefetch_factor=prefetch,
        subjects_per_fold=subject_limit,
        device=device.strip(),
        evaluate_train=evaluate_train,
        split=split,
        number_patients=number_patients,
        apply_postprocessing=apply_postprocessing,
        min_component_sizes=min_component_sizes,
        closing_radius=closing_radius,
    )


def read_model_spec(payload: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError("Checkpoint must contain a 'model' state dict")
    model_config = payload.get("config")
    if not isinstance(model_config, Mapping):
        raise ValueError("Checkpoint must contain a 'config' mapping")

    names = (
        "model",
        "patch_size",
        "linear_projection",
        "residual_connections",
        "dwconv2d_spiking",
        "patch_embedding_spiking",
    )
    missing = [name for name in names if name not in model_config]
    if missing:
        raise ValueError(f"Checkpoint config is missing: {', '.join(missing)}")

    kwargs = {
        "model_name": model_config["model"],
        "out_channels": 3,
        "patch_size": model_config["patch_size"],
        "linear_projection": model_config["linear_projection"],
        "residual_connections": model_config["residual_connections"],
        "dwconv2d_spiking": model_config["dwconv2d_spiking"],
        "patch_embedding_spiking": model_config["patch_embedding_spiking"],
        "input_skip": model_config.get("input_skip", False),
    }
    return payload["model"], kwargs


def _stack_slices(slices: np.ndarray, view: str, shape: Tuple[int, int, int]) -> np.ndarray:
    if view == "sagittal":
        volume = np.moveaxis(slices.transpose(1, 2, 3, 0), -1, 1)
    elif view == "coronal":
        volume = np.moveaxis(slices.transpose(1, 2, 3, 0), -1, 2)
    else:
        volume = slices.transpose(1, 2, 3, 0)
    return volume[:, : shape[0], : shape[1], : shape[2]]


def _dice_per_channel(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    scores = []
    for channel in range(prediction.shape[0]):
        pred = prediction[channel].reshape(-1).astype(np.uint8)
        truth = target[channel].reshape(-1).astype(np.uint8)
        intersection = (pred & truth).sum()
        scores.append(float((2 * intersection + 1e-6) / (pred.sum() + truth.sum() + 1e-6)))
    return np.asarray(scores, dtype=np.float64)


def _metadata_shape(xyz) -> Tuple[int, int, int]:
    values = []
    for value in xyz:
        if isinstance(value, torch.Tensor):
            value = value.reshape(-1)[0].item()
        values.append(int(value))
    return tuple(values)


def _metadata_labels(raw_labels) -> list:
    if isinstance(raw_labels, torch.Tensor):
        return sorted({int(value) for value in raw_labels.reshape(-1).tolist()})

    values = []
    for value in raw_labels:
        if isinstance(value, torch.Tensor):
            values.extend(int(item) for item in value.reshape(-1).tolist())
        else:
            values.append(int(value))
    return sorted(set(values))


@torch.no_grad()
def evaluate_loader(
    model, loader, device, view: str, window_size: int, threshold: float,
    apply_postprocessing: bool = False,
    min_component_sizes: Tuple[int, int, int] = (0, 0, 0),
    closing_radius: int = 1,
) -> Dict[str, Any]:
    model.eval()
    subject_metrics = []

    for inputs, targets, metadata in loader:
        if inputs.shape[0] != 1:
            raise ValueError("Evaluation requires DataLoader batch_size=1")
        inputs = inputs.to(device)
        num_slices = inputs.shape[1]
        probability_windows = []

        for start in range(0, num_slices, window_size):
            stop = min(start + window_size, num_slices)
            logits = model(inputs[:, start:stop], t0=start)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            probability_windows.append(np.transpose(probabilities, (0, 2, 1, 3, 4))[0])

        slice_probabilities = np.concatenate(probability_windows, axis=0)
        slice_targets = targets.cpu().numpy()[0]
        shape = _metadata_shape(metadata["xyz"])
        volume_probabilities = _stack_slices(slice_probabilities, view, shape)
        volume_targets = _stack_slices(slice_targets, view, shape).astype(np.uint8)
        volume_prediction = (volume_probabilities >= threshold).astype(np.uint8)
        if apply_postprocessing:
            volume_prediction = postprocess_brats_prediction(
                volume_prediction,
                min_component_sizes=min_component_sizes,
                closing_radius=closing_radius,
            )
        dice = _dice_per_channel(volume_prediction, volume_targets)
        gt_voxel_counts = volume_targets.reshape(3, -1).sum(axis=1)

        subject_id = metadata["sid"]
        if isinstance(subject_id, (list, tuple)):
            subject_id = subject_id[0]
        row = {
            "subject": str(subject_id),
            "gt_raw_labels": _metadata_labels(metadata["raw_labels"]),
            "gt_voxels": {
                "ET": int(gt_voxel_counts[0]),
                "TC": int(gt_voxel_counts[1]),
                "WT": int(gt_voxel_counts[2]),
            },
            "dice_ET": float(dice[0]),
            "dice_TC": float(dice[1]),
            "dice_WT": float(dice[2]),
            "dice_mean": float(dice.mean()),
        }
        subject_metrics.append(row)

    if not subject_metrics:
        raise RuntimeError("The selected fold contains no subjects")
    means = np.asarray(
        [[row["dice_ET"], row["dice_TC"], row["dice_WT"]] for row in subject_metrics],
        dtype=np.float64,
    ).mean(axis=0)
    return {
        "subjects": subject_metrics,
        "dice_ET": float(means[0]),
        "dice_TC": float(means[1]),
        "dice_WT": float(means[2]),
        "dice_mean": float(means.mean()),
        "n_subjects": len(subject_metrics),
    }


def build_training_dataset(config: EvalConfig, checkpoint_config: Mapping[str, Any], dataset_factory):
    """Reconstruct training membership using the checkpoint's original limits."""
    fold = checkpoint_config.get("val_fold")
    if fold != config.val_fold:
        raise ValueError("Training comparison requires val_fold to match the checkpoint")
    if checkpoint_config.get("view") != config.view:
        raise ValueError("Training comparison requires view to match the checkpoint")
    datasets = [
        dataset_factory(
            root=str(config.data_root), val_fold=f, view=config.view,
            subjects_per_fold=checkpoint_config.get("subjects_per_fold"),
            cache_root=str(config.cache_root) if config.cache_root is not None else None,
            cache_required=config.cache_required, label_format=config.label_format,
        )
        for f in range(1, 6) if f != fold
    ]
    dataset = ConcatDataset(datasets)
    overfit_subjects = checkpoint_config.get("overfit_subjects")
    if overfit_subjects is not None:
        if not 0 < overfit_subjects <= len(dataset):
            raise ValueError("Checkpoint overfit_subjects exceeds available training subjects")
        dataset = Subset(dataset, range(overfit_subjects))
    return dataset


def run_evaluation(config: EvalConfig) -> Dict[str, Any]:
    if not config.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is not available: {config.device}")

    payload = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    state_dict, model_kwargs = read_model_spec(payload)

    from snn_fptt import BratsVolumeDataset, build_model

    split = config.split or ("both" if config.evaluate_train else "testing")
    if split not in ("testing", "training", "both"):
        raise ValueError("split must be 'testing', 'training', or 'both'")
    selected_splits = ("testing", "training") if split == "both" else (split,)
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": config.loader_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.loader_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=config.loader_prefetch_factor,
        )

    model = build_model(**model_kwargs)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    print(f"Device: {device}")
    print(f"Checkpoint: {config.checkpoint}")
    results = {}
    for selected in selected_splits:
        if selected == "training":
            dataset = build_training_dataset(config, payload["config"], BratsVolumeDataset)
        else:
            dataset = BratsVolumeDataset(
                root=str(config.data_root), val_fold=config.val_fold, view=config.view,
                subjects_per_fold=config.subjects_per_fold,
                cache_root=str(config.cache_root) if config.cache_root is not None else None,
                cache_required=config.cache_required, label_format=config.label_format,
            )
        available = len(dataset)
        if config.number_patients is not None:
            dataset = Subset(dataset, range(min(config.number_patients, available)))
        print(f"Split: {selected} | View: {config.view} | Subjects: {len(dataset)}/{available}")
        loader = DataLoader(dataset, **loader_kwargs)
        progress = tqdm(loader, desc=f"{selected} {config.view}")
        results[selected] = evaluate_loader(
            model, progress, device=device, view=config.view,
            window_size=config.eval_batch_slices, threshold=config.prob_threshold,
            apply_postprocessing=config.apply_postprocessing,
            min_component_sizes=config.min_component_sizes,
            closing_radius=config.closing_radius,
        )
        # Free this split's worker processes before creating the next loader.
        del progress, loader
    if split != "both":
        return {**results[split], "split": split}
    metrics = {**results["testing"], "split": "both", "training": results["training"]}
    metrics["train_minus_validation"] = {
        key: results["training"][key] - results["testing"][key]
        for key in ("dice_ET", "dice_TC", "dice_WT", "dice_mean")
    }
    return metrics


def print_metrics(metrics: Mapping[str, Any]) -> None:
    if "training" in metrics:
        print("Testing / Validation (held-out fold)")
    elif metrics.get("split") == "training":
        print("Training")
    elif metrics.get("split") == "testing":
        print("Testing / Validation (held-out fold)")
    for row in metrics["subjects"]:
        gt_voxels = row["gt_voxels"]
        print(
            f"{row['subject']} | "
            f"GT raw labels={row['gt_raw_labels']} | "
            f"GT voxels: ET={gt_voxels['ET']} "
            f"TC={gt_voxels['TC']} WT={gt_voxels['WT']} | "
            f"Dice: ET={row['dice_ET']:.4f} "
            f"TC={row['dice_TC']:.4f} "
            f"WT={row['dice_WT']:.4f} "
            f"mean={row['dice_mean']:.4f}"
        )
    print(
        f"Mean over {metrics['n_subjects']} subjects | "
        f"ET={metrics['dice_ET']:.4f} "
        f"TC={metrics['dice_TC']:.4f} "
        f"WT={metrics['dice_WT']:.4f} "
        f"mean={metrics['dice_mean']:.4f}"
    )

    if "training" in metrics:
        print("Training")
        print_metrics(metrics["training"])
        gap = metrics["train_minus_validation"]
        print(
            "Dice gap (training - validation) | "
            f"ET={gap['dice_ET']:+.4f} TC={gap['dice_TC']:+.4f} "
            f"WT={gap['dice_WT']:+.4f} mean={gap['dice_mean']:+.4f}"
        )
        print("A positive gap indicates better training performance; interpret alongside validation trends.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Dice for one SNN checkpoint, fold, and view."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML file containing checkpoint, dataset, fold, and view settings.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_eval_config(args.config)
    metrics = run_evaluation(config)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
