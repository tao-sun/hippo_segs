"""Start an isolated SNN or stock nnU-Net training run from YAML."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


NATIVE_CONFIGURATIONS = {
    "unet2d": "2d",
    "unet3d": "3d_fullres",
}


def _integer(config: dict[str, Any], name: str, minimum: int, maximum: int) -> int:
    value = config.get(name)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The experiment file must contain a YAML mapping")
    if config.get("family") not in {"unet2d", "unet3d", "snn"}:
        raise ValueError("family must be unet2d, unet3d, or snn")
    _integer(config, "dataset_id", 0, 999)
    _integer(config, "fold", 0, 4)
    _integer(config, "gpus", 1, 64)
    if os.environ.get("SLURM_JOB_ID"):
        import torch

        allocated = torch.cuda.device_count()
        if allocated < 1:
            raise ValueError("CUDA non disponibile nel job SLURM")
        config["gpus"] = allocated
    if not isinstance(config.get("run_dir"), str) or not config["run_dir"].strip():
        raise ValueError("run_dir must be a nonempty path")
    if config["family"] == "snn":
        snn = config.get("snn")
        if not isinstance(snn, dict):
            raise ValueError("snn settings must be a YAML mapping")
        from snn_nnunet.config_layout import normalize_snn_layout
        snn = config["snn"] = normalize_snn_layout(snn)
        if snn.get("variant") not in {"orig", "shallow", "medium", "deep"}:
            raise ValueError("snn.variant must be orig, shallow, medium, or deep")
        _integer(snn, "temporal_axis", 0, 2)
        _integer(snn, "k", 1, 4096)
        snn.setdefault("lr", 0.01)
        if type(snn["lr"]) not in (int, float) or not math.isfinite(snn["lr"]) or snn["lr"] <= 0:
            raise ValueError("snn.lr must be a positive finite number")
        snn["lr"] = float(snn["lr"])
        snn.setdefault("epochs", 300)
        if type(snn["epochs"]) is not int or snn["epochs"] <= 0:
            raise ValueError("snn.epochs must be a positive integer")
        training_patch = snn.get("training_patch_size")
        if training_patch is not None and (
                not isinstance(training_patch, list) or len(training_patch) != 3
                or any(type(value) is not int or value <= 0 for value in training_patch)):
            raise ValueError("snn.training_patch_size must contain three positive integers")
        snn.setdefault("validation_every", 0)
        if type(snn["validation_every"]) is not int or snn["validation_every"] < 0:
            raise ValueError("snn.validation_every must be a nonnegative integer")
        snn.setdefault("validation_patch_size", [128, 128, 128])
        patch = snn["validation_patch_size"]
        if (not isinstance(patch, list) or len(patch) != 3
                or any(type(value) is not int or value <= 0 for value in patch)):
            raise ValueError("snn.validation_patch_size must contain three positive integers")
        if type(snn.get("use_fptt")) is not bool:
            raise ValueError("snn.use_fptt must be true or false")
        from snn_nnunet.model import output_spiking_option
        snn["output_spiking"] = output_spiking_option(snn)
        snn.pop("vss_output_spiking", None)
        spatial = snn.get("spatial_mamba", {})
        if not isinstance(spatial, dict):
            raise ValueError("snn.spatial_mamba must be a YAML mapping")
        spatial_defaults = {
            "patch_embedding_spiking": True,
            "dwconv2d_spiking": True,
            "output_spiking": False,  # Legacy compatibility; SS2D output has no activation.
        }
        from snn_nnunet.model import ARCHITECTURE_DEFAULTS, architecture_options
        from snn_nnunet.config_aliases import normalize_spiking_options
        spatial = normalize_spiking_options(spatial)
        if spatial.keys() - (spatial_defaults.keys() | ARCHITECTURE_DEFAULTS.keys()):
            raise ValueError("snn.spatial_mamba has unknown options")
        for key in spatial_defaults:
            if key in spatial:
                spatial_defaults[key] = spatial[key]
        for key, value in spatial_defaults.items():
            if type(value) is not bool:
                raise ValueError(f"snn.spatial_mamba.{key} must be true or false")
        spatial_defaults.update(architecture_options(spatial))
        if spatial_defaults["decoder_block"] == "resdecoder" and snn["variant"] != "orig":
            raise ValueError("decoder_block=resdecoder requires snn.variant: orig")
        if any(spatial_defaults[key] for key in ("fue_skip1", "fue_skip2")) and snn["variant"] != "orig":
            raise ValueError("FUE skips require snn.variant: orig")
        if snn["variant"] != "orig" and any(
                spatial_defaults[key] is not None
                for key in ("encoder_output_spiking", "decoder_conv_spiking")):
            raise ValueError("Stage spiking controls require snn.variant: orig")
        if spatial_defaults["recurrent_neuron"] is not None and snn["variant"] != "orig":
            raise ValueError("recurrent_neuron requires snn.variant: orig")
        snn["spatial_mamba"] = spatial_defaults
        temporal = snn.get("temporal_mamba", {})
        if not isinstance(temporal, dict):
            raise ValueError("snn.temporal_mamba must be a YAML mapping")
        defaults = {
            "enabled": False, "kernel_size": 3, "d_state": 16,
            "dt_rank": "auto", "debug_shapes": False,
            "norm_spiking": True, "dwconv1D_spiking": True,
        }
        if temporal.keys() - (defaults.keys() | {"plif_enabled"}):
            raise ValueError("snn.temporal_mamba has unknown options")
        legacy_plif = temporal.pop("plif_enabled", None)
        if legacy_plif is not None:
            if type(legacy_plif) is not bool:
                raise ValueError("snn.temporal_mamba.plif_enabled must be true or false")
            defaults["norm_spiking"] = legacy_plif
            defaults["dwconv1D_spiking"] = legacy_plif
        defaults.update(temporal)
        if type(defaults["enabled"]) is not bool:
            raise ValueError("snn.temporal_mamba.enabled must be true or false")
        for key in ("debug_shapes", "norm_spiking", "dwconv1D_spiking"):
            if type(defaults[key]) is not bool:
                raise ValueError(f"snn.temporal_mamba.{key} must be true or false")
        for key in ("kernel_size", "d_state"):
            if type(defaults[key]) is not int or defaults[key] <= 0:
                raise ValueError(f"snn.temporal_mamba.{key} must be a positive integer")
        rank = defaults["dt_rank"]
        if rank != "auto" and (type(rank) is not int or rank <= 0):
            raise ValueError("snn.temporal_mamba.dt_rank must be 'auto' or a positive integer")
        if defaults["enabled"] and snn["variant"] != "orig":
            raise ValueError("snn.temporal_mamba.enabled=true requires snn.variant: orig")
        snn["temporal_mamba"] = defaults
    return config


def _dataset_folder(preprocessed: Path, dataset_id: int) -> Path:
    matches = sorted(preprocessed.glob(f"Dataset{dataset_id:03d}_*"))
    if len(matches) != 1 or not matches[0].is_dir():
        raise ValueError(
            f"Expected one preprocessed Dataset{dataset_id:03d}_* folder in {preprocessed}"
        )
    return matches[0]


def _require_preprocessed_configuration(
    dataset_folder: Path, plans: str, configuration: str
) -> None:
    plans_path = dataset_folder / f"{plans}.json"
    if not plans_path.is_file():
        raise ValueError(f"Missing plans: {plans_path}")
    with plans_path.open(encoding="utf-8") as stream:
        plans_data = json.load(stream)
    configuration_data = plans_data.get("configurations", {}).get(configuration)
    if not isinstance(configuration_data, dict):
        raise ValueError(
            f"Configuration '{configuration}' is missing from plans: {plans_path}"
        )
    data_identifier = configuration_data.get("data_identifier")
    if not isinstance(data_identifier, str) or not data_identifier:
        raise ValueError(
            f"Configuration '{configuration}' has no data_identifier in {plans_path}"
        )
    data_folder = dataset_folder / data_identifier
    if not data_folder.is_dir():
        raise ValueError(
            f"Missing preprocessed data for configuration '{configuration}': "
            f"{data_folder}. Run nnUNetv2_preprocess with -c {configuration}."
        )


def run(config_path: Path) -> None:
    config_path = config_path.expanduser().resolve()
    config = _config(config_path)
    family = config["family"]
    dataset_id = config["dataset_id"]
    fold = config["fold"]
    gpus = config["gpus"]
    run_dir = Path(config["run_dir"]).expanduser().resolve()

    missing = [name for name in ("nnUNet_raw", "nnUNet_preprocessed") if not os.environ.get(name)]
    if missing:
        raise ValueError("Set these nnU-Net paths before training: " + ", ".join(missing))
    preprocessed = Path(os.environ["nnUNet_preprocessed"]).expanduser().resolve()
    dataset_folder = _dataset_folder(preprocessed, dataset_id)
    plans = "SNNPlans" if family == "snn" else "nnUNetPlans"
    configuration = "3d_fullres" if family == "snn" else NATIVE_CONFIGURATIONS[family]
    _require_preprocessed_configuration(dataset_folder, plans, configuration)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Run directory already contains files: {run_dir}. Choose a new run_dir.")

    run_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open(encoding="utf-8") as stream:
        saved_config = yaml.safe_load(stream)
    saved_config["gpus"] = gpus
    (run_dir / "experiment.yaml").write_text(yaml.safe_dump(saved_config, sort_keys=False))
    os.environ["nnUNet_results"] = str(run_dir)
    print(
        f"Model family: {family}; configuration: {configuration}; "
        f"dataset: {dataset_id}; fold: {fold}; GPUs: {gpus}",
        flush=True,
    )
    print(f"Run directory: {run_dir}", flush=True)

    if family == "snn":
        from snn_nnunet import cli

        snn = config["snn"]
        architecture_args = ["--recurrent-neuron", json.dumps(snn["spatial_mamba"]["recurrent_neuron"])]
        if snn.get("training_patch_size") is not None:
            architecture_args.extend(["--training-patch-size", *map(str, snn["training_patch_size"])])
        architecture_args.extend(["--silif", json.dumps(snn["spatial_mamba"]["silif"])])
        for key in ("stem_depth", "stem_channels", "downsample_kernel_size", "dwconv2d_kernel_size",
                    "final_skip_fusion", "decoder_block", "neuron_model"):
            architecture_args.extend(["--" + key.replace("_", "-"), str(snn["spatial_mamba"][key])])
        for key in ("encoder_channels", "vss_depths", "downsample_strides"):
            architecture_args.extend(["--" + key.replace("_", "-"),
                                      *map(str, snn["spatial_mamba"][key])])
        for key in ("hierarchical_encoder", "full_resolution_skip", "stem_spiking", "downsampling_spiking",
                    "resdecoder_spiking1", "resdecoder_spiking2", "fue_skip1", "fue_skip2"):
            architecture_args.append(("--" if snn["spatial_mamba"][key] else "--no-") + key.replace("_", "-"))
        if snn["variant"] == "orig":
            for key in ("encoder_output_spiking", "decoder_conv_spiking"):
                enabled = snn["spatial_mamba"][key]
                if enabled is None:
                    enabled = snn["output_spiking"]
                architecture_args.append(("--" if enabled else "--no-") + key.replace("_", "-"))
        cli.main([
            "train", "--dataset-id", str(dataset_id), "--fold", str(fold),
            "--run-dir", str(run_dir), "--gpus", str(gpus),
            "--model", snn["variant"],
            "--temporal-axis", str(snn["temporal_axis"]),
            "--k", str(snn["k"]),
            "--epochs", str(snn["epochs"]),
            "--lr", str(snn["lr"]),
            "--validation-every", str(snn["validation_every"]),
            "--validation-patch-size", *map(str, snn["validation_patch_size"]),
            "--fptt" if snn["use_fptt"] else "--no-fptt",
            "--output-spiking" if snn["output_spiking"]
            else "--no-output-spiking",
            "--temporal-mamba" if snn["temporal_mamba"]["enabled"] else "--no-temporal-mamba",
            "--temporal-mamba-kernel-size", str(snn["temporal_mamba"]["kernel_size"]),
            "--temporal-mamba-d-state", str(snn["temporal_mamba"]["d_state"]),
            "--temporal-mamba-dt-rank", str(snn["temporal_mamba"]["dt_rank"]),
            "--temporal-mamba-norm-spiking" if snn["temporal_mamba"]["norm_spiking"]
            else "--no-temporal-mamba-norm-spiking",
            "--temporal-mamba-dwconv1d-spiking" if snn["temporal_mamba"]["dwconv1D_spiking"]
            else "--no-temporal-mamba-dwconv1d-spiking",
            "--temporal-mamba-debug-shapes" if snn["temporal_mamba"]["debug_shapes"]
            else "--no-temporal-mamba-debug-shapes",
            "--spatial-patch-embedding-spiking" if snn["spatial_mamba"]["patch_embedding_spiking"]
            else "--no-spatial-patch-embedding-spiking",
            "--spatial-dwconv2d-spiking" if snn["spatial_mamba"]["dwconv2d_spiking"]
            else "--no-spatial-dwconv2d-spiking",
            "--spatial-output-spiking" if snn["spatial_mamba"]["output_spiking"]
            else "--no-spatial-output-spiking",
            *architecture_args,
        ])
    else:
        from nnunetv2.run.run_training import run_training
        from nnunetv2.utilities.file_path_utilities import get_output_folder
        from snn_nnunet.empty_dice_evaluation import finalize_fold_summary, native_empty_dice_evaluation

        with native_empty_dice_evaluation():
            run_training(
                dataset_name_or_id=str(dataset_id),
                configuration=configuration,
                fold=fold,
                trainer_class_name="nnUNetTrainer",
                plans_identifier="nnUNetPlans",
                num_gpus=gpus,
            )
        model_folder = Path(get_output_folder(
            dataset_id, "nnUNetTrainer", "nnUNetPlans", configuration
        ))
        summary_path = model_folder / f"fold_{fold}" / "validation" / "summary.json"
        if summary_path.is_file():
            metrics = finalize_fold_summary(model_folder, fold)
            print(
                "Mean Validation Dice (empty-empty=1): "
                f"{metrics['foreground_mean']['Dice']}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML experiment file")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
