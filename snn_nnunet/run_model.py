"""Start an isolated SNN or stock 3D nnU-Net training run from YAML."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Any

import yaml


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
    if config.get("family") not in {"snn", "unet3d"}:
        raise ValueError("family must be 'snn' or 'unet3d'")
    _integer(config, "dataset_id", 0, 999)
    _integer(config, "fold", 0, 4)
    _integer(config, "gpus", 1, 64)
    if not isinstance(config.get("run_dir"), str) or not config["run_dir"].strip():
        raise ValueError("run_dir must be a nonempty path")
    if config["family"] == "snn":
        snn = config.get("snn")
        if not isinstance(snn, dict):
            raise ValueError("snn settings must be a YAML mapping")
        if snn.get("variant") not in {"orig", "shallow", "medium", "deep"}:
            raise ValueError("snn.variant must be orig, shallow, medium, or deep")
        _integer(snn, "temporal_axis", 0, 2)
        _integer(snn, "k", 1, 4096)
        if type(snn.get("use_fptt")) is not bool:
            raise ValueError("snn.use_fptt must be true or false")
    return config


def _dataset_folder(preprocessed: Path, dataset_id: int) -> Path:
    matches = sorted(preprocessed.glob(f"Dataset{dataset_id:03d}_*"))
    if len(matches) != 1 or not matches[0].is_dir():
        raise ValueError(
            f"Expected one preprocessed Dataset{dataset_id:03d}_* folder in {preprocessed}"
        )
    return matches[0]


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
    if not (dataset_folder / f"{plans}.json").is_file():
        raise ValueError(f"Missing plans: {dataset_folder / (plans + '.json')}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Run directory already contains files: {run_dir}. Choose a new run_dir.")

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "experiment.yaml")
    os.environ["nnUNet_results"] = str(run_dir)
    print(f"Model family: {family}; dataset: {dataset_id}; fold: {fold}; GPUs: {gpus}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)

    if family == "snn":
        from snn_nnunet import cli

        snn = config["snn"]
        cli.main([
            "train", "--dataset-id", str(dataset_id), "--fold", str(fold),
            "--run-dir", str(run_dir), "--gpus", str(gpus),
            "--model", snn["variant"],
            "--temporal-axis", str(snn["temporal_axis"]),
            "--k", str(snn["k"]),
            "--fptt" if snn["use_fptt"] else "--no-fptt",
        ])
    else:
        from nnunetv2.run.run_training import run_training

        run_training(
            dataset_name_or_id=str(dataset_id),
            configuration="3d_fullres",
            fold=fold,
            trainer_class_name="nnUNetTrainer",
            plans_identifier="nnUNetPlans",
            num_gpus=gpus,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML experiment file")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
