"""Delegate dataset preparation to nnU-Net and derive SNN training plans."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprints,
    plan_experiments,
    preprocess,
)
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from snn_nnunet.network_adapter import SNNConfig


PLANS_IDENTIFIER = "SNNPlans"
DEFAULT_SNN_CONFIG: dict[str, Any] = {
    "model_name": "orig",
    "model_kwargs": {
        "patch_size": 4,
        "linear_projection": True,
        "residual_connections": True,
        "dwconv2d_spiking": True,
        "patch_embedding_spiking": True,
        "input_skip": False,
        "output_spiking": True,
    },
    "temporal_axis": 0,
    "k": 16,
    "use_fptt": True,
    "fptt_alpha": 0.5,
    "fptt_beta": 0.5,
    "fptt_rho": 0.0,
    "fptt_lambda": 2.0,
    "num_input_channels": 4,
    "num_output_channels": 3,
}


def _validated_config(snn_config: Mapping[str, Any]) -> dict[str, Any]:
    return SNNConfig.from_plans({"snn_config": snn_config}).to_dict()


def derive_snn_plans(stock: Mapping[str, Any], snn_config: Mapping[str, Any]) -> dict:
    """Copy native plans, changing only the agreed SNN fields."""

    if "3d_fullres" not in stock.get("configurations", {}):
        raise ValueError("Stock plans must contain a 3d_fullres configuration")
    result = deepcopy(dict(stock))
    result["plans_name"] = PLANS_IDENTIFIER
    result["configurations"]["3d_fullres"]["patch_size"] = [128, 128, 128]
    result["configurations"]["3d_fullres"]["batch_size"] = 4
    result["snn_config"] = _validated_config(snn_config)
    return result


def _atomic_write_json(destination: Path, data: Mapping[str, Any]) -> None:
    """Replace one JSON file without exposing a partially written version."""

    destination = Path(destination)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".tmp", prefix=f".{destination.name}.",
            dir=destination.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _dataset_folder(dataset_id: int) -> Path:
    return Path(nnUNet_preprocessed.require()) / convert_id_to_dataset_name(dataset_id)


def create_snn_plans(
    dataset_id: int, stock_identifier: str, snn_config: Mapping[str, Any]
) -> Path:
    """Read a native plans file and atomically write its SNN derivative."""

    folder = _dataset_folder(dataset_id)
    with (folder / f"{stock_identifier}.json").open(encoding="utf-8") as stream:
        stock = json.load(stream)
    derived = derive_snn_plans(stock, snn_config)
    destination = folder / f"{PLANS_IDENTIFIER}.json"
    _atomic_write_json(destination, derived)
    return destination


def run_native_prepare(
    dataset_id: int, fingerprint_processes: int, preprocess_processes: int
) -> Path:
    """Run verified native fingerprinting, stock planning, and fullres preprocessing."""

    extract_fingerprints(
        [dataset_id], num_processes=fingerprint_processes, check_dataset_integrity=True
    )
    stock_identifier = plan_experiments([dataset_id])
    preprocess(
        [dataset_id], plans_identifier=stock_identifier,
        configurations=("3d_fullres",), num_processes=(preprocess_processes,),
    )
    destination = create_snn_plans(dataset_id, stock_identifier, DEFAULT_SNN_CONFIG)
    with destination.open(encoding="utf-8") as stream:
        derived = json.load(stream)
    data_identifier = derived["configurations"]["3d_fullres"]["data_identifier"]
    data_folder = destination.parent / data_identifier
    if not data_folder.is_dir():
        raise FileNotFoundError(f"Native preprocessed data folder is missing: {data_folder}")
    return destination


def validate_results_compatibility(
    results_folder: Path, requested_plans: Mapping[str, Any]
) -> None:
    """Refuse a result directory already bound to different plans."""

    folder = Path(results_folder)
    existing_plans = folder / "plans.json"
    requested = deepcopy(dict(requested_plans))
    requested["snn_config"] = _validated_config(requested["snn_config"])
    if existing_plans.is_file():
        with existing_plans.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        try:
            existing["snn_config"] = _validated_config(existing["snn_config"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Existing results plans are incompatible: {existing_plans}") from exc
        if existing != requested:
            raise ValueError(f"Existing results plans are incompatible: {existing_plans}")
    elif folder.is_dir() and any(folder.rglob("checkpoint_*.pth")):
        raise ValueError(f"Existing results checkpoints have no plans to verify: {folder}")


def update_snn_config(
    plans_path: Path,
    snn_config: Mapping[str, Any],
    results_folder: Path | None = None,
) -> Path:
    """Validate result compatibility and atomically persist a complete SNN config."""

    plans_path = Path(plans_path)
    with plans_path.open(encoding="utf-8") as stream:
        plans = json.load(stream)
    plans["snn_config"] = _validated_config(snn_config)
    if results_folder is not None:
        validate_results_compatibility(results_folder, plans)
    _atomic_write_json(plans_path, plans)
    return plans_path
