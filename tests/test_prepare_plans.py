"""Contract tests for deriving plans while reusing native nnU-Net preprocessing."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from snn_nnunet import prepare_plans


CONFIG = {
    "model_name": "orig",
    "model_kwargs": {
        "patch_size": 4,
        "linear_projection": True,
        "residual_connections": True,
        "dwconv2d_spiking": True,
        "patch_embedding_spiking": True,
        "vss_output_spiking": True,
        "input_skip": False,
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


@pytest.fixture
def stock_plans():
    return {
        "dataset_name": "Dataset011_BraTS",
        "plans_name": "StockCustom",
        "transpose_forward": [2, 0, 1],
        "image_reader_writer": "SimpleITKIO",
        "configurations": {
            "3d_fullres": {
                "data_identifier": "StockCustom_3d_fullres",
                "patch_size": [96, 112, 80],
                "batch_size": 2,
                "spacing": [1.1, 1.2, 1.3],
                "normalization_schemes": ["ZScoreNormalization"] * 4,
                "resampling_fn_data": "resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": {"order": 3, "order_z": 0},
                "architecture": {"network_class_name": "PlainConvUNet"},
            },
            "2d": {"data_identifier": "StockCustom_2d", "patch_size": [128, 128]},
        },
    }


def test_derive_changes_only_approved_fields_and_does_not_mutate_stock(stock_plans):
    original = deepcopy(stock_plans)
    derived = prepare_plans.derive_snn_plans(stock_plans, CONFIG)
    expected = deepcopy(original)
    expected["plans_name"] = "SNNPlans"
    expected["configurations"]["3d_fullres"]["patch_size"] = [128, 128, 128]
    expected["configurations"]["3d_fullres"]["batch_size"] = 4
    expected["snn_config"] = CONFIG
    assert derived == expected
    assert stock_plans == original
    assert derived["configurations"]["3d_fullres"]["data_identifier"] == "StockCustom_3d_fullres"
    derived["configurations"]["3d_fullres"]["resampling_fn_data_kwargs"]["order"] = 1
    assert stock_plans["configurations"]["3d_fullres"]["resampling_fn_data_kwargs"]["order"] == 3


def test_derive_rejects_missing_fullres_and_invalid_config(stock_plans):
    del stock_plans["configurations"]["3d_fullres"]
    with pytest.raises(ValueError, match="3d_fullres"):
        prepare_plans.derive_snn_plans(stock_plans, CONFIG)
    stock_plans["configurations"]["3d_fullres"] = {"data_identifier": "stock"}
    with pytest.raises(ValueError, match="k"):
        prepare_plans.derive_snn_plans(stock_plans, {**CONFIG, "k": 0})


def test_create_snn_plans_uses_stock_identifier_and_atomic_replace(tmp_path, monkeypatch, stock_plans):
    dataset_dir = tmp_path / "Dataset011_BraTS"
    dataset_dir.mkdir()
    (dataset_dir / "StockCustom.json").write_text(json.dumps(stock_plans))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path))
    monkeypatch.setattr(prepare_plans, "convert_id_to_dataset_name", lambda dataset_id: "Dataset011_BraTS")
    destination = prepare_plans.create_snn_plans(11, "StockCustom", CONFIG)
    assert destination == dataset_dir / "SNNPlans.json"
    assert json.loads(destination.read_text())["plans_name"] == "SNNPlans"
    assert list(dataset_dir.iterdir()) == [dataset_dir / "StockCustom.json", destination]


def test_native_prepare_delegates_verified_stock_flow_and_checks_shared_folder(
    tmp_path, monkeypatch, stock_plans
):
    dataset_dir = tmp_path / "Dataset011_BraTS"
    dataset_dir.mkdir()
    (dataset_dir / "StockCustom.json").write_text(json.dumps(stock_plans))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path))
    monkeypatch.setattr(prepare_plans, "convert_id_to_dataset_name", lambda dataset_id: "Dataset011_BraTS")
    calls = []

    def fingerprints(ids, **kwargs):
        calls.append(("fingerprints", ids, kwargs))

    def planning(ids, **kwargs):
        calls.append(("planning", ids, kwargs))
        return "StockCustom"

    def preprocessing(ids, **kwargs):
        calls.append(("preprocessing", ids, kwargs))
        (dataset_dir / "StockCustom_3d_fullres").mkdir()

    monkeypatch.setattr(prepare_plans, "extract_fingerprints", fingerprints)
    monkeypatch.setattr(prepare_plans, "plan_experiments", planning)
    monkeypatch.setattr(prepare_plans, "preprocess", preprocessing)

    destination = prepare_plans.run_native_prepare(11, 3, 2)
    assert destination == dataset_dir / "SNNPlans.json"
    assert calls == [
        ("fingerprints", [11], {"num_processes": 3, "check_dataset_integrity": True}),
        ("planning", [11], {}),
        ("preprocessing", [11], {"plans_identifier": "StockCustom", "configurations": ("3d_fullres",), "num_processes": (2,)}),
    ]
    assert json.loads(destination.read_text())["configurations"]["3d_fullres"]["data_identifier"] == "StockCustom_3d_fullres"


def test_native_prepare_rejects_missing_referenced_preprocessed_folder(tmp_path, monkeypatch, stock_plans):
    dataset_dir = tmp_path / "Dataset011_BraTS"
    dataset_dir.mkdir()
    (dataset_dir / "StockCustom.json").write_text(json.dumps(stock_plans))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path))
    monkeypatch.setattr(prepare_plans, "convert_id_to_dataset_name", lambda dataset_id: "Dataset011_BraTS")
    monkeypatch.setattr(prepare_plans, "extract_fingerprints", lambda *args, **kwargs: None)
    monkeypatch.setattr(prepare_plans, "plan_experiments", lambda *args, **kwargs: "StockCustom")
    monkeypatch.setattr(prepare_plans, "preprocess", lambda *args, **kwargs: None)
    with pytest.raises(FileNotFoundError, match="StockCustom_3d_fullres"):
        prepare_plans.run_native_prepare(11, 3, 2)


def test_update_snn_config_rejects_incompatible_result_before_overwrite(tmp_path, stock_plans):
    plans_path = tmp_path / "SNNPlans.json"
    plans = prepare_plans.derive_snn_plans(stock_plans, CONFIG)
    plans_path.write_text(json.dumps(plans))
    results = tmp_path / "results"
    results.mkdir()
    (results / "plans.json").write_text(json.dumps(plans))
    changed = {**CONFIG, "k": 8}
    with pytest.raises(ValueError, match="incompatible"):
        prepare_plans.update_snn_config(plans_path, changed, results)
    assert json.loads(plans_path.read_text())["snn_config"] == CONFIG


def test_update_snn_config_rejects_result_with_different_spatial_plans(tmp_path, stock_plans):
    plans_path = tmp_path / "SNNPlans.json"
    plans = prepare_plans.derive_snn_plans(stock_plans, CONFIG)
    plans_path.write_text(json.dumps(plans))
    results = tmp_path / "results"
    results.mkdir()
    old_plans = deepcopy(plans)
    old_plans["configurations"]["3d_fullres"]["spacing"] = [2, 2, 2]
    (results / "plans.json").write_text(json.dumps(old_plans))
    with pytest.raises(ValueError, match="incompatible"):
        prepare_plans.update_snn_config(plans_path, CONFIG, results)
    assert json.loads(plans_path.read_text()) == plans


def test_update_snn_config_writes_exact_validated_config_atomically(tmp_path, stock_plans):
    plans_path = tmp_path / "SNNPlans.json"
    plans = prepare_plans.derive_snn_plans(stock_plans, CONFIG)
    plans_path.write_text(json.dumps(plans))
    changed = {**CONFIG, "k": 8}
    prepare_plans.update_snn_config(plans_path, changed, tmp_path / "empty_results")
    assert json.loads(plans_path.read_text())["snn_config"] == changed
    assert list(tmp_path.glob("*.tmp")) == []
