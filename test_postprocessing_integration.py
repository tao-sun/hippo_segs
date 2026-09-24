from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from accelerate import Accelerator

import evaluate_snn_fold_view as standalone_evaluation
import snn_fptt as training


class _PerfectPredictionModel(torch.nn.Module):
    def forward(self, x_window, t0=0):
        prediction = x_window[:, :, :3].permute(0, 2, 1, 3, 4)
        return (prediction * 2 - 1) * 20


class _HoledCubeDataset(torch.utils.data.Dataset):
    view = "axial"

    def __init__(self):
        volume = np.zeros((3, 5, 5, 5), dtype=np.float32)
        volume[:, 1:4, 1:4, 1:4] = 1
        volume[:, 2, 2, 2] = 0
        self.volume = volume
        self.targets = torch.from_numpy(volume.transpose(3, 0, 1, 2).copy())
        self.inputs = torch.zeros((5, 4, 5, 5), dtype=torch.float32)
        self.inputs[:, :3] = self.targets

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.inputs, self.targets, {
            "sid": "BraTS-holed-cube",
            "xyz": (5, 5, 5),
            "raw_labels": (0, 1, 2, 3),
        }


def test_training_yaml_exposes_validated_postprocessing_parameters(tmp_path):
    config = yaml.safe_load(Path("experiments_snn_fptt.yaml").read_text())
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config))

    loaded = training.load_experiment_from_yaml(str(path))

    assert loaded["apply_postprocessing"] is False
    assert loaded["min_component_sizes"] == (0, 20, 50)
    assert loaded["closing_radius"] == 1


def test_training_config_defaults_preserve_raw_evaluation(tmp_path):
    config = yaml.safe_load(Path("experiments_snn_fptt.yaml").read_text())
    for key in ("apply_postprocessing", "min_component_sizes", "closing_radius"):
        config.pop(key, None)
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config))

    loaded = training.load_experiment_from_yaml(str(path))

    assert loaded["apply_postprocessing"] is False
    assert loaded["min_component_sizes"] == (0, 0, 0)
    assert loaded["closing_radius"] == 1


def test_standalone_yaml_exposes_validated_postprocessing_parameters(tmp_path):
    config = yaml.safe_load(Path("evaluation_snn.yaml").read_text())
    config["checkpoint"] = "checkpoint.pt"
    config["data_root"] = "dataset"
    config["cache_required"] = False
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(config))

    loaded = standalone_evaluation.load_eval_config(path)

    assert loaded.apply_postprocessing is False
    assert loaded.min_component_sizes == (0, 20, 50)
    assert loaded.closing_radius == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("apply_postprocessing", "false"),
        ("min_component_sizes", [0, 20]),
        ("min_component_sizes", [0, True, 50]),
        ("closing_radius", 1.5),
    ],
)
def test_training_config_rejects_invalid_postprocessing_parameters(
    tmp_path, key, value
):
    config = yaml.safe_load(Path("experiments_snn_fptt.yaml").read_text())
    config[key] = value
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match=key):
        training.load_experiment_from_yaml(str(path))


def test_standalone_evaluation_postprocesses_prediction_but_not_ground_truth():
    dataset = _HoledCubeDataset()
    original_ground_truth = dataset.targets.clone()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    raw = standalone_evaluation.evaluate_loader(
        _PerfectPredictionModel(),
        loader,
        device=torch.device("cpu"),
        view="axial",
        window_size=2,
        threshold=0.5,
        apply_postprocessing=False,
    )
    processed = standalone_evaluation.evaluate_loader(
        _PerfectPredictionModel(),
        loader,
        device=torch.device("cpu"),
        view="axial",
        window_size=2,
        threshold=0.5,
        apply_postprocessing=True,
        min_component_sizes=(0, 0, 0),
        closing_radius=1,
    )

    assert raw["dice_mean"] == 1.0
    assert processed["dice_mean"] == pytest.approx(52 / 53)
    assert processed["subjects"][0]["gt_voxels"] == {"ET": 26, "TC": 26, "WT": 26}
    assert torch.equal(dataset.targets, original_ground_truth)


def test_evaluate_3d_snn_postprocesses_after_stacking_without_changing_ground_truth():
    dataset = _HoledCubeDataset()
    original_ground_truth = dataset.targets.clone()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    accelerator = Accelerator(cpu=True)

    raw = training.evaluate_3d_snn(
        _PerfectPredictionModel(),
        loader,
        accelerator,
        prob_threshold=0.5,
        k=2,
        apply_postprocessing=False,
    )
    processed = training.evaluate_3d_snn(
        _PerfectPredictionModel(),
        loader,
        accelerator,
        prob_threshold=0.5,
        k=2,
        apply_postprocessing=True,
        min_component_sizes=(0, 0, 0),
        closing_radius=1,
    )

    assert raw["dice_mean"] == 1.0
    assert processed["dice_mean"] == pytest.approx(52 / 53)
    assert torch.equal(dataset.targets, original_ground_truth)
