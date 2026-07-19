from collections import deque
from types import SimpleNamespace

import pytest
import torch

from snn_fptt import (
    reduce_dice_totals,
    reduce_loss_totals,
    save_checkpoint_if_main,
)


class FakeAccelerator:
    def __init__(self, reduced_values):
        self.device = torch.device("cpu")
        self._reduced_values = deque(reduced_values)

    def reduce(self, tensor, reduction):
        assert reduction == "sum"
        expected = self._reduced_values.popleft()
        return torch.as_tensor(expected, device=tensor.device, dtype=tensor.dtype)


def test_reduce_loss_totals_uses_global_sum_and_count():
    accelerator = FakeAccelerator([15.0, 5.0])

    result = reduce_loss_totals(
        accelerator,
        torch.tensor(6.0),
        torch.tensor(2.0),
    )

    assert result == pytest.approx(3.0)


def test_reduce_dice_totals_returns_global_per_class_means():
    accelerator = FakeAccelerator([[1.6, 1.2, 0.8], 2])

    result = reduce_dice_totals(
        accelerator,
        torch.tensor([0.8, 0.6, 0.4]),
        torch.tensor(1),
    )

    assert result == {
        "dice_ET": pytest.approx(0.8),
        "dice_TC": pytest.approx(0.6),
        "dice_WT": pytest.approx(0.4),
        "dice_mean": pytest.approx(0.6),
        "n_subjects": 2,
    }


class CheckpointAccelerator:
    def __init__(self, is_main_process):
        self.is_main_process = is_main_process
        self.saved = []
        self.unwrapped = SimpleNamespace()

    def unwrap_model(self, model):
        assert model == "wrapped-model"
        return self.unwrapped

    def save(self, payload, path):
        self.saved.append((payload, path))


def test_save_checkpoint_uses_unwrapped_model_only_on_main_process(tmp_path):
    main = CheckpointAccelerator(is_main_process=True)
    main.unwrapped.state_dict = lambda: {"weight": torch.tensor([1.0])}
    checkpoint_path = tmp_path / "model.pt"

    did_save = save_checkpoint_if_main(
        main, "wrapped-model", checkpoint_path, epoch=3,
        dice_mean=0.75, config={"name": "run"},
    )

    assert did_save is True
    assert main.saved[0][0]["model"]["weight"].item() == 1.0
    assert main.saved[0][0]["epoch"] == 3
    assert main.saved[0][0]["dice_mean"] == 0.75

    worker = CheckpointAccelerator(is_main_process=False)
    did_save = save_checkpoint_if_main(
        worker, "wrapped-model", checkpoint_path, epoch=3,
        dice_mean=0.75, config={"name": "run"},
    )

    assert did_save is False
    assert worker.saved == []
