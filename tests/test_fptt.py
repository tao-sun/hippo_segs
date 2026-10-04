import pytest
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

import snn_nnunet.fptt as fptt_module
from test_trainer import prepare_window_trainer, trainer

from snn_nnunet.fptt import (
    export_fptt_tensors,
    init_running_params,
    regularizer_loss,
    reset_running_params,
    restore_fptt_tensors,
    update_running_params,
)


class TwoParameterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
        self.bias = torch.nn.Parameter(torch.tensor([0.5]))


def test_initial_state_clones_parameters_and_zeros_lambdas():
    model = TwoParameterModel()
    init_running_params(model)

    assert set(model.avg_weights) == {"weight", "bias"}
    torch.testing.assert_close(model.avg_weights["weight"], torch.tensor([2.0, -1.0]))
    torch.testing.assert_close(model.avg_weights["bias"], torch.tensor([0.5]))
    torch.testing.assert_close(model.lambdas["weight"], torch.zeros(2))
    torch.testing.assert_close(model.lambdas["bias"], torch.zeros(1))
    assert model.avg_weights["weight"].data_ptr() != model.weight.data_ptr()
    assert model.avg_weights["bias"].data_ptr() != model.bias.data_ptr()

    with torch.no_grad():
        model.weight.copy_(torch.tensor([3.0, 1.0]))
        model.bias.copy_(torch.tensor([-0.5]))
    torch.testing.assert_close(model.avg_weights["weight"], torch.tensor([2.0, -1.0]))
    torch.testing.assert_close(model.avg_weights["bias"], torch.tensor([0.5]))


def test_regularizer_uses_both_linear_and_quadratic_terms():
    model = TwoParameterModel()
    init_running_params(model)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([3.0, 1.0]))
        model.bias.copy_(torch.tensor([-0.5]))
    model.lambdas["weight"].copy_(torch.tensor([0.25, -0.5]))
    model.lambdas["bias"].copy_(torch.tensor([1.0]))

    result = regularizer_loss(model, torch.tensor(0.7), alpha=0.5, rho=0.25, _lambda=2.0)

    # Initial 0.7 + linear 0.1875 + quadratic 3.0.
    assert result.item() == pytest.approx(3.8875)


def test_update_and_reset_use_preserved_fptt_recurrence():
    model = TwoParameterModel()
    init_running_params(model)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([3.0, 1.0]))
        model.bias.copy_(torch.tensor([-0.5]))

    update_running_params(model, alpha=0.5, beta=0.25)

    torch.testing.assert_close(model.lambdas["weight"], torch.tensor([-0.5, -1.0]))
    torch.testing.assert_close(model.lambdas["bias"], torch.tensor([0.5]))
    torch.testing.assert_close(model.avg_weights["weight"], torch.tensor([2.5, 0.0]))
    torch.testing.assert_close(model.avg_weights["bias"], torch.tensor([0.0]))
    reset_running_params(model)
    torch.testing.assert_close(model.weight, torch.tensor([2.5, 0.0]))
    torch.testing.assert_close(model.bias, torch.tensor([0.0]))


def test_export_copies_cpu_tensors_and_restore_matches_each_parameter():
    source = TwoParameterModel()
    init_running_params(source)
    source.avg_weights["weight"].copy_(torch.tensor([4.0, 5.0]))
    source.lambdas["bias"].copy_(torch.tensor([-3.0]))

    state = export_fptt_tensors(source)

    assert set(state) == {"avg_weights", "lambdas"}
    for name in ("weight", "bias"):
        for field in ("avg_weights", "lambdas"):
            exported = state[field][name]
            assert exported.device.type == "cpu"
            assert exported.data_ptr() != getattr(source, field)[name].data_ptr()
    source.avg_weights["weight"].add_(10)
    torch.testing.assert_close(state["avg_weights"]["weight"], torch.tensor([4.0, 5.0]))

    restored = TwoParameterModel().to(dtype=torch.float64)
    restore_fptt_tensors(restored, state)

    for name, parameter in restored.named_parameters():
        for field in ("avg_weights", "lambdas"):
            tensor = getattr(restored, field)[name]
            assert tensor.device == parameter.device
            assert tensor.dtype == parameter.dtype
            torch.testing.assert_close(tensor, state[field][name].to(dtype=torch.float64))


def test_enabled_fptt_initializes_then_regularizes_each_window_and_resets_before_native_epoch(
    trainer, monkeypatch
):
    prepare_window_trainer(trainer, monkeypatch, use_fptt=True)
    events = []
    native_epoch = nnUNetTrainer.on_train_epoch_end

    def native_initialize(self):
        if self.was_initialized:
            raise RuntimeError("already initialized")
        events.append("native_initialize")
        self.was_initialized = True

    def native_on_train_epoch_end(self, outputs):
        events.append("native_epoch")
        return native_epoch(self, outputs)

    monkeypatch.setattr(nnUNetTrainer, "initialize", native_initialize)
    monkeypatch.setattr(nnUNetTrainer, "on_train_epoch_end", native_on_train_epoch_end)
    for name in ("init_running_params", "regularizer_loss", "update_running_params", "reset_running_params"):
        original = getattr(fptt_module, name)

        def record(*args, _name=name, _original=original, **kwargs):
            events.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(fptt_module, name, record)

    trainer.initialize()
    assert events == ["native_initialize", "init_running_params"]
    with pytest.raises(RuntimeError, match="already initialized"):
        trainer.initialize()
    output = trainer.train_step({"data": torch.ones(1, 4, 128, 1, 1),
                                 "target": torch.zeros(1, 3, 128, 1, 1)})
    assert events.count("regularizer_loss") == 8
    assert events.count("update_running_params") == 8
    assert events.count("reset_running_params") == 0
    assert output["optimizer_updates"] == 8
    assert output["fptt_mode"] is True
    assert output["total_loss"] == pytest.approx(
        output["task_loss"] + output["fptt_regularization"]
    )

    trainer.on_train_epoch_end([output])
    assert events[-2:] == ["reset_running_params", "native_epoch"]
    assert events.count("reset_running_params") == 1
    assert trainer.logger.get_checkpoint()["optimizer_updates"] == [8]
