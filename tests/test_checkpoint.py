"""Native checkpoint compatibility for the SNN trainer."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from snn_nnunet import fptt
from snn_nnunet.network_adapter import SNNnnUNetAdapter
from snn_nnunet.trainer import nnUNetTrainerSNNFPTT
from test_trainer import REGION_DATASET, StatefulTinyCore, make_plans


NATIVE_KEYS = {
    "network_weights", "optimizer_state", "grad_scaler_state", "logging",
    "_best_ema", "current_epoch", "init_args", "trainer_name",
    "inference_allowed_mirroring_axes",
}


@pytest.fixture(autouse=True)
def native_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "results"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "preprocessed"))


def prepared_trainer(tmp_path, *, use_fptt=True, dtype=torch.float32):
    plans = make_plans()
    plans["snn_config"]["use_fptt"] = use_fptt
    trainer = nnUNetTrainerSNNFPTT(
        plans, "3d_fullres", 0, REGION_DATASET, torch.device("cpu")
    )
    trainer.network = SNNnnUNetAdapter(
        trainer.snn_config, core=StatefulTinyCore()
    ).to(dtype=dtype)
    trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=0.01, momentum=0.9)
    trainer.was_initialized = True
    trainer.inference_allowed_mirroring_axes = (1, 2)
    if use_fptt:
        fptt.init_running_params(trainer.network)
    return trainer


def make_saved_checkpoint(tmp_path, *, use_fptt=True):
    source = prepared_trainer(tmp_path, use_fptt=use_fptt)
    parameter = next(source.network.parameters())
    parameter.grad = torch.ones_like(parameter)
    source.optimizer.step()
    source.optimizer.zero_grad(set_to_none=True)
    source.current_epoch = 3
    source._best_ema = 0.72
    source.logger.log("train_losses", 0.4, 0)
    if use_fptt:
        source.network.avg_weights["core.weight"].fill_(2.5)
        source.network.lambdas["core.weight"].fill_(-0.75)
    filename = tmp_path / "checkpoint.pth"
    source.save_checkpoint(str(filename))
    return source, filename


@pytest.mark.parametrize("use_fptt", [True, False])
def test_checkpoint_preserves_native_payload_and_adds_only_enabled_tensors(
    tmp_path, monkeypatch, use_fptt
):
    import snn_nnunet.trainer as trainer_module

    replacements = []
    native_replace = trainer_module.os.replace

    def record_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return native_replace(source, destination)

    monkeypatch.setattr(trainer_module.os, "replace", record_replace)
    source, filename = make_saved_checkpoint(tmp_path, use_fptt=use_fptt)
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)

    assert set(checkpoint) == NATIVE_KEYS | {"fptt_state"}
    assert checkpoint["current_epoch"] == 4
    assert checkpoint["logging"]["train_losses"] == [0.4]
    assert checkpoint["inference_allowed_mirroring_axes"] == (1, 2)
    assert checkpoint["optimizer_state"]["state"]
    assert checkpoint["grad_scaler_state"] is None
    state = checkpoint["fptt_state"]
    assert {key: state[key] for key in (
        "use_fptt", "k", "fptt_alpha", "fptt_beta", "fptt_rho", "fptt_lambda"
    )} == {
        "use_fptt": use_fptt, "k": 16, "fptt_alpha": 0.5,
        "fptt_beta": 0.5, "fptt_rho": 0.0, "fptt_lambda": 2.0,
    }
    assert ("avg_weights" in state) is use_fptt
    assert ("lambdas" in state) is use_fptt
    if use_fptt:
        assert set(state["avg_weights"]) == {"core.weight"}
        assert state["avg_weights"]["core.weight"].device.type == "cpu"
        torch.testing.assert_close(state["avg_weights"]["core.weight"], torch.tensor(2.5))
        torch.testing.assert_close(state["lambdas"]["core.weight"], torch.tensor(-0.75))
    assert len(replacements) == 1
    assert replacements[0][0].parent == filename.parent
    assert replacements[0][1] == filename
    assert not replacements[0][0].exists()
    assert set(source.network.state_dict()) == set(checkpoint["network_weights"])


@pytest.mark.parametrize("load_as_dict", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_resume_restores_native_and_fptt_state(tmp_path, load_as_dict, dtype):
    source, filename = make_saved_checkpoint(tmp_path)
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    target = prepared_trainer(tmp_path, dtype=dtype)
    target.network.avg_weights["core.weight"].zero_()
    target.network.lambdas["core.weight"].zero_()
    target.load_checkpoint(checkpoint if load_as_dict else str(filename))

    assert target.current_epoch == 4
    assert target._best_ema == 0.72
    assert target.inference_allowed_mirroring_axes == (1, 2)
    assert target.logger.get_checkpoint()["train_losses"] == [0.4]
    assert target.optimizer.state_dict()["state"]
    for name, parameter in target.network.named_parameters():
        torch.testing.assert_close(parameter, dict(source.network.named_parameters())[name].to(dtype))
        for key, expected in (("avg_weights", 2.5), ("lambdas", -0.75)):
            value = getattr(target.network, key)[name]
            assert value.device == parameter.device
            assert value.dtype == parameter.dtype
            torch.testing.assert_close(value, torch.tensor(expected, dtype=dtype))


def test_disabled_resume_has_no_auxiliary_state(tmp_path):
    _, filename = make_saved_checkpoint(tmp_path, use_fptt=False)
    target = prepared_trainer(tmp_path, use_fptt=False)
    target.load_checkpoint(str(filename))
    assert target.current_epoch == 4
    assert not hasattr(target.network, "avg_weights")
    assert not hasattr(target.network, "lambdas")


@pytest.mark.parametrize("field,other", [
    ("k", 8), ("use_fptt", False), ("fptt_alpha", 0.2),
    ("fptt_beta", 0.2), ("fptt_rho", 0.3), ("fptt_lambda", 1.0),
])
@pytest.mark.parametrize("load_as_dict", [False, True])
def test_config_mismatch_fails_before_native_state_mutation(tmp_path, field, other, load_as_dict):
    _, filename = make_saved_checkpoint(tmp_path)
    target = prepared_trainer(tmp_path)
    target.snn_config = replace(target.snn_config, **{field: other})
    before = {name: value.clone() for name, value in target.network.state_dict().items()}
    target.current_epoch = 81
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)

    with pytest.raises(ValueError, match=field):
        target.load_checkpoint(checkpoint if load_as_dict else str(filename))

    assert target.current_epoch == 81
    for name, value in target.network.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_native_predictor_fields_remain_directly_readable(tmp_path):
    _, filename = make_saved_checkpoint(tmp_path)
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    # The native predictor uses these fields and ignores unknown checkpoint keys.
    prediction_model = SNNnnUNetAdapter(
        prepared_trainer(tmp_path).snn_config, core=StatefulTinyCore()
    )
    prediction_model.load_state_dict(checkpoint["network_weights"])
    assert checkpoint["trainer_name"] == "nnUNetTrainerSNNFPTT"
    assert checkpoint["inference_allowed_mirroring_axes"] == (1, 2)
    torch.testing.assert_close(
        prediction_model.core.weight, checkpoint["network_weights"]["core.weight"]
    )


def test_disabled_native_checkpointing_does_not_write_extension(tmp_path):
    source = prepared_trainer(tmp_path)
    source.disable_checkpointing = True
    filename = tmp_path / "disabled.pth"
    source.save_checkpoint(str(filename))
    assert not filename.exists()


def test_nonzero_rank_does_not_write_extension(tmp_path):
    source = prepared_trainer(tmp_path)
    source.local_rank = 1
    filename = tmp_path / "nonzero.pth"
    source.save_checkpoint(str(filename))
    assert not filename.exists()
