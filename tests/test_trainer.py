"""Native trainer construction contracts for the SNN adapter."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from nnunetv2.training.logging.nnunet_logger import MetaLogger
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from snn_nnunet import fptt, network_adapter
import snn_nnunet.trainer as trainer_module
from snn_nnunet.prepare_plans import DEFAULT_SNN_CONFIG
from snn_nnunet.trainer import nnUNetTrainerSNNFPTT


REGION_DATASET = {
    "channel_names": {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"},
    "labels": {
        "background": 0,
        "whole_tumor": [1, 2, 3],
        "tumor_core": [1, 3],
        "enhancing_tumor": 3,
    },
    "regions_class_order": [2, 1, 3],
}


def make_plans(axis=0):
    return {
        "continue_training": False,
        "dataset_name": "Dataset011_BraTS",
        "plans_name": "SNNPlans",
        "snn_config": {**deepcopy(DEFAULT_SNN_CONFIG), "temporal_axis": axis},
        "configurations": {
            "3d_fullres": {
                "data_identifier": "nnUNetPlans_3d_fullres",
                "batch_size": 4,
                "patch_size": [128, 128, 128],
                "batch_dice": False,
                "architecture": {"network_class_name": "unused"},
            }
        },
    }


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "results"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "preprocessed"))
    return nnUNetTrainerSNNFPTT(make_plans(), "3d_fullres", 0, REGION_DATASET, torch.device("cpu"))


def test_trainer_sets_approved_native_schedule_and_config(trainer):
    assert (trainer.num_epochs, trainer.num_iterations_per_epoch, trainer.num_val_iterations_per_epoch) == (300, 250, 50)
    assert (trainer.initial_lr, trainer.weight_decay, trainer.oversample_foreground_percent) == (1e-2, 3e-5, 0.33)
    assert trainer.enable_deep_supervision is False
    assert trainer.snn_config.k == 16


def test_architecture_hook_reads_plans_and_uses_runtime_core_factory(monkeypatch):
    class TinyCore(torch.nn.Module):
        def forward(self, x, t0):
            return x[:, :, :3].movedim(1, 2)

        def detach_states(self):
            pass

    core = TinyCore()
    monkeypatch.setattr(network_adapter, "build_core", lambda config: core)
    plans_manager = PlansManager(make_plans())
    configuration_manager = plans_manager.get_configuration("3d_fullres")

    adapter = nnUNetTrainerSNNFPTT.build_network_architecture(
        plans_manager, configuration_manager, 4, 3, enable_deep_supervision=False
    )

    assert type(adapter) is network_adapter.SNNnnUNetAdapter
    assert adapter.core is core
    assert adapter.config.to_dict() == DEFAULT_SNN_CONFIG
    assert adapter(torch.ones(1, 4, 2, 3, 4)).shape == (1, 3, 2, 3, 4)


@pytest.mark.parametrize("channels", [(3, 3), (4, 4)])
def test_architecture_hook_rejects_native_channel_mismatch(channels):
    plans_manager = PlansManager(make_plans())
    with pytest.raises(ValueError, match="channels"):
        nnUNetTrainerSNNFPTT.build_network_architecture(
            plans_manager, plans_manager.get_configuration("3d_fullres"), *channels
        )


def test_deep_supervision_switch_does_not_require_decoder(trainer):
    trainer.network = network_adapter.SNNnnUNetAdapter(
        trainer.snn_config, core=torch.nn.Identity()
    )
    trainer.set_deep_supervision_enabled(False)
    assert not hasattr(trainer.network, "decoder")


def test_external_discovery_finds_native_subclass(monkeypatch):
    monkeypatch.setenv("nnUNet_extTrainer", str(Path(network_adapter.__file__).parent))
    discovered = recursive_find_trainer_class_by_name("nnUNetTrainerSNNFPTT")
    assert discovered.__name__ == "nnUNetTrainerSNNFPTT"
    assert issubclass(discovered, nnUNetTrainer)
    assert Path(discovered.__module__.replace(".", "/")).name == "trainer"


def test_compile_defaults_off_but_respects_explicit_override(trainer, monkeypatch):
    monkeypatch.delenv("nnUNet_compile", raising=False)
    assert trainer._do_i_compile() is False
    trainer.device = torch.device("cuda")
    monkeypatch.setenv("nnUNet_compile", "true")
    assert trainer._do_i_compile() is True


def test_adapter_unwraps_ddp_and_optimized_module(trainer, monkeypatch):
    class Wrapper:
        def __init__(self, module):
            self.module = module

    adapter = network_adapter.SNNnnUNetAdapter(trainer.snn_config, core=torch.nn.Identity())
    monkeypatch.setattr(trainer_module, "DDP", Wrapper)
    trainer.network = Wrapper(torch.compile(adapter, backend="eager"))
    assert trainer._get_adapter() is adapter


def test_adapter_unwrap_rejects_unexpected_network(trainer):
    trainer.network = torch.nn.Identity()
    with pytest.raises(TypeError, match="SNNnnUNetAdapter"):
        trainer._get_adapter()


def test_custom_metrics_extend_native_local_logger_without_shortening_epoch(trainer):
    assert type(trainer.logger) is MetaLogger
    local = trainer.logger.local_logger.my_fantastic_logging
    for key in ("task_losses", "fptt_regularization", "total_losses", "optimizer_updates", "k", "fptt_mode"):
        assert local[key] == []
    for key, value in {
        "train_losses": 1.0, "val_losses": 1.0, "lrs": 0.01,
        "epoch_start_timestamps": 1.0, "epoch_end_timestamps": 2.0,
        "mean_fg_dice": 0.5, "ema_fg_dice": 0.5,
        "dice_per_class_or_region": [0.5, 0.5, 0.5],
        "task_losses": 0.9, "fptt_regularization": 0.1, "total_losses": 1.0,
        "optimizer_updates": 8, "k": 16, "fptt_mode": True,
    }.items():
        trainer.logger.log(key, value, 0)
    assert {len(values) for values in trainer.logger.get_checkpoint().values()} == {1}
    trainer.logger.plot_progress_png(trainer.output_folder)
    assert (Path(trainer.output_folder) / "progress.png").is_file()


def test_completed_native_epoch_fills_all_custom_logger_lists(trainer):
    trainer.network = network_adapter.SNNnnUNetAdapter(trainer.snn_config, core=StatefulTinyCore())
    fptt.init_running_params(trainer.network)
    trainer.on_train_epoch_end([
        {"loss": 1.0, "task_loss": 0.8, "fptt_regularization": 0.2,
         "total_loss": 1.0, "optimizer_updates": 2},
        {"loss": 3.0, "task_loss": 2.6, "fptt_regularization": 0.4,
         "total_loss": 3.0, "optimizer_updates": 1},
    ])
    logged = trainer.logger.get_checkpoint()
    assert logged["train_losses"] == [2.0]
    assert logged["task_losses"] == [pytest.approx(1.4)]
    assert logged["fptt_regularization"] == [pytest.approx(0.2666666667)]
    assert logged["total_losses"] == [pytest.approx(1.6666666667)]
    assert logged["optimizer_updates"] == [3]
    assert logged["k"] == [16]
    assert logged["fptt_mode"] == [True]


class StatefulTinyCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.state = None
        self.starts = []
        self.had_previous_state = []

    def forward(self, x, t0):
        self.starts.append(t0)
        self.had_previous_state.append(self.state is not None and t0 != 0)
        if t0 == 0:
            self.state = None
        logits = x[:, :, :3].movedim(1, 2) * self.weight
        if self.state is not None:
            logits = logits + self.state
        self.state = logits.mean()
        return logits

    def detach_states(self):
        if self.state is not None:
            self.state = self.state.detach()


class RecordingNetwork(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.calls = []

    def forward(self, x, *, t0, chunk_size):
        self.calls.append((t0, chunk_size, x.shape))
        return self.module(x, t0=t0, chunk_size=chunk_size)


def prepare_window_trainer(trainer, monkeypatch, *, k=16, axis=0, use_fptt=False):
    trainer.snn_config = replace(trainer.snn_config, k=k, temporal_axis=axis, use_fptt=use_fptt)
    core = StatefulTinyCore()
    adapter = network_adapter.SNNnnUNetAdapter(trainer.snn_config, core=core)
    wrapper = RecordingNetwork(adapter)
    monkeypatch.setattr(trainer_module, "DDP", RecordingNetwork)
    trainer.network = wrapper
    trainer.optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.001)
    trainer.grad_scaler = None
    seen_targets = []

    def loss(logits, target):
        seen_targets.append(target.detach().clone())
        return torch.nn.functional.mse_loss(logits, target)

    trainer.loss = loss
    return core, wrapper, seen_targets


@pytest.mark.parametrize("k,updates", [(16, 8), (8, 16), (1, 128)])
def test_train_step_updates_once_per_complete_window(trainer, monkeypatch, k, updates):
    core, wrapper, targets = prepare_window_trainer(trainer, monkeypatch, k=k)
    steps = []
    original_step = trainer.optimizer.step

    def step(*args, **kwargs):
        steps.append(len(wrapper.calls))
        return original_step(*args, **kwargs)

    monkeypatch.setattr(trainer.optimizer, "step", step)
    output = trainer.train_step({
        "data": torch.ones(1, 4, 128, 1, 1),
        "target": torch.zeros(1, 3, 128, 1, 1),
    })

    assert len(steps) == updates
    assert steps == list(range(1, updates + 1))
    assert [call[:2] for call in wrapper.calls] == [(i * k, k) for i in range(updates)]
    assert core.starts == [i * k for i in range(updates)]
    assert core.had_previous_state == [False] + [True] * (updates - 1)
    assert core.state.grad_fn is None
    assert len(targets) == updates
    assert output["optimizer_updates"] == updates


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_train_step_slices_native_target_axis_and_partial_final_window(trainer, monkeypatch, axis):
    core, wrapper, targets = prepare_window_trainer(trainer, monkeypatch, axis=axis)
    shape = [1, 1, 1, 1, 1]
    shape[2 + axis] = 17
    target = torch.arange(17, dtype=torch.float32).reshape(shape).expand(1, 3, *shape[2:]) / 100
    data = torch.ones(1, 4, *shape[2:])

    output = trainer.train_step({"data": data, "target": target})

    assert [(t0, size) for t0, size, _ in wrapper.calls] == [(0, 16), (16, 1)]
    assert [tuple(t.shape) for t in targets] == [tuple(target.narrow(2 + axis, 0, 16).shape),
                                                   tuple(target.narrow(2 + axis, 16, 1).shape)]
    torch.testing.assert_close(targets[0], target.narrow(2 + axis, 0, 16))
    torch.testing.assert_close(targets[1], target.narrow(2 + axis, 16, 1))
    assert core.starts == [0, 16]
    assert output["optimizer_updates"] == 2


def test_disabled_fptt_never_calls_auxiliary_functions(trainer, monkeypatch):
    prepare_window_trainer(trainer, monkeypatch, use_fptt=False)
    monkeypatch.setattr(trainer_module, "fptt", fptt, raising=False)
    for name in ("init_running_params", "regularizer_loss", "update_running_params", "reset_running_params"):
        monkeypatch.setattr(trainer_module.fptt, name, lambda *args, **kwargs: pytest.fail("FPTT was called"))
    monkeypatch.setattr(nnUNetTrainer, "initialize", lambda self: setattr(self, "was_initialized", True))

    trainer.initialize()
    trainer.train_step({"data": torch.ones(1, 4, 17, 1, 1),
                        "target": torch.zeros(1, 3, 17, 1, 1)})
    trainer.on_train_epoch_end([{"loss": 1.0, "task_loss": 1.0,
                                 "fptt_regularization": 0.0, "total_loss": 1.0,
                                 "optimizer_updates": 2}])


def test_epoch_metrics_reduce_window_sums_without_multiplying_ddp_update_count(trainer, monkeypatch):
    trainer.snn_config = replace(trainer.snn_config, use_fptt=False)
    trainer.is_ddp = True
    monkeypatch.setattr(nnUNetTrainer, "on_train_epoch_end", lambda self, outputs: None)
    monkeypatch.setattr(trainer_module.dist, "get_world_size", lambda: 2)

    def gather(output, local):
        output[:] = [local, (3, {"task_loss": 9.0,
                                 "fptt_regularization": 3.0, "total_loss": 12.0})]

    monkeypatch.setattr(trainer_module.dist, "all_gather_object", gather)
    trainer.on_train_epoch_end([
        {"loss": 1.0, "task_loss": 0.8, "fptt_regularization": 0.2,
         "total_loss": 1.0, "optimizer_updates": 2},
        {"loss": 3.0, "task_loss": 2.6, "fptt_regularization": 0.4,
         "total_loss": 3.0, "optimizer_updates": 1},
    ])

    logged = trainer.logger.get_checkpoint()
    assert logged["task_losses"] == [pytest.approx(2.2)]
    assert logged["fptt_regularization"] == [pytest.approx(3.8 / 6)]
    assert logged["total_losses"] == [pytest.approx(17 / 6)]
    assert logged["optimizer_updates"] == [3]


def test_train_step_uses_native_scaler_and_clipping_order_for_each_window(trainer, monkeypatch):
    prepare_window_trainer(trainer, monkeypatch)
    events = []
    original_clip = torch.nn.utils.clip_grad_norm_

    class RecordingScaler:
        def scale(self, loss):
            events.append("scale")
            return loss

        def unscale_(self, optimizer):
            events.append("unscale")

        def step(self, optimizer):
            events.append("step")
            optimizer.step()

        def update(self):
            events.append("update")

    def clip(parameters, max_norm):
        assert max_norm == 12
        events.append("clip")
        return original_clip(parameters, max_norm)

    trainer.grad_scaler = RecordingScaler()
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    output = trainer.train_step({"data": torch.ones(1, 4, 17, 1, 1),
                                 "target": torch.zeros(1, 3, 17, 1, 1)})

    assert events == ["scale", "unscale", "clip", "step", "update"] * 2
    assert output["optimizer_updates"] == 2
