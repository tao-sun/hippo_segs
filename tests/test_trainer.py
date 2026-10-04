"""Native trainer construction contracts for the SNN adapter."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from nnunetv2.training.logging.nnunet_logger import MetaLogger
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from snn_nnunet import network_adapter
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
    trainer.on_train_epoch_end([{"loss": 1.0}, {"loss": 3.0}])
    logged = trainer.logger.get_checkpoint()
    assert logged["train_losses"] == [2.0]
    assert logged["task_losses"] == [2.0]
    assert logged["fptt_regularization"] == [0.0]
    assert logged["total_losses"] == [2.0]
    assert logged["optimizer_updates"] == [2]
    assert logged["k"] == [16]
    assert logged["fptt_mode"] == [True]
