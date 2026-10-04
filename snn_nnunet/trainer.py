"""Plans-driven SNN trainer that retains nnU-Net's native pipeline."""

import os

import torch
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from snn_nnunet import network_adapter
from snn_nnunet.network_adapter import SNNConfig, SNNnnUNetAdapter


_EPOCH_METRICS = (
    "task_losses",
    "fptt_regularization",
    "total_losses",
    "optimizer_updates",
    "k",
    "fptt_mode",
)


class nnUNetTrainerSNNFPTT(nnUNetTrainer):
    """Construct the sequential adapter while reusing nnU-Net training services."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.snn_config = SNNConfig.from_plans(self.plans_manager.plans)
        self.num_epochs = 300
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.enable_deep_supervision = False
        for key in _EPOCH_METRICS:
            self.logger.local_logger.my_fantastic_logging.setdefault(key, [])

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        config = SNNConfig.from_plans(plans_manager.plans)
        if (num_input_channels, num_output_channels) != (
            config.num_input_channels,
            config.num_output_channels,
        ):
            raise ValueError("Native channels do not match the SNN plans")
        # SNNnnUNetAdapter resolves network_adapter.build_core when constructed.
        # Keeping that lookup live also lets the native predictor rebuild this model.
        return network_adapter.SNNnnUNetAdapter(config)

    def set_deep_supervision_enabled(self, enabled: bool):
        """The adapter has no nnU-Net decoder or deep-supervision heads."""

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy_2d, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        allowed_axes = tuple(axis for axis in mirror_axes if axis != self.snn_config.temporal_axis)
        self.inference_allowed_mirroring_axes = allowed_axes
        return rotation, dummy_2d, initial_patch_size, allowed_axes

    def _do_i_compile(self):
        if "nnUNet_compile" not in os.environ:
            return False
        return super()._do_i_compile()

    def _get_adapter(self) -> SNNnnUNetAdapter:
        network = self.network
        while isinstance(network, (DDP, OptimizedModule)):
            network = network.module if isinstance(network, DDP) else network._orig_mod
        if not isinstance(network, SNNnnUNetAdapter):
            raise TypeError("Trainer network is not an SNNnnUNetAdapter")
        return network

    def on_train_epoch_end(self, train_outputs: list[dict]):
        super().on_train_epoch_end(train_outputs)
        native_loss = float(self.logger.get_value("train_losses", self.current_epoch))
        values = {
            "task_losses": native_loss,
            "fptt_regularization": 0.0,
            "total_losses": native_loss,
            "optimizer_updates": len(train_outputs),
            "k": self.snn_config.k,
            "fptt_mode": self.snn_config.use_fptt,
        }
        for key, value in values.items():
            self.logger.log(key, value, self.current_epoch)
