"""Plans-driven SNN trainer that retains nnU-Net's native pipeline."""

import os
import tempfile

import torch
from torch import autocast
from torch import nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from snn_nnunet import fptt, network_adapter
from snn_nnunet.network_adapter import SNNConfig, SNNnnUNetAdapter, slice_temporal_window


_EPOCH_METRICS = (
    "task_losses",
    "fptt_regularization",
    "total_losses",
    "optimizer_updates",
    "k",
    "fptt_mode",
)
_CHECKPOINT_CONFIG_FIELDS = (
    "use_fptt", "k", "fptt_alpha", "fptt_beta", "fptt_rho", "fptt_lambda",
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

    def initialize(self):
        super().initialize()
        if self.snn_config.use_fptt:
            fptt.init_running_params(self._get_adapter())

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank != 0 or self.disable_checkpointing:
            return

        checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
        state = {name: getattr(self.snn_config, name) for name in _CHECKPOINT_CONFIG_FIELDS}
        if self.snn_config.use_fptt:
            state.update(fptt.export_fptt_tensors(self._get_adapter()))
        checkpoint["fptt_state"] = state

        directory = os.path.dirname(os.path.abspath(filename))
        descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pth", dir=directory)
        os.close(descriptor)
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, filename)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_checkpoint(self, filename_or_checkpoint: dict | str) -> None:
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location="cpu", weights_only=False)
        else:
            checkpoint = filename_or_checkpoint
        state = checkpoint.get("fptt_state")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint is missing fptt_state")
        for name in _CHECKPOINT_CONFIG_FIELDS:
            if name not in state or state[name] != getattr(self.snn_config, name):
                raise ValueError(f"Checkpoint {name} does not match the current SNN plans")
        if self.snn_config.use_fptt and not all(
            name in state for name in ("avg_weights", "lambdas")
        ):
            raise ValueError("Checkpoint is missing FPTT tensors")

        if isinstance(filename_or_checkpoint, str):
            super().load_checkpoint(filename_or_checkpoint)
        else:
            # nnU-Net 2.8.1 advertises dict loads but leaves its local
            # checkpoint variable unset on that path. A file keeps all native
            # model, optimizer, logger, and scaler restoration in super().
            with tempfile.NamedTemporaryFile(suffix=".pth") as temporary:
                torch.save(checkpoint, temporary.name)
                super().load_checkpoint(temporary.name)
        if self.snn_config.use_fptt:
            adapter = self._get_adapter()
            fptt.init_running_params(adapter)
            fptt.restore_fptt_tensors(adapter, state)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        config = self.snn_config
        axis = config.temporal_axis
        length = data.shape[2 + axis]
        adapter = self._get_adapter()
        task_sum = 0.0
        regularization_sum = 0.0
        total_sum = 0.0
        updates = 0
        windows = 0

        for t0 in range(0, length, config.k):
            end = min(t0 + config.k, length)
            window_length = end - t0
            data_window = slice_temporal_window(data, t0, end, axis)
            target_window = slice_temporal_window(target, t0, end, axis)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                logits = self.network(data_window, t0=t0, chunk_size=window_length)
                task_loss = self.loss(logits, target_window)
                regularization = torch.zeros_like(task_loss)
                if config.use_fptt:
                    regularization = fptt.regularizer_loss(
                        adapter, regularization, config.fptt_alpha,
                        config.fptt_rho, config.fptt_lambda,
                    )
                total_loss = task_loss + regularization

            if self.grad_scaler is not None:
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                scale_before = self.grad_scaler.get_scale()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                did_step = self.grad_scaler.get_scale() >= scale_before
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
                did_step = True

            if config.use_fptt and did_step:
                fptt.update_running_params(adapter, config.fptt_alpha, config.fptt_beta)
            adapter.detach_states()
            task_sum += float(task_loss.detach())
            regularization_sum += float(regularization.detach())
            total_sum += float(total_loss.detach())
            updates += int(did_step)
            windows += 1

        return {
            "loss": total_sum / windows,
            "task_loss": task_sum / windows,
            "fptt_regularization": regularization_sum / windows,
            "total_loss": total_sum / windows,
            "window_count": windows,
            "optimizer_updates": updates,
            "k": config.k,
            "fptt_mode": config.use_fptt,
        }

    def on_train_epoch_end(self, train_outputs: list[dict]):
        if self.snn_config.use_fptt:
            fptt.reset_running_params(self._get_adapter())
        super().on_train_epoch_end(train_outputs)
        keys = ("task_loss", "fptt_regularization", "total_loss")
        local_windows = sum(int(row.get("window_count", row["optimizer_updates"])) for row in train_outputs)
        local_updates = sum(int(row["optimizer_updates"]) for row in train_outputs)
        local_sums = {
            key: sum(float(row[key]) * int(row.get("window_count", row["optimizer_updates"])) for row in train_outputs)
            for key in keys
        }
        if self.is_ddp:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, (local_windows, local_updates, local_sums))
        else:
            gathered = [(local_windows, local_updates, local_sums)]
        global_windows = sum(item[0] for item in gathered)
        global_updates = sum(item[1] for item in gathered)
        values = {
            "task_losses": sum(item[2]["task_loss"] for item in gathered) / global_windows,
            "fptt_regularization": sum(item[2]["fptt_regularization"] for item in gathered) / global_windows,
            "total_losses": sum(item[2]["total_loss"] for item in gathered) / global_windows,
            "optimizer_updates": global_updates // len(gathered),
            "k": self.snn_config.k,
            "fptt_mode": self.snn_config.use_fptt,
        }
        for key, value in values.items():
            self.logger.log(key, value, self.current_epoch)
