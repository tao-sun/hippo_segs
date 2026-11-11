"""Plans-driven SNN trainer that retains nnU-Net's native pipeline."""

from collections.abc import Mapping
from contextlib import nullcontext
import os
import tempfile

import torch
from torch import nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from snn_nnunet import fptt, network_adapter
from snn_nnunet.recurrent_neuron import named_selective_neurons
from snn_nnunet.inference import native_fp32_predictor
from snn_nnunet.empty_dice_evaluation import native_empty_dice_evaluation
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
        # The base trainer enables CUDA AMP by default. Keep SNN training in FP32.
        self.grad_scaler = None
        self.snn_config = SNNConfig.from_plans(self.plans_manager.plans)
        if self.snn_config.bidirectional:
            # Save the effective whole-Z BPTT policy, including in native init_args.
            self.plans_manager.plans['snn_config'] = self.snn_config.to_dict()
            self.my_init_kwargs['plans']['snn_config'] = self.snn_config.to_dict()
        self.num_epochs = self.snn_config.num_epochs
        self.num_iterations_per_epoch = self.snn_config.train_batches_per_epoch
        self.num_val_iterations_per_epoch = self.snn_config.val_batches_per_epoch
        self.initial_lr = self.snn_config.initial_lr
        self.validation_every = self.snn_config.validation_every
        self._periodic_validation = None
        self._overfit_subject_ids = None
        self._overfit_datasets = None
        self._overfit_last_evaluation = None
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.enable_deep_supervision = False
        # Keep a complete recovery point after every completed epoch.
        self.save_every = 1
        for key in _EPOCH_METRICS:
            self.logger.local_logger.my_fantastic_logging.setdefault(key, [])

    def configure_optimizers(self):
        # Select by parameter identity: wrappers can change names, and all other
        # selective-neuron parameters must retain the native weight decay.
        exempt = {
            id(cell.temporal_raw)
            for _, cell in named_selective_neurons(self.network)
            if not cell.config['temporal_weight_decay']
        }
        if not exempt:
            # Preserve the single-group optimizer layout of legacy checkpoints.
            return super().configure_optimizers()
        parameters = list(self.network.parameters())
        optimizer = torch.optim.SGD(
            [
                {'params': [p for p in parameters if id(p) not in exempt],
                 'weight_decay': self.weight_decay},
                {'params': [p for p in parameters if id(p) in exempt],
                 'weight_decay': 0.},
            ],
            lr=self.initial_lr, momentum=0.99, nesterov=True,
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

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

    def _ensure_overfit_subjects(self):
        """Select and restrict volume files before native datasets, patches or caches."""
        if not self.snn_config.overfit_num_volumes:
            return None
        if getattr(self, '_overfit_subject_ids', None) is not None:
            return self._overfit_subject_ids
        from pathlib import Path
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from snn_nnunet.overfit import select_overfit_subjects, stage_selected_data

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        distributed = getattr(self, 'is_ddp', False)
        status = [None]
        if not distributed or dist.get_rank() == 0:
            try:
                ids = select_overfit_subjects(
                    self, self.snn_config.overfit_num_volumes, self.snn_config.seed)
                folder = stage_selected_data(
                    self.preprocessed_dataset_folder, Path(self.output_folder) / 'overfit_data', ids)
                status[0] = {'ids': ids, 'folder': str(folder)}
            except Exception as error:
                if not distributed:
                    raise
                status[0] = {'error': f'{type(error).__name__}: {error}'}
        if distributed:
            dist.broadcast_object_list(status, src=0)
        if 'error' in status[0]:
            raise RuntimeError(f"Overfit selection failed: {status[0]['error']}")
        self._overfit_subject_ids = tuple(status[0]['ids'])
        # Native unpack_dataset scans this folder; only selected files are visible.
        self.preprocessed_dataset_folder = status[0]['folder']
        if not distributed or dist.get_rank() == 0:
            self.print_to_log_file(
                f'OVERFIT DIAGNOSTIC ONLY: train=validation=test; '
                f'{len(self._overfit_subject_ids)} unique labeled training volumes; '
                f'seed={self.snn_config.seed}; subjects={list(self._overfit_subject_ids)}')
        return self._overfit_subject_ids

    def do_split(self):
        if not self.snn_config.overfit_num_volumes:
            return super().do_split()
        ids = self._ensure_overfit_subjects()
        return list(ids), list(ids)

    def get_tr_and_val_datasets(self):
        if not self.snn_config.overfit_num_volumes:
            return super().get_tr_and_val_datasets()
        from snn_nnunet.overfit import make_overfit_datasets
        ids = self._ensure_overfit_subjects()
        if getattr(self, '_overfit_datasets', None) is None:
            self._overfit_datasets = make_overfit_datasets(self, ids)
        return self._overfit_datasets['train'], self._overfit_datasets['validation']

    def _evaluate_overfit(self):
        """All ranks rendezvous; rank zero evaluates identical deterministic splits."""
        ids = self._ensure_overfit_subjects()
        if self._overfit_last_evaluation is not None:
            epoch, summary = self._overfit_last_evaluation
            if epoch == self.current_epoch:
                return summary
        from snn_nnunet.overfit import evaluate_overfit_volumes
        distributed = getattr(self, 'is_ddp', False)
        status = [None]
        if distributed:
            dist.barrier()
        if not distributed or dist.get_rank() == 0:
            network = self.network
            try:
                # A rank-zero DDP forward would otherwise wait for buffer broadcasts.
                self.network = self._unwrapped_network()
                summary = evaluate_overfit_volumes(self, ids, epoch=self.current_epoch)
                status[0] = {'summary': summary}
            except Exception as error:
                if not distributed:
                    raise
                status[0] = {'error': f'{type(error).__name__}: {error}'}
            finally:
                self.network = network
        if distributed:
            dist.broadcast_object_list(status, src=0)
        if 'error' in status[0]:
            raise RuntimeError(f"Overfit evaluation failed: {status[0]['error']}")
        summary = status[0]['summary']
        self._overfit_last_evaluation = (self.current_epoch, summary)
        return summary

    def on_epoch_end(self):
        # Native epoch handling writes latest before best. Suppress only its
        # periodic write, then publish latest after best and epoch bookkeeping.
        save_every = self.save_every
        self.save_every = self.num_epochs + 1
        try:
            super().on_epoch_end()
        finally:
            self.save_every = save_every
        from snn_nnunet.resume_validation import save_completed_checkpoint

        save_completed_checkpoint(self, os.path.join(self.output_folder, 'checkpoint_latest.pth'))
        if self.snn_config.overfit_num_volumes:
            self._evaluate_overfit()

    def on_train_end(self):
        super().on_train_end()
        if self.snn_config.overfit_num_volumes:
            self._evaluate_overfit()

    def on_train_start(self):
        overfit = getattr(getattr(self, 'snn_config', None), 'overfit_num_volumes', 0)
        if overfit:
            self._ensure_overfit_subjects()
        super().on_train_start()
        if not overfit and self.validation_every and self._periodic_validation is None:
            from snn_nnunet.resume_validation import install_periodic_validation

            controller = install_periodic_validation(self, self.validation_every)
            # Retry an interrupted validation when resuming exactly on a boundary.
            controller.validate_current()

    def set_deep_supervision_enabled(self, enabled: bool):
        """Native target pyramids stay disabled; paper heads are requested explicitly."""

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

    def _unwrapped_network(self) -> nn.Module:
        network = self.network
        while isinstance(network, (DDP, OptimizedModule)):
            network = network.module if isinstance(network, DDP) else network._orig_mod
        return network

    def _get_adapter(self) -> SNNnnUNetAdapter:
        network = self._unwrapped_network()
        if not isinstance(network, SNNnnUNetAdapter):
            raise TypeError("Trainer network is not an SNNnnUNetAdapter")
        return network

    def initialize(self):
        if not self.was_initialized:
            from snn_nnunet.reproducibility import configure_reproducibility
            self.reproducibility = configure_reproducibility(
                self.snn_config.seed, self.snn_config.deterministic, self.local_rank)
            self.reproducibility.update(
                augmentation_workers=self.snn_config.augmentation_workers,
                device=str(self.device),
                gpu_name=torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None)
            self.print_to_log_file('Reproducibility:', self.reproducibility)
            self.print_to_log_file(
                f'Batches per epoch: training={self.num_iterations_per_epoch}, '
                f'validation={self.num_val_iterations_per_epoch} (per process)')
            if self.snn_config.bidirectional:
                self.print_to_log_file(
                    f'Bidirectional Paper U-Net: {self.snn_config.bidirectional_fusion} '
                    f'{"memory fusion before pooling" if self.snn_config.bidirectional_fusion == "Shared_SOMA" else "fusion after pooling"}; full-Z BPTT '
                    f'k={self.snn_config.k}, temporal_axis={self.snn_config.temporal_axis}, use_fptt=False')
        super().initialize()
        if self.local_rank == 0:
            for name, cell in named_selective_neurons(self._unwrapped_network()):
                if getattr(cell, 'is_linear_attention', False):
                    if cell.gamma is None:
                        self.print_to_log_file(f'Attention residual {name}: attention_gamma={cell.attention_gamma:g} (fixed)')
                    else:
                        self.print_to_log_file(
                            f'Attention residual {name}: attention_gamma_mode={cell.attention_gamma_mode}; '
                            f'gamma shape={tuple(cell.gamma.shape)} init={cell.config["attention_gamma_init"]:g} (learnable)')
        if self.snn_config.use_fptt:
            fptt.init_running_params(self._get_adapter())
        self._check_temporal_regularization()

    def get_dataloaders(self):
        if self.snn_config.overfit_num_volumes:
            self._ensure_overfit_subjects()
        if not self.snn_config.deterministic:
            return super().get_dataloaders()
        from snn_nnunet.reproducibility import deterministic_dataloaders
        return deterministic_dataloaders(self)

    def _check_temporal_regularization(self):
        """Check the actual optimizer policy, including after checkpoint restore."""
        groups = {id(p): group for group in self.optimizer.param_groups for p in group['params']}
        adapter = self._get_adapter()
        cells = list(named_selective_neurons(adapter))
        temporal_ids = {id(p) for name, p in adapter.named_parameters() if name.endswith('.temporal_raw')}
        if temporal_ids != {id(cell.temporal_raw) for _, cell in cells}:
            raise ValueError('Unrecognized temporal parameters; cannot verify their regularization policy')
        fptt_exempt = getattr(adapter, '_fptt_exempt_parameter_names', frozenset())
        for name, cell in cells:
            if getattr(cell, 'is_linear_attention', False) and cell.gamma is not None:
                if not cell.gamma.requires_grad or id(cell.gamma) not in groups:
                    raise ValueError(f'{name}.gamma must be trainable and included in the optimizer')
            decay = groups[id(cell.temporal_raw)]['weight_decay']
            expected = self.weight_decay if cell.config['temporal_weight_decay'] else 0.
            if decay != expected:
                raise ValueError(f'{name}.temporal_raw optimizer decay is {decay}, expected {expected}')
            regularized = self.snn_config.use_fptt and cell.config['temporal_fptt_regularization']
            if self.snn_config.use_fptt:
                actual = f'{name}.temporal_raw' not in fptt_exempt
                if actual != regularized:
                    raise ValueError(f'{name}.temporal_raw FPTT policy does not match configuration')
            if self.local_rank == 0:
                self.print_to_log_file(
                    f'Temporal regularization {name}: parameters={cell.temporal_raw.numel()} '
                    f'weight_decay={decay:g} fptt_regularization={regularized} '
                    f'trainable={cell.temporal_raw.requires_grad}'
                )

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank != 0 or self.disable_checkpointing:
            super().save_checkpoint(filename)
            return

        # Build the entire native + SNN payload beside the destination. A failed
        # write must leave the preceding complete checkpoint available for resume.
        directory = os.path.dirname(os.path.abspath(filename))
        descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pth", dir=directory)
        os.close(descriptor)
        try:
            super().save_checkpoint(temporary)
            checkpoint = torch.load(temporary, map_location="cpu", weights_only=False)
            state = {name: getattr(self.snn_config, name) for name in _CHECKPOINT_CONFIG_FIELDS}
            if self.snn_config.use_fptt:
                state.update(fptt.export_fptt_tensors(self._get_adapter()))
            checkpoint["fptt_state"] = state
            checkpoint['reproducibility'] = self.reproducibility
            if self.snn_config.model_name == "paper_unet":
                checkpoint["snn_config"] = self.snn_config.to_dict()
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
        saved_plans = checkpoint.get("init_args", {}).get("plans", {})
        saved_config = saved_plans.get("snn_config", {})
        restored_policy = SNNConfig.from_plans(saved_plans)
        for field in ('seed', 'deterministic', 'augmentation_workers', 'overfit_num_volumes'):
            if getattr(restored_policy, field) != getattr(self.snn_config, field):
                raise ValueError(f'Checkpoint {field} differs from the current reproducibility policy')
        if self.snn_config.model_name == "paper_unet" or saved_config.get("model_name") == "paper_unet":
            restored = SNNConfig.from_plans(saved_plans)
            if (restored.model_name != self.snn_config.model_name or
                    restored.bidirectional != self.snn_config.bidirectional or
                    restored.bidirectional_fusion != self.snn_config.bidirectional_fusion or
                    restored.model_kwargs != self.snn_config.model_kwargs or
                    restored.temporal_axis != self.snn_config.temporal_axis):
                raise ValueError("Checkpoint paper_unet architecture does not match current plans")
            metadata = checkpoint.get("snn_config")
            if metadata is not None and SNNConfig.from_plans({"snn_config": metadata}).to_dict() != restored.to_dict():
                raise ValueError("Checkpoint paper_unet metadata differs from saved plans")
        state = checkpoint.get("fptt_state")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint is missing fptt_state")
        for name in _CHECKPOINT_CONFIG_FIELDS:
            if name not in state or state[name] != getattr(self.snn_config, name):
                raise ValueError(f"Checkpoint {name} does not match the current SNN plans")
        saved_kwargs = checkpoint.get("init_args", {}).get("plans", {}).get("snn_config", {}).get("model_kwargs", {})
        if saved_kwargs.get("neuron_model", "plif") != self.snn_config.model_kwargs.get("neuron_model", "plif"):
            raise ValueError("Checkpoint neuron_model does not match the current SNN plans; start a new run")
        from snn_nnunet.recurrent_config import recurrent_options
        saved_recurrence = recurrent_options(saved_kwargs.get("recurrent_neuron"))
        current_recurrence = recurrent_options(self.snn_config.model_kwargs.get("recurrent_neuron"))
        # Diagnostic cadence may change on resume; dynamical settings may not.
        for options in (saved_recurrence, current_recurrence):
            if options is not None:
                options.pop("diagnostics")
        if saved_recurrence != current_recurrence:
            raise ValueError("Checkpoint recurrent_neuron does not match the current SNN plans; start a new run")
        for name in ("fue_skip1", "fue_skip2"):
            saved = saved_kwargs.get(name, False)
            current = self.snn_config.model_kwargs.get(name, False)
            if type(saved) is not bool or saved != current:
                raise ValueError(f"Checkpoint {name} does not match the current SNN plans; start a new run")
        if not self.was_initialized:
            self.initialize()
        if any(named_selective_neurons(self._get_adapter())):
            saved_groups = checkpoint['optimizer_state']['param_groups']
            current_groups = self.optimizer.param_groups
            if (len(saved_groups) != len(current_groups) or any(
                    len(saved['params']) != len(current['params'])
                    or saved['weight_decay'] != current['weight_decay']
                    for saved, current in zip(saved_groups, current_groups))):
                raise ValueError(
                    'Checkpoint optimizer conflicts with the temporal regularization policy; '
                    'start a new run instead of restoring this optimizer'
                )
        restored_tensors = {}
        if self.snn_config.use_fptt:
            parameters = dict(self._get_adapter().named_parameters())
            for field in ("avg_weights", "lambdas"):
                saved = state.get(field)
                if not isinstance(saved, Mapping):
                    raise ValueError(f"Checkpoint {field} must be a parameter mapping")
                missing = set(parameters) - set(saved)
                extra = set(saved) - set(parameters)
                if missing or extra:
                    raise ValueError(
                        f"Checkpoint {field} parameter names differ: "
                        f"missing {sorted(missing)}, extra {sorted(extra)}"
                    )
                prepared = {}
                for name, parameter in parameters.items():
                    tensor = saved[name]
                    if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape:
                        raise ValueError(f"Checkpoint {field}[{name}] has an invalid tensor or shape")
                    try:
                        prepared[name] = tensor.detach().to(
                            device=parameter.device, dtype=parameter.dtype
                        ).clone()
                    except (RuntimeError, TypeError) as error:
                        raise ValueError(f"Checkpoint {field}[{name}] cannot be restored") from error
                restored_tensors[field] = prepared

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
            adapter.avg_weights = restored_tensors["avg_weights"]
            adapter.lambdas = restored_tensors["lambdas"]
        self._check_temporal_regularization()

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
        recurrent_cells = list(named_selective_neurons(adapter))
        diagnostic_sums, diagnostic_counts = {}, {}

        window_size = length if config.bidirectional else config.k
        for t0 in range(0, length, window_size):
            end = min(t0 + window_size, length)
            window_length = end - t0
            data_window = slice_temporal_window(data, t0, end, axis)
            target_window = slice_temporal_window(target, t0, end, axis)
            self.optimizer.zero_grad(set_to_none=True)
            window_index = getattr(self, "_recurrent_window_count", 0)
            for _, cell in recurrent_cells:
                frequency = cell.config["diagnostics"]["every_n_windows"]
                cell.collect_diagnostics = bool(frequency and window_index % frequency == 0)
            supervision = (config.model_kwargs.get('deep_supervision', {})
                           if config.model_name == 'paper_unet' else {})
            kwargs = {'return_auxiliary': True} if supervision.get('enabled', False) else {}
            logits = self.network(data_window, t0=t0, chunk_size=window_length, **kwargs)
            if supervision.get('enabled', False):
                from snn_nnunet.deep_supervision import deep_supervision_loss
                task_loss = deep_supervision_loss(logits, target_window, self.loss, supervision['weights'],
                                                  temporal_axis=axis,
                                                  spatial_stride=config.model_kwargs['pool']['stride'])
            else:
                task_loss = self.loss(logits, target_window)
            regularization = torch.zeros_like(task_loss)
            if config.use_fptt:
                regularization = fptt.regularizer_loss(
                    adapter, regularization, config.fptt_alpha,
                    config.fptt_rho, config.fptt_lambda,
                )
            total_loss = task_loss + regularization
            if not torch.isfinite(total_loss).all().item():
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {self.current_epoch}, window {t0}:{end}"
                )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), 12, error_if_nonfinite=True
            )
            self.optimizer.step()
            for _, cell in recurrent_cells:
                if getattr(cell, 'is_linear_attention', False):
                    cell.project_gamma_()
            did_step = True

            if config.use_fptt and did_step:
                fptt.update_running_params(adapter, config.fptt_alpha, config.fptt_beta)
            sampled = {}
            for name, cell in recurrent_cells:
                if cell.collect_diagnostics:
                    sampled.update({f"recurrent/{name}/{key}": value
                                    for key, value in cell.diagnostics().items()})
                cell.collect_diagnostics = False
            if sampled:
                # One host transfer per sampled window, no scalar GPU sync per metric.
                values = torch.stack(list(sampled.values())).detach().cpu().tolist()
                for key, value in zip(sampled, values):
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.) + value
                    diagnostic_counts[key] = diagnostic_counts.get(key, 0) + 1
            self._recurrent_window_count = window_index + 1
            if not config.bidirectional:
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
            "recurrent_diagnostics": {key: value / diagnostic_counts[key]
                                      for key, value in diagnostic_sums.items()},
            "recurrent_diagnostic_counts": diagnostic_counts,
        }

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Use FP32 and consistent empty-region Dice before native metrics are logged."""
        if self.snn_config.overfit_num_volumes:
            return self._evaluate_overfit()
        geometry = nullcontext()
        if self.snn_config.model_name == "paper_unet":
            from snn_nnunet.custom_full_validation import temporary_patch_size
            geometry = temporary_patch_size(self.configuration_manager, self.snn_config.validation_patch_size)
        with geometry, native_fp32_predictor(), native_empty_dice_evaluation():
            return super().perform_actual_validation(save_probabilities=save_probabilities)

    def validation_step(self, batch: dict) -> dict:
        """Compute the native online metrics in FP32, without nnU-Net's AMP wrapper."""
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)

        output = self.network(data)
        loss = self.loss(output, target)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted = torch.zeros_like(output)
            predicted.scatter_(1, output_seg, 1)

        if self.label_manager.has_ignore_label:
            if self.label_manager.has_regions:
                mask = ~target[:, -1:] if target.dtype == torch.bool else 1 - target[:, -1:]
                target = target[:, :-1]
            else:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def _log_recurrent_diagnostics(self, values):
        store = self.logger.local_logger.my_fantastic_logging
        keys = {key for key in store if key.startswith("recurrent/")} | values.keys()
        for key in sorted(keys):
            history = store.setdefault(key, [])
            history.extend([float("nan")] * max(0, self.current_epoch - len(history)))
            self.logger.log(key, values.get(key, float("nan")), self.current_epoch)
        summary = {key: round(value, 6) for key, value in values.items()
                   if key.endswith(("soma/firing_rate", "projection/input_ratio", "lambda/min_gap"))}
        attention_prefixes = {key.removesuffix('/gamma/mean') for key in values
                              if key.endswith('/gamma/mean')}
        for prefix in attention_prefixes:
            for metric in ('gamma/mean', 'gamma/min', 'gamma/max', 'input/rms',
                           'projection/rms', 'projection/input_ratio', 'gradient/gamma/norm'):
                key = f'{prefix}/{metric}'
                if key in values:
                    summary[key] = round(values[key], 6)
        if summary:
            self.print_to_log_file("Recurrent neuron diagnostics:", summary)
        anomalies = {key: value for key, value in values.items()
                     if (key.endswith(("state/nonfinite", "state/growth_anomaly", "/nonfinite", "/missing")) and value > 0)
                     or (key.endswith("lambda/ordered") and value < 1)
                     or (key.endswith("lambda/nearly_equal") and value > 0)
                     or (key.endswith("consecutive_zero_samples") and value >= 3)}
        if anomalies:
            self.print_to_log_file("Recurrent neuron anomalies:", anomalies)

    @torch.no_grad()
    def _log_temporal_parameters(self):
        """Snapshot actual epoch-end weights, independent of diagnostic cadence."""
        if self.local_rank != 0:
            return
        store = self.logger.local_logger.my_fantastic_logging
        for name, cell in named_selective_neurons(self._unwrapped_network()):
            raw = cell.temporal_raw.detach().float()
            coefficients = cell.temporal_coefficients()
            tau = cell.temporal_time_constants()
            # One small device-to-host transfer per cell, once per epoch.
            stats = torch.stack([
                reduction(value, dim=1)
                for value in (raw, coefficients, tau)
                for reduction in (torch.mean, torch.amin, torch.amax)
            ]).cpu().tolist()
            for i, compartment in enumerate(cell.compartment_names):
                prefix = f'temporal/{name}/{compartment}'
                values = {
                    f'{prefix}/{quantity}/{stat}': stats[3*q+s][i]
                    for q, quantity in enumerate(('raw', 'lambda', 'tau'))
                    for s, stat in enumerate(('mean', 'min', 'max'))
                }
                delta = stats[3][i] - cell.config['lambda_init'][i]
                values[f'{prefix}/lambda/delta_init_mean'] = delta
                for key, value in values.items():
                    history = store.setdefault(key, [])
                    history.extend([float('nan')] * max(0, self.current_epoch - len(history)))
                    self.logger.log(key, value, self.current_epoch)
                self.print_to_log_file(
                    f'Temporal parameters epoch {self.current_epoch + 1} {name}/{compartment}: '
                    f'raw={stats[0][i]:.6f} [{stats[1][i]:.6f}, {stats[2][i]:.6f}] '
                    f'lambda={stats[3][i]:.6f} [{stats[4][i]:.6f}, {stats[5][i]:.6f}] '
                    f'delta_init={delta:+.6e} '
                    f'tau={stats[6][i]:.4f} [{stats[7][i]:.4f}, {stats[8][i]:.4f}] slices '
                    f'temporal_weight_decay={cell.config["temporal_weight_decay"]} '
                    f'temporal_fptt_regularization={cell.config["temporal_fptt_regularization"]}'
                )

    @torch.no_grad()
    def _log_attention_gammas(self):
        """Log actual epoch-end coefficients, independent of diagnostic sampling."""
        if self.local_rank != 0:
            return
        store = self.logger.local_logger.my_fantastic_logging
        for name, cell in named_selective_neurons(self._unwrapped_network()):
            if not getattr(cell, 'is_linear_attention', False) or cell.gamma is None:
                continue
            gamma = cell.gamma.detach()
            values = torch.stack((gamma.mean(), gamma.amin(), gamma.amax())).cpu().tolist()
            for stat, value in zip(('mean', 'min', 'max'), values):
                key = f'attention_gamma/{name}/{stat}'
                history = store.setdefault(key, [])
                history.extend([float('nan')] * max(0, self.current_epoch - len(history)))
                self.logger.log(key, value, self.current_epoch)
            self.print_to_log_file(
                f'Attention gamma epoch {self.current_epoch + 1} {name}: '
                f'mean={values[0]:.8g} min={values[1]:.8g} max={values[2]:.8g}')

    def on_train_epoch_end(self, train_outputs: list[dict]):
        if self.snn_config.use_fptt:
            fptt.reset_running_params(self._get_adapter())
        self._log_temporal_parameters()
        self._log_attention_gammas()
        # nnU-Net collates every key; nested recurrent diagnostics are aggregated below.
        super().on_train_epoch_end([{"loss": row["loss"]} for row in train_outputs])
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

        local_diagnostics = {}
        for row in train_outputs:
            for key, value in row.get("recurrent_diagnostics", {}).items():
                count = row["recurrent_diagnostic_counts"][key]
                total, previous_count = local_diagnostics.get(key, (0., 0))
                local_diagnostics[key] = (total + value * count, previous_count + count)
        gathered_diagnostics = [local_diagnostics]
        if self.is_ddp:
            gathered_diagnostics = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_diagnostics, local_diagnostics)
        totals = {}
        for rank_values in gathered_diagnostics:
            for key, (value, count) in rank_values.items():
                previous, n = totals.get(key, (0., 0))
                totals[key] = (previous + value, n + count)
        self._log_recurrent_diagnostics({key: value / count for key, (value, count) in totals.items()})
