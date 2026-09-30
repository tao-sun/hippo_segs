# SNN BraTS Integration with nnU-Net v2.8.1

## Purpose

Integrate the repository's existing sequential SNN/SpikMamba BraTS models into a real nnU-Net v2 training, validation, and inference workflow. nnU-Net remains responsible for dataset fingerprinting, preprocessing, planning, sampling, augmentation, region loss, cross-validation, logging, checkpoint scheduling, sliding-window prediction, ensembling, resampling, export, and native postprocessing.

The custom code is limited to:

- adapting the existing 2D sequential SNN contract to nnU-Net's 3D network contract;
- processing a patch sequentially along a configurable preprocessed spatial axis;
- performing one TBPTT/FPTT optimizer update per `k` slices;
- preserving the existing FPTT formulas and auxiliary state;
- excluding the temporal axis from native mirroring;
- converting an extracted BraTS24 GLI dataset into nnU-Net raw format when required;
- providing a thin command-line interface over nnU-Net's native APIs.

The existing SNN architecture will not be converted to Conv3D or redesigned.

## Supported Runtime

- Python 3.10 managed with `uv`.
- PyTorch 2.5.1 with the matching CUDA 12 build already used by the project.
- `nnunetv2==2.8.1`, pinned to the current requested API baseline.
- `mamba-ssm==2.2.6.post3` using the Python 3.10 wheel matching the existing CUDA, PyTorch, and C++ ABI settings.

`pyproject.toml`, `uv.lock`, and the README will be updated together. The integration will use the environment variables required by nnU-Net 2.8.1: `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`.

## External Trainer Discovery

The custom trainer remains in `snn_nnunet/trainer.py`. The installed `nnunetv2` package will not be modified or vendored.

nnU-Net 2.8.1 supports external trainers through `nnUNet_extTrainer`. The project CLI will set this variable to the custom trainer source directory before invoking native training or prediction. This lets both `nnunetv2.run.run_training` and `nnUNetPredictor.initialize_from_trained_model_folder` resolve `nnUNetTrainerSNNFPTT` by the class name stored in checkpoints.

Architecture and inference settings will not come from environment variables. They will be stored in the plans copied into the model-results directory. Users who call native nnU-Net commands directly instead of `python -m snn_nnunet.cli` must set `nnUNet_extTrainer` as documented.

`nnUNet_compile` will default to `false` for project CLI commands because the network has mutable neuronal state and custom Mamba operations. An explicit user override remains possible.

## Package Layout

```text
snn_nnunet/
├── __init__.py
├── network_adapter.py
├── fptt.py
├── trainer.py
├── dataset_conversion.py
├── prepare_plans.py
└── cli.py
```

- `network_adapter.py` owns axis mapping, sequential window execution, model construction from plans, state detachment, and the 3D nnU-Net network contract.
- `fptt.py` moves the existing FPTT functions without changing their mathematics and adds serialization helpers.
- `trainer.py` contains the minimal `nnUNetTrainer` subclass.
- `dataset_conversion.py` handles only extracted BraTS discovery and raw-dataset conversion.
- `prepare_plans.py` delegates fingerprinting, planning, and preprocessing to nn-U-Net and derives `SNNPlans` from the stock plans.
- `cli.py` exposes `prepare`, `train`, `validate`, `predict`, and `full-cv`.

The existing `model.py`, `spike_neurons.py`, `spikmamba.py`, and `surrogate.py` remain the architecture implementation. Existing uncommitted changes in those files are preserved.

## Persisted SNN Configuration

`SNNPlans.json` has a top-level `snn_config` entry. The agreed default is:

```json
{
  "model_name": "orig",
  "model_kwargs": {
    "patch_size": 4,
    "linear_projection": true,
    "residual_connections": true,
    "dwconv2d_spiking": true,
    "patch_embedding_spiking": true,
    "vss_output_spiking": true,
    "input_skip": false
  },
  "temporal_axis": 0,
  "k": 16,
  "use_fptt": true,
  "fptt_alpha": 0.5,
  "fptt_beta": 0.5,
  "fptt_rho": 0.0,
  "fptt_lambda": 2.0,
  "num_input_channels": 4,
  "num_output_channels": 3
}
```

Supported `model_name` values are `orig`, `shallow`, `medium`, and `deep`, using the current `build_model` behavior. Architecture-defining values are persisted even when they equal defaults so prediction cannot silently build a different model.

The nnU-Net spatial patch size remains `[128, 128, 128]`. The model's `patch_size: 4` is the separate internal 2D SpikMamba patch-embedding factor.

Before a native training launch, the project CLI atomically updates `snn_config` with the requested model, temporal axis, `k`, and FPTT mode. If the target results directory already contains plans or checkpoints with incompatible settings, the command fails instead of silently overwriting or resuming it. All folds in one cross-validation experiment use the same persisted configuration.

## Network Adapter

### Contracts

nnU-Net input and output:

```text
input:  [B, 4, X, Y, Z]
output: [B, 3, X, Y, Z]
```

Existing core-model input and output:

```text
input:  [B, k, 4, H, W]
output: [B, 3, k, H, W]
```

`SNNnnUNetAdapter` validates the input and output channel counts against both the plans and the native label manager.

### Axis Mapping

`temporal_axis` is expressed in nnU-Net preprocessed spatial coordinates. The corresponding tensor dimension is `2 + temporal_axis`.

| `temporal_axis` | Sequence coordinate | Frame extraction |
|---|---|---|
| 0 | X | `data[:, :, t, :, :]` |
| 1 | Y | `data[:, :, :, t, :]` |
| 2 | Z | `data[:, :, :, :, t]` |

Only the selected temporal window is rearranged into `[B, k, C, H, W]`. The entire volume is not permuted. Returned window logits are restored to the same native spatial-axis position. Axis extraction and restoration are implemented once and reused by training and inference.

### Stateful Sequence Behavior

Each independent nnU-Net patch or inference tile is a new SNN sequence. The first window starts at absolute `time_step=0`, which uses the existing `PLIFNode` initialization path. No parallel state system is added.

Within one patch, later windows pass their absolute `t0` to the existing core model. Neuronal state therefore continues across all 128 slices. At a TBPTT boundary, `detach_states()` truncates the graph but does not reset membrane state.

`forward(x)` starts a fresh sequence, walks through the full temporal extent in anatomical order, and joins the restored logits. Its internal chunk size defaults to persisted `k`, but changing the chunk size must not change predictions. It never performs optimizer, backward, FPTT, running-parameter, or intermediate reset operations.

The adapter also exposes `forward_window` for direct testing and non-DDP use. During DDP training, each window is sent through the DDP wrapper's normal `forward` method with explicit `t0` and sequence-continuation arguments. This preserves DDP reducer setup and gradient synchronization rather than bypassing the wrapper through a custom method.

## Dataset Conversion

The input must be an extracted directory. ZIP archives are unsupported and rejected.

The converter recursively locates subject directories. Each subject must contain exactly one file for each modality:

- `*t1n*.nii.gz`
- `*t1c*.nii.gz`
- `*t2w*.nii.gz`
- `*t2f*.nii.gz`

A subject may contain one `*seg*.nii.gz`. Duplicate modalities, ambiguous segmentations, identifier collisions, and partial subjects are errors.

Labeled subjects are written to `imagesTr` and `labelsTr`; unlabeled subjects are written to `imagesTs`. Modalities use nnU-Net suffixes `_0000` through `_0003` in T1, T1ce, T2, and FLAIR order.

Images are copied without intensity or geometry changes. Labels are loaded with nibabel, and only raw label `4` is replaced by `0`; labels 0, 1, 2, and 3 remain unchanged. The original affine and relevant NIfTI header geometry are preserved.

`dataset.json` is generated with `nnunetv2.dataset_conversion.generate_dataset_json.generate_dataset_json`, not hand-written. Its region definition is:

```json
{
  "channel_names": {
    "0": "T1",
    "1": "T1ce",
    "2": "T2",
    "3": "FLAIR"
  },
  "labels": {
    "background": 0,
    "whole_tumor": [1, 2, 3],
    "tumor_core": [1, 3],
    "enhancing_tumor": 3
  },
  "regions_class_order": [2, 1, 3]
}
```

nnU-Net's native transforms convert label maps into region tensors.

If the supplied root already has a valid `DatasetXXX_Name` raw structure and `dataset.json`, its images and labels are not transformed. If it is outside `nnUNet_raw`, it is copied unchanged into the expected dataset location after collision checks.

## Planning and Preprocessing

The `prepare` command invokes the public nnU-Net 2.8.1 fingerprinting, planning, and preprocessing APIs with dataset-integrity verification. It does not implement normalization, cropping, resampling, caching, or planning.

After stock planning, it deep-copies the generated plans to `SNNPlans.json`. It preserves stock spacing, transpose metadata, normalization, resampling, preprocessor, architecture metadata, and every other field. It changes only:

- the plans identifier/name required for `SNNPlans`;
- `configurations["3d_fullres"]["patch_size"]` to `[128, 128, 128]`;
- `configurations["3d_fullres"]["batch_size"]` to `4`;
- the new top-level `snn_config` entry.

The `3d_fullres.data_identifier` remains the stock value. Preprocessing is run once through the native stock plan, and the command verifies that the data referenced by `SNNPlans` exists. It does not duplicate preprocessed arrays.

## Custom Trainer

`nnUNetTrainerSNNFPTT` subclasses the installed 2.8.1 `nnUNetTrainer` and uses the current static architecture hook:

```python
build_network_architecture(
    plans_manager,
    configuration_manager,
    num_input_channels,
    num_output_channels,
    enable_deep_supervision=True,
)
```

The hook reads `snn_config` from `plans_manager.plans`, builds the existing model, and wraps it in `SNNnnUNetAdapter`. This same hook is used by native predictor reconstruction.

The trainer explicitly sets:

```text
num_epochs = 300
num_iterations_per_epoch = 250
num_val_iterations_per_epoch = 50
initial_lr = 1e-2
weight_decay = 3e-5
oversample_foreground_percent = 0.33
enable_deep_supervision = False
```

The inherited optimizer configuration supplies SGD with momentum 0.99 and Nesterov, plus `PolyLRScheduler`. No alternative optimizer or scheduler is introduced. `set_deep_supervision_enabled` is a no-op for the adapter because it has no nnU-Net decoder object.

### Native Region Loss

The trainer does not override `_build_loss`. For the region dataset, nnU-Net constructs its native `DC_and_BCE_loss` with `MemoryEfficientSoftDiceLoss`. The latter is nnU-Net's memory-efficient differentiable soft-Dice component; it is not a project-defined or additional loss.

For each window:

```text
task_loss = self.loss(logits_window, target_window)
total_loss = task_loss + fptt_regularizer  # FPTT enabled
total_loss = task_loss                     # FPTT disabled
```

The internal Dice and BCE formulas, weights, target conversion, and region transforms remain unchanged.

### Training Step

`train_step(self, batch: dict) -> dict` follows native device transfer and autocast behavior. Deep supervision is disabled, so the target is the full-resolution native region tensor.

For each `k`-slice window it performs:

1. `optimizer.zero_grad(set_to_none=True)`;
2. DDP-safe adapter forward with the absolute `t0`;
3. matching target-window extraction;
4. native task loss and optional additive FPTT regularizer;
5. scaled or unscaled backward;
6. GradScaler unscale when active;
7. native gradient-norm clipping threshold of 12;
8. one optimizer step and GradScaler update;
9. optional FPTT running-parameter update;
10. SNN state detachment.

There is exactly one optimizer update per temporal window. For temporal length 128 and 250 nnU-Net minibatches per epoch:

| `k` | Updates/minibatch | Updates/epoch | Updates/300 epochs |
|---:|---:|---:|---:|
| 16 | 8 | 2,000 | 600,000 |
| 8 | 16 | 4,000 | 1,200,000 |
| 4 | 32 | 8,000 | 2,400,000 |
| 1 | 128 | 32,000 | 9,600,000 |

These are optimizer-update counts for the distributed training job, not multiplied by the number of DDP ranks.

The returned dictionary retains the native `loss` entry and adds task loss, FPTT regularization, total loss, optimizer-update count, `k`, and FPTT mode. The trainer extends the existing `MetaLogger` configuration and epoch aggregation; it does not replace the logger.

## FPTT

The following existing functions move to `snn_nnunet/fptt.py` without mathematical changes:

- `init_running_params(model)`
- `regularizer_loss(model, reg_loss, alpha, rho=0.0, _lambda=2.0)`
- `update_running_params(model, alpha, beta)`
- `reset_running_params(model)`

When FPTT is enabled, auxiliary tensors are initialized after native network initialization, the regularizer and update run once per window, and `reset_running_params` runs once in `on_train_epoch_end` before calling `super().on_train_epoch_end`. This occurs before native online validation.

When FPTT is disabled, auxiliary tensors are not created, and none of the FPTT functions are called. Training remains TBPTT with one update per `k` slices.

Network unwrapping handles native DDP and `OptimizedModule` safely. No custom distributed implementation is added. Synchronized model updates make the deterministic FPTT auxiliary updates consistent across ranks.

## Mirroring and Augmentation

The trainer calls `super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()` and removes `temporal_axis` from the returned native mirror axes. It also assigns the filtered tuple to `self.inference_allowed_mirroring_axes`.

Expected values are:

| Temporal axis | Allowed mirror axes |
|---:|---|
| 0 | `(1, 2)` |
| 1 | `(0, 2)` |
| 2 | `(0, 1)` |

This affects native volumetric `MirrorTransform` and native predictor TTA. All other nnU-Net augmentation remains inherited. No slice-wise flips or custom `torch.flip` implementation are introduced.

## Checkpoint Extension

Native checkpoint scheduling and contents remain intact. The trainer first calls `super().save_checkpoint`. On global rank zero it then loads that checkpoint, adds `fptt_state`, writes a temporary file in the same directory, and atomically replaces the original.

`fptt_state` records:

- FPTT enabled/disabled;
- `k`;
- alpha, beta, rho, and lambda;
- `avg_weights` and `lambdas` only when FPTT is enabled.

`load_checkpoint` accepts either a filename or native checkpoint dictionary, validates the persisted training configuration against current plans, delegates native restoration to `super`, and restores FPTT tensors to the appropriate device afterward. Predictor reconstruction does not instantiate FPTT state and ignores the extra checkpoint key.

## Validation and Inference

The inherited validation dataloader, 50-iteration validation loop, and `validation_step` are retained. Calling the adapter through `network(data)` starts a new sequence for each validation patch. Validation loss is only the native region loss.

Full-volume validation uses inherited `perform_actual_validation`. Prediction uses exactly `nnunetv2.inference.predict_from_raw_data.nnUNetPredictor`, configured with:

```text
tile_step_size = 0.5
use_gaussian = True
use_mirroring = True
```

The predictor uses the `[128, 128, 128]` configuration patch, checkpoint-provided mirror axes, native Gaussian overlap fusion, resampling, export, and fold parameter averaging. No predictor subclass or custom sliding-window code is introduced.

Optional postprocessing delegates to the native nnU-Net configuration-selection and postprocessing utilities. No custom connected components, morphology, or ET thresholds are added.

## Command-Line Interface

All commands are exposed as `python -m snn_nnunet.cli`.

### `prepare`

Converts an extracted dataset when necessary, generates native metadata, runs verified fingerprinting/planning/preprocessing, derives `SNNPlans`, and checks the shared preprocessed data identifier.

### `train`

Persists the selected `model`, `temporal_axis`, `k`, and FPTT mode in `SNNPlans`, validates result-folder compatibility, and delegates to native `run_training` with configuration `3d_fullres`, trainer `nnUNetTrainerSNNFPTT`, plans `SNNPlans`, and the requested GPU count. Continue-training uses the native checkpoint-selection behavior.

### `validate`

Delegates to native validation-only training execution and supports final or best checkpoints through native flags.

### `predict`

Creates an exact `nnUNetPredictor`, initializes it from the model-results folder, and invokes native raw-data prediction. Fold lists and checkpoint names pass directly to native model loading, including five-fold averaging.

### `full-cv`

Runs folds zero through four sequentially with one shared configuration. It does not launch concurrent folds unless a future explicit option requests that behavior.

## Testing Strategy

Tests are organized around the required files:

- `tests/test_network_adapter.py`
- `tests/test_temporal_axis.py`
- `tests/test_fptt.py`
- `tests/test_loss.py`
- `tests/test_mirroring.py`
- `tests/test_inference_contract.py`

Additional focused tests may be added for conversion, plans, CLI, checkpoints, and the native smoke workflow.

Fast unit tests use a deterministic injectable sequential core where appropriate. This isolates axis mapping, temporal ordering, state boundaries, and chunk invariance from GPU-only Mamba kernels. A separate construction test instantiates all four real model variants with persisted settings.

Coverage includes:

1. `[2, 4, 16, 20, 24]` to `[2, 3, 16, 20, 24]` for every temporal axis;
2. increasing slice order;
3. repeatability of tile A around an independent tile B;
4. equivalent inference for chunk sizes 1, 8, and 16;
5. exact optimizer-step counts for `k=16`, `k=8`, and `k=1` at length 128;
6. zero FPTT calls when disabled;
7. exact per-window FPTT calls and one epoch reset when enabled;
8. exact native region-loss classes;
9. exact mirror-axis tuples;
10. exact native predictor type;
11. `[128,128,128]` in training and predictor configuration;
12. native `splits_final.json` creation/use and disjoint fold membership;
13. BraTS label remapping and NIfTI geometry preservation;
14. checkpoint configuration validation and FPTT round trip.

An integration smoke test creates a tiny synthetic NIfTI dataset and runs real nnU-Net fingerprinting, planning, and preprocessing. It then uses a test-only minimal network substitution to exercise trainer initialization, one training minibatch, inherited validation, checkpoint save/load, predictor reconstruction, native sliding-window prediction, and export without requiring a full CPU execution of the production SNN on a 128-cubed patch. Production adapter behavior and real-model construction are tested separately.

## Error Handling and Safety

Commands fail clearly for:

- missing nnU-Net path variables;
- unsupported Python or dependency versions;
- ZIP input;
- malformed or ambiguous subjects;
- dataset ID/name mismatches;
- raw-dataset destination collisions;
- missing `3d_fullres` stock plans or preprocessing;
- invalid temporal axes or non-positive `k`;
- input/output channel mismatches;
- incompatible existing result plans;
- incompatible checkpoint FPTT configuration;
- missing external trainer discovery.

Dataset directories and result folders are never silently overwritten. Temporary checkpoint and plan writes use same-directory atomic replacement.

## Non-Goals

The integration does not implement custom data loading, sampling, augmentation, normalization, resampling, crop logic, region conversion, Dice/BCE loss, cross-validation, predictor, sliding-window coordinates, Gaussian weighting, fold averaging, export, or postprocessing. It also does not add deep-supervision heads, Conv3D layers, SwinCLNet 96-cubed windows, ZIP ingestion, or custom parallel fold scheduling.

## Completion Evidence

Completion requires recorded output for:

- all unit tests;
- import checks under the Python 3.10 `uv` environment;
- custom trainer initialization;
- one miniature training iteration;
- inherited validation step;
- checkpoint round trip;
- native predictor reconstruction and prediction smoke test.

The final handoff will list changed files, exact commands, test outputs, remaining assumptions, any nnU-Net 2.8.1 incompatibilities, and optimizer-update estimates for the selected `k`.
