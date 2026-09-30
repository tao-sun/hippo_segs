# SNN BraTS nnU-Net v2.8.1 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a runnable Python 3.10 integration that trains, validates, and predicts with the existing sequential SNN models through unmodified nnU-Net v2.8.1 infrastructure.

**Architecture:** An installable `snn_nnunet` package supplies a plans-driven 3D adapter and an externally discovered `nnUNetTrainer` subclass. Thin conversion, planning, and CLI modules call native nnU-Net APIs; the custom code owns only axis mapping, sequential state, per-window TBPTT/FPTT, checkpoint additions, and temporal-axis mirroring exclusion.

**Tech Stack:** Python 3.10, uv, PyTorch 2.5.1/CUDA 12, mamba-ssm 2.2.6.post3, nnunetv2 2.8.1, nibabel, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-snn-nnunet-v2-integration-design.md`

## Global Constraints

- Pin `nnunetv2==2.8.1`, use Python 3.10, keep PyTorch 2.5.1, and use the CPython 3.10 CUDA 12/PyTorch 2.5/CXX11-ABI-false Mamba wheel.
- Never modify or vendor the installed `nnunetv2` package; use its official `nnUNet_extTrainer` discovery.
- Do not implement custom preprocessing, sampling, augmentation, region conversion, loss, cross-validation, predictor, sliding window, Gaussian fusion, ensembling, export, or postprocessing.
- Preserve the existing SNN architecture and absolute `time_step`; do not introduce Conv3D or reset neuronal state between `k` windows.
- Keep nnU-Net patch size `[128, 128, 128]`, global batch size `4`, three region outputs, and no deep supervision.
- Persist every architecture/inference choice in `SNNPlans`; environment variables may locate the trainer but must not define the network.
- Default `nnUNet_compile=false`; keep native SGD/Nesterov, PolyLR, `DC_and_BCE_loss`, validation, checkpoints, and predictor behavior.
- Preserve all pre-existing uncommitted working-tree changes and stage only files owned by each task.

## Review Focus

- A temporal length not divisible by `k`, including `k` larger than the sequence, must process one short final window and perform `ceil(length/k)` updates (Tasks 3 and 8).
- Duplicate modality matches, subject-ID collisions, and a formatted dataset colliding with a non-identical raw destination must fail before copying partial data (Task 5).
- A second run targeting an existing results folder with different model, axis, `k`, or FPTT settings must fail rather than overwrite or resume (Task 10).
- External trainer discovery must work from an unrelated current directory in a fresh Python process, not only from the repository root (Tasks 1 and 10).
- FPTT checkpoint restore must reject hyperparameter mismatches and restore tensors with each parameter's device and dtype (Task 9).

---

## File Map

- `model.py`: canonical shared `build_model` factory over the existing four architectures.
- `snn_fptt.py`: legacy training remains functional and delegates model construction to `model.build_model`.
- `snn_nnunet/network_adapter.py`: plans schema, axis mapping, sequential inference, and state detachment.
- `snn_nnunet/fptt.py`: unchanged FPTT mathematics and state serialization.
- `snn_nnunet/dataset_conversion.py`: extracted BraTS discovery and raw-format conversion only.
- `snn_nnunet/prepare_plans.py`: native planning/preprocessing delegation and derived plans management.
- `snn_nnunet/trainer.py`: minimal trainer overrides for construction, windows, logging, mirroring, and checkpoints.
- `snn_nnunet/cli.py`: user-facing orchestration over native APIs.
- `tests/conftest.py`: deterministic fake sequential cores and temporary nnU-Net path fixtures.
- Required and focused test modules: adapter, temporal axis, FPTT, loss, mirroring, inference, conversion, plans, trainer, checkpoint, CLI, and smoke.
- `README.md`: installation and raw-data-to-prediction instructions.

### Task 1: Python 3.10 Runtime and Installable Package

**Files:**
- Modify: `pyproject.toml:1-37`
- Modify: `.python-version`
- Modify: `uv.lock`
- Create: `snn_nnunet/__init__.py`
- Create: `tests/test_environment.py`

**Interfaces:**
- Consumes: existing top-level modules `model`, `spike_neurons`, `spikmamba`, and `surrogate`.
- Produces: importable distribution `hippo-segs`, package `snn_nnunet`, and exact runtime dependencies available to all later tasks.

- [ ] **Step 1: Write the failing environment test**

Assert Python is 3.10, `importlib.metadata.version("nnunetv2") == "2.8.1"`, `snn_nnunet` imports, and the package remains importable after `os.chdir(tmp_path)`.

- [ ] **Step 2: Run the test and record the dependency failure**

Run: `uv run --python 3.10 pytest tests/test_environment.py -v`

Expected: FAIL because `nnunetv2` and `snn_nnunet` are not installed in a Python 3.10 project environment.

- [ ] **Step 3: Make the project installable under Python 3.10**

Set `.python-version` to `3.10`, set `requires-python = ">=3.10,<3.11"`, add `nnunetv2==2.8.1`, replace the CPython 3.9 Mamba source URL with the verified CPython 3.10 URL, enable package installation, and configure setuptools to include `snn_nnunet` plus the four existing top-level model modules. Add the minimal package initializer.

- [ ] **Step 4: Resolve and synchronize the environment**

Run: `uv lock --python 3.10`

Run: `uv sync --python 3.10`

Expected: both commands succeed with nnU-Net 2.8.1 and the CPython 3.10 Mamba wheel locked.

- [ ] **Step 5: Run the environment test**

Run: `uv run --python 3.10 pytest tests/test_environment.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the runtime boundary**

```bash
git add .python-version pyproject.toml uv.lock snn_nnunet/__init__.py tests/test_environment.py
git commit -m "build: target Python 3.10 and nnU-Net 2.8.1"
```

### Task 2: Shared Model Factory and Persisted Configuration

**Files:**
- Modify: `model.py:1-547`
- Modify: `snn_fptt.py:1383-1410`
- Create: `snn_nnunet/network_adapter.py`
- Create: `tests/test_network_adapter.py`

**Interfaces:**
- Consumes: existing `SNNBraTS`, `SNNBraTSUNetShallow`, `SNNBraTSUNetMedium`, and `SNNBraTSUNetDeep` constructors.
- Produces: `model.build_model(model_name: str, out_channels: int = 3, patch_size: int = 4, linear_projection: bool = True, residual_connections: bool = True, dwconv2d_spiking: bool = True, patch_embedding_spiking: bool = False, vss_output_spiking: bool = True, input_skip: bool = False) -> nn.Module`; immutable `SNNConfig.from_plans(plans: Mapping[str, Any]) -> SNNConfig`; `SNNConfig.to_dict() -> dict[str, Any]`; `build_core(config: SNNConfig) -> nn.Module`.

- [ ] **Step 1: Add failing factory and configuration tests**

Test all four names, unknown-name rejection, `orig` with `patch_size=4` and all agreed booleans, exact parsing of the approved `snn_config`, channel-count validation, invalid axis, and non-positive `k`.

- [ ] **Step 2: Verify the tests fail**

Run: `uv run --python 3.10 pytest tests/test_network_adapter.py -v`

Expected: FAIL because the shared factory and `SNNConfig` do not exist.

- [ ] **Step 3: Add the canonical factory without changing architecture code**

Move only the dispatch behavior into `model.build_model`; keep the existing constructor defaults and make the legacy `snn_fptt.build_model` a compatible delegate so current experiments continue working.

- [ ] **Step 4: Implement strict plans parsing**

Add `SNNConfig` with the exact persisted fields from the spec. Reject missing keys, unexpected model names, non-boolean model flags, channel counts other than 4/3, axes outside 0–2, and non-positive `k`.

- [ ] **Step 5: Run focused and legacy model tests**

Run: `uv run --python 3.10 pytest tests/test_network_adapter.py test_input_skip.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the shared construction contract**

```bash
git add model.py snn_fptt.py snn_nnunet/network_adapter.py tests/test_network_adapter.py
git commit -m "refactor: share SNN model construction with nnU-Net"
```

### Task 3: Temporal-Axis Adapter and Stateful Inference

**Files:**
- Modify: `snn_nnunet/network_adapter.py`
- Create: `tests/conftest.py`
- Create: `tests/test_temporal_axis.py`
- Create: `tests/test_inference_contract.py`
- Modify: `tests/test_network_adapter.py`

**Interfaces:**
- Consumes: `SNNConfig`, `build_core`, and core contract `forward(x_win: Tensor, t0: int) -> Tensor`.
- Produces: `slice_temporal_window(x: Tensor, start: int, end: int, axis: int) -> Tensor`; `to_core_layout(window: Tensor, axis: int) -> Tensor`; `from_core_layout(logits: Tensor, axis: int) -> Tensor`; `SNNnnUNetAdapter.forward(x: Tensor, *, t0: int = 0, chunk_size: int | None = None) -> Tensor`; `forward_window`; `detach_states`.

- [ ] **Step 1: Write failing shape, ordering, state, and chunk tests**

Use a deterministic fake core to assert `[2,4,16,20,24] -> [2,3,16,20,24]` on axes 0/1/2, increasing absolute time, tile-A/tile-B/tile-A equality, and equal outputs for chunk sizes 1/8/16. Add a final-window case with temporal length 17 and `k=8`.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run --python 3.10 pytest tests/test_network_adapter.py tests/test_temporal_axis.py tests/test_inference_contract.py -v`

Expected: FAIL because the adapter methods are absent.

- [ ] **Step 3: Implement window-local layout mapping**

Use `narrow` on tensor dimension `2 + axis`, then `movedim` only on that window. Restore logits with the inverse movement and concatenate on the native temporal dimension.

- [ ] **Step 4: Implement the adapter's sequential forwards**

`forward_window` calls the existing core once with absolute `t0`; `forward` starts at zero by default and walks through the supplied tensor in `chunk_size or self.k` chunks without detaching or resetting between chunks. Delegate `detach_states` to the core.

- [ ] **Step 5: Run adapter tests**

Run: `uv run --python 3.10 pytest tests/test_network_adapter.py tests/test_temporal_axis.py tests/test_inference_contract.py -v`

Expected: PASS, including the non-divisible and `k > length` cases.

- [ ] **Step 6: Commit the 3D network contract**

```bash
git add snn_nnunet/network_adapter.py tests/conftest.py tests/test_network_adapter.py tests/test_temporal_axis.py tests/test_inference_contract.py
git commit -m "feat: adapt sequential SNNs to nnU-Net volumes"
```

### Task 4: FPTT Module with Preserved Mathematics

**Files:**
- Create: `snn_nnunet/fptt.py`
- Create: `tests/test_fptt.py`

**Interfaces:**
- Consumes: any `nn.Module` with named trainable parameters.
- Produces: `init_running_params`, `regularizer_loss`, `update_running_params`, `reset_running_params`, `export_fptt_tensors`, and `restore_fptt_tensors` with the formulas in `snn_fptt.py:2027-2056` unchanged.

- [ ] **Step 1: Write failing numerical FPTT tests**

For a two-parameter module, assert cloned initial weights, zero lambdas, exact regularizer terms, exact lambda/average updates, and reset-to-average behavior. Assert exported tensors are CPU copies and restored tensors match each parameter's device/dtype.

- [ ] **Step 2: Verify failure**

Run: `uv run --python 3.10 pytest tests/test_fptt.py -v`

Expected: FAIL because `snn_nnunet.fptt` does not exist.

- [ ] **Step 3: Move the formulas verbatim and add serialization helpers**

Store `avg_weights` and `lambdas` on the unwrapped model exactly as the legacy implementation does; serialization helpers must not alter the update mathematics.

- [ ] **Step 4: Run FPTT and legacy toggle tests**

Run: `uv run --python 3.10 pytest tests/test_fptt.py test_fptt_toggle.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the isolated FPTT logic**

```bash
git add snn_nnunet/fptt.py tests/test_fptt.py
git commit -m "feat: isolate preserved FPTT state operations"
```

### Task 5: Extracted BraTS Dataset Conversion

**Files:**
- Create: `snn_nnunet/dataset_conversion.py`
- Create: `tests/test_dataset_conversion.py`

**Interfaces:**
- Produces: `SubjectFiles`; `discover_subjects(root: Path) -> list[SubjectFiles]`; `is_nnunet_raw_dataset(root: Path) -> bool`; `remap_brats24_segmentation(source: Path, destination: Path) -> None`; `convert_or_register_dataset(dataset_root: Path, dataset_id: int, dataset_name: str, nnunet_raw: Path) -> Path`.

- [ ] **Step 1: Write failing discovery and conversion tests**

Create labeled and unlabeled NIfTI subjects; assert modality suffix order, `imagesTr/labelsTr/imagesTs` routing, labels `1/2/3` unchanged and `4 -> 0`, affine/header geometry unchanged, and generated region metadata. Add ZIP rejection, duplicate modality, incomplete subject, sanitized-ID collision, and non-identical destination-collision tests.

- [ ] **Step 2: Verify failure**

Run: `uv run --python 3.10 pytest tests/test_dataset_conversion.py -v`

Expected: FAIL because conversion APIs do not exist.

- [ ] **Step 3: Implement deterministic subject discovery**

Group files by containing subject directory, require exactly four modalities and at most one segmentation, produce stable sanitized identifiers, and validate the entire source before creating destination content.

- [ ] **Step 4: Implement conversion and formatted-dataset registration**

Use `shutil.copy2` for images, nibabel for the single label remap, and nnU-Net's `generate_dataset_json` with the exact regions. Already formatted data is copied byte-for-byte only when its ID/name and destination are compatible.

- [ ] **Step 5: Run conversion tests**

Run: `uv run --python 3.10 pytest tests/test_dataset_conversion.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the thin converter**

```bash
git add snn_nnunet/dataset_conversion.py tests/test_dataset_conversion.py
git commit -m "feat: convert extracted BraTS24 data for nnU-Net"
```

### Task 6: Native Preparation and Derived SNN Plans

**Files:**
- Create: `snn_nnunet/prepare_plans.py`
- Create: `tests/test_prepare_plans.py`

**Interfaces:**
- Consumes: nn-U-Net 2.8.1 `extract_fingerprints`, `plan_experiments`, and `preprocess` APIs plus `SNNConfig` serialization.
- Produces: `derive_snn_plans(stock: Mapping[str, Any], snn_config: Mapping[str, Any]) -> dict`; `create_snn_plans(dataset_id: int, stock_identifier: str, snn_config: Mapping[str, Any]) -> Path`; `run_native_prepare(dataset_id: int, fingerprint_processes: int, preprocess_processes: int) -> Path`; atomic `update_snn_config` and result-compatibility validation.

- [ ] **Step 1: Write failing plans tests**

Assert a deep copy changes only plans name, `3d_fullres.patch_size`, `batch_size`, and top-level `snn_config`; assert stock `data_identifier` and nested resampling/normalization metadata are unchanged. Mock native APIs to assert integrity verification, stock planning, and only stock `3d_fullres` preprocessing are invoked.

- [ ] **Step 2: Verify failure**

Run: `uv run --python 3.10 pytest tests/test_prepare_plans.py -v`

Expected: FAIL because plan helpers do not exist.

- [ ] **Step 3: Implement pure plans derivation and atomic writes**

Require `3d_fullres`; preserve all stock keys; set `[128,128,128]`, batch `4`, identifier `SNNPlans`, and validated configuration. Write a same-directory temporary JSON and replace atomically.

- [ ] **Step 4: Implement native preparation delegation**

Call the public 2.8.1 APIs, use the plans identifier returned by `plan_experiments`, preprocess the stock `3d_fullres` configuration once, create `SNNPlans.json`, and verify the referenced preprocessed folder exists.

- [ ] **Step 5: Run plans tests**

Run: `uv run --python 3.10 pytest tests/test_prepare_plans.py -v`

Expected: PASS.

- [ ] **Step 6: Commit native preparation**

```bash
git add snn_nnunet/prepare_plans.py tests/test_prepare_plans.py
git commit -m "feat: derive SNN plans from native nnU-Net planning"
```

### Task 7: Trainer Construction, Native Loss, Mirroring, and Logger

**Files:**
- Create: `snn_nnunet/trainer.py`
- Create: `tests/test_loss.py`
- Create: `tests/test_mirroring.py`
- Create: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `SNNConfig`, `SNNnnUNetAdapter`, and nn-U-Net 2.8.1 `nnUNetTrainer`.
- Produces: externally discoverable `nnUNetTrainerSNNFPTT`; current-signature `build_network_architecture`; no-op `set_deep_supervision_enabled`; filtered native mirror axes; custom logger-key initialization; safe DDP/`OptimizedModule` unwrapping.

- [ ] **Step 1: Write failing trainer contract tests**

Assert explicit epochs/iterations/lr/decay/oversampling/deep-supervision values, the static hook returns an adapter built from plans, `set_deep_supervision_enabled` does not require `.decoder`, and external discovery returns a subclass of `nnUNetTrainer`.

- [ ] **Step 2: Add failing native-loss and mirroring tests**

With a region label manager, assert `_build_loss()` is exactly nn-U-Net `DC_and_BCE_loss` whose Dice member uses `MemoryEfficientSoftDiceLoss`. Mock `super` mirroring to `(0,1,2)` and assert the three exact filtered tuples and assignment to inference axes.

- [ ] **Step 3: Verify failure**

Run: `uv run --python 3.10 pytest tests/test_trainer.py tests/test_loss.py tests/test_mirroring.py -v`

Expected: FAIL because the trainer does not exist.

- [ ] **Step 4: Implement only the required construction overrides**

Set training constants after `super().__init__`, parse plans once, build the adapter through the current hook, leave `_build_loss` and `configure_optimizers` inherited, and filter the mirror tuple returned by `super`.

- [ ] **Step 5: Extend the existing local logger safely**

Add epoch lists for task loss, FPTT regularization, total loss, optimizer updates, `k`, and FPTT mode. Ensure every list has one value per completed epoch so native checkpointing and `plot_progress_png` remain valid.

- [ ] **Step 6: Run trainer contract tests**

Run: `uv run --python 3.10 pytest tests/test_trainer.py tests/test_loss.py tests/test_mirroring.py -v`

Expected: PASS.

- [ ] **Step 7: Commit the minimal trainer shell**

```bash
git add snn_nnunet/trainer.py tests/test_trainer.py tests/test_loss.py tests/test_mirroring.py
git commit -m "feat: add plans-driven nnU-Net SNN trainer"
```

### Task 8: Windowed Training and FPTT Lifecycle

**Files:**
- Modify: `snn_nnunet/trainer.py`
- Modify: `tests/test_trainer.py`
- Modify: `tests/test_fptt.py`

**Interfaces:**
- Consumes: adapter forward kwargs `t0`/`chunk_size`, native `self.loss`, native GradScaler contract, and FPTT functions.
- Produces: `train_step(self, batch: dict) -> dict`; `on_train_epoch_end(self, train_outputs: list[dict])`; exactly one optimizer update per window.

- [ ] **Step 1: Write failing update-count and target-window tests**

For temporal length 128 assert 8/16/128 optimizer steps for `k=16/8/1`, native-axis target slicing for axes 0/1/2, and two updates for length 17 with `k=16`. Assert calls go through the trainer's wrapped network `forward`, not `module.forward_window`.

- [ ] **Step 2: Write failing FPTT mode tests**

Disabled mode must have zero init/regularizer/update/reset calls. Enabled `k=16` mode must have eight regularizer/update calls per minibatch and one reset before `super().on_train_epoch_end`.

- [ ] **Step 3: Verify failure**

Run: `uv run --python 3.10 pytest tests/test_trainer.py tests/test_fptt.py -v`

Expected: FAIL because native `train_step` performs one whole-patch update.

- [ ] **Step 4: Initialize FPTT only when enabled**

Override `initialize` by calling `super().initialize()` first, unwrapping the constructed adapter, and calling `init_running_params` only when `use_fptt` is true. Repeated initialization remains an error through the native guard.

- [ ] **Step 5: Implement DDP-safe per-window training**

Mirror native transfer/autocast/scaler logic, call `self.network(data_window, t0=t0, chunk_size=window_length)`, calculate `self.loss(logits, target_window)`, add optional FPTT, clip at 12, step once, update optional FPTT, and detach adapter state.

- [ ] **Step 6: Implement epoch aggregation and FPTT reset order**

Aggregate task/reg/total/update metrics across local iterations and DDP ranks, reset running parameters once when enabled, call `super().on_train_epoch_end`, then log one custom value per epoch.

- [ ] **Step 7: Run trainer/FPTT tests**

Run: `uv run --python 3.10 pytest tests/test_trainer.py tests/test_fptt.py -v`

Expected: PASS.

- [ ] **Step 8: Commit windowed optimization**

```bash
git add snn_nnunet/trainer.py tests/test_trainer.py tests/test_fptt.py
git commit -m "feat: train SNN windows with TBPTT and optional FPTT"
```

### Task 9: Native-Compatible Checkpoint Extension

**Files:**
- Modify: `snn_nnunet/trainer.py`
- Create: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: native `save_checkpoint`/`load_checkpoint`, `SNNConfig`, and FPTT serialization helpers.
- Produces: extra `fptt_state` key with atomic save; compatibility validation; native predictor-compatible checkpoint.

- [ ] **Step 1: Write failing checkpoint tests**

Assert all native keys remain, disabled checkpoints omit auxiliary tensors, enabled checkpoints round-trip averages/lambdas, filename and dict loads both work, and predictor-style reads ignore the extra key. Add mismatched `k`, mode, alpha/beta/rho/lambda, device, and dtype cases.

- [ ] **Step 2: Verify failure**

Run: `uv run --python 3.10 pytest tests/test_checkpoint.py -v`

Expected: FAIL because checkpoints have no FPTT extension.

- [ ] **Step 3: Implement save extension after `super`**

On global rank zero, read the completed native checkpoint, add only the specified state, save to a same-directory temporary file, and replace atomically. Respect disabled native checkpointing.

- [ ] **Step 4: Implement validated restore around `super`**

Load metadata once, validate all persisted values before state restoration, delegate native restore, initialize FPTT only when enabled, and restore tensors against named parameters with exact device/dtype.

- [ ] **Step 5: Run checkpoint tests**

Run: `uv run --python 3.10 pytest tests/test_checkpoint.py -v`

Expected: PASS.

- [ ] **Step 6: Commit checkpoint compatibility**

```bash
git add snn_nnunet/trainer.py tests/test_checkpoint.py
git commit -m "feat: persist FPTT state in native checkpoints"
```

### Task 10: Native CLI Orchestration and External Discovery

**Files:**
- Create: `snn_nnunet/cli.py`
- Create: `tests/test_cli.py`
- Modify: `tests/test_inference_contract.py`
- Modify: `tests/test_prepare_plans.py`

**Interfaces:**
- Consumes: conversion/preparation functions, native `run_training`, native `nnUNetPredictor`, native postprocessing utilities, and `nnUNet_extTrainer`.
- Produces: `build_parser() -> argparse.ArgumentParser`; `configure_runtime() -> None`; `main(argv: Sequence[str] | None = None) -> int`; five required subcommands.

- [ ] **Step 1: Write failing parser and delegation tests**

Assert all documented arguments and defaults; mock native calls to verify `3d_fullres`, `nnUNetTrainerSNNFPTT`, `SNNPlans`, GPU count, continue/validation flags, exact predictor constructor settings, checkpoint normalization, fold lists, and sequential fold order.

- [ ] **Step 2: Add failing safety and discovery tests**

Assert incompatible existing result plans fail before native launch. Start a subprocess from `tmp_path` with only the installed project and CLI-configured external trainer path; assert nn-U-Net resolves `nnUNetTrainerSNNFPTT`.

- [ ] **Step 3: Verify failure**

Run: `uv run --python 3.10 pytest tests/test_cli.py tests/test_inference_contract.py tests/test_prepare_plans.py -v`

Expected: FAIL because the CLI does not exist.

- [ ] **Step 4: Implement runtime setup and `prepare`**

Set `nnUNet_extTrainer` to the installed package directory and default `nnUNet_compile` to false. Validate required native path variables, convert/register the dataset, and call native preparation.

- [ ] **Step 5: Implement native train/validate/full-cv delegation**

Persist the chosen config atomically, validate any existing results config, then call `run_training` with the exact native flags. Loop folds synchronously for `full-cv`.

- [ ] **Step 6: Implement exact native prediction and optional postprocessing**

Instantiate `nnUNetPredictor` directly, initialize it from the computed results folder, invoke its raw-data prediction method, and call only official postprocessing utilities when requested.

- [ ] **Step 7: Run CLI and discovery tests**

Run: `uv run --python 3.10 pytest tests/test_cli.py tests/test_inference_contract.py tests/test_prepare_plans.py -v`

Expected: PASS, and the inference object type is exactly `nnUNetPredictor`.

- [ ] **Step 8: Commit the user-facing workflow**

```bash
git add snn_nnunet/cli.py tests/test_cli.py tests/test_inference_contract.py tests/test_prepare_plans.py
git commit -m "feat: expose native nnU-Net SNN workflow"
```

### Task 11: Native Synthetic Smoke Workflow

**Files:**
- Create: `tests/test_smoke_nnunet.py`
- Modify: `tests/conftest.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: the full package, native planning/preprocessing, trainer, checkpoints, and exact native predictor.
- Produces: one reproducible integration test covering the requested lifecycle without production-scale compute.

- [ ] **Step 1: Write the synthetic dataset fixture and failing smoke test**

Create at least five tiny four-channel NIfTI training cases with region-compatible labels and one test case. Use temporary native path variables and a test-only lightweight network factory; do not replace any nn-U-Net data, validation, checkpoint, sliding-window, or export component.

- [ ] **Step 2: Run through preparation and capture the first failure**

Run: `uv run --python 3.10 pytest tests/test_smoke_nnunet.py -m integration -v -s`

Expected: FAIL at the first incomplete integration boundary, with native fingerprinting/planning/preprocessing already exercised where reachable.

- [ ] **Step 3: Add the explicit smoke-test substitution seam**

Register the `integration` pytest marker. In the test, monkeypatch `snn_nnunet.network_adapter.build_core` to return `TinySequentialCore` and derive a temporary smoke-only plans copy with patch `[8,8,8]` and batch `1`; keep the production `SNNPlans` object and its tests fixed at `[128,128,128]` and batch `4`.

- [ ] **Step 4: Assert the full native lifecycle**

Verify trainer initialization, one `train_step`, inherited `validation_step`, checkpoint save/load, native split generation with disjoint folds, predictor reconstruction from results plans/checkpoint, prediction, resampling, and segmentation export.

- [ ] **Step 5: Run the smoke test and focused production contracts**

Run: `uv run --python 3.10 pytest tests/test_smoke_nnunet.py -m integration -v -s`

Run: `uv run --python 3.10 pytest tests/test_loss.py tests/test_mirroring.py tests/test_inference_contract.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the native smoke workflow**

```bash
git add pyproject.toml tests/test_smoke_nnunet.py tests/conftest.py
git commit -m "test: cover native nnU-Net SNN lifecycle"
```

### Task 12: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `.gitignore` only if generated nn-U-Net smoke artifacts require a narrow ignore rule.

**Interfaces:**
- Consumes: final CLI and verified commands.
- Produces: exact installation, preparation, training, resume, validation, prediction, full-CV, DDP, and troubleshooting documentation.

- [ ] **Step 1: Write a documentation checklist test**

Add assertions in `tests/test_cli.py` that README includes Python 3.10/uv, `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`, `nnUNet_extTrainer`, every subcommand, `k`, FPTT/TBPTT, 2,000 updates/epoch for `k=16`, temporal mirroring, pseudo-Dice versus actual validation, resume, multi-GPU, and five-fold prediction.

- [ ] **Step 2: Verify the checklist fails**

Run: `uv run --python 3.10 pytest tests/test_cli.py -k readme -v`

Expected: FAIL against the legacy README.

- [ ] **Step 3: Rewrite the README around the native pipeline**

Keep a brief legacy-workflow note, but make the nn-U-Net 2.8.1 path primary. Include copy-paste commands from `uv sync --python 3.10` and environment setup through raw conversion, five folds, validation, one-fold prediction, five-fold ensemble prediction, and optional native postprocessing.

- [ ] **Step 4: Run the complete verification matrix**

Run: `uv run --python 3.10 pytest -v`

Run: `uv run --python 3.10 python -c "import snn_nnunet; from snn_nnunet.trainer import nnUNetTrainerSNNFPTT; from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor; print('imports ok')"`

Run: `uv run --python 3.10 pytest tests/test_smoke_nnunet.py -m integration -v -s`

Run: `git diff --check`

Expected: all tests PASS, import prints `imports ok`, and diff check is silent.

- [ ] **Step 5: Record completion evidence**

Capture exact command output, changed files, assumptions, any 2.8.1 incompatibilities, and optimizer updates per epoch for the final handoff. Do not claim GPU production training was run unless it actually was.

- [ ] **Step 6: Commit documentation and verification updates**

```bash
git add README.md tests/test_cli.py .gitignore
git commit -m "docs: document native nnU-Net SNN workflow"
```
