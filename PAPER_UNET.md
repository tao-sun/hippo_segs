# Original paper U-Net in nnU-Net

Fresh paper runs replace obsolete architecture settings in shared preparation plans before validating the requested YAML. Resume continues to validate saved settings strictly; preprocessing does not need to be repeated.

`paper_unet` runs through the existing SNN nnU-Net launcher, plans, adapter, trainer, validation, resume, and FP32 sliding-window predictor. There is no separate training pipeline. Use `snn.variant: paper_unet` in `unet_config.yaml`; settings live directly under `snn`. There is no experiment mode selector. Change experiment settings only before starting a fresh run.

The reference is `git show main:model.py`, class `SNNBraTS`. The branch `SpikingArch2_v2` has no root `model.py`; its `snn_nnunet/model.py` implements Mamba encoders and has different activation/head behavior. The new core copies the original graph: 4→32→64→128 encoder with three pooling operations, three 128-channel upsampling stages, concatenations with **pool2** and **pool1**, and the original convolution plus **non-spiking PLIF voltage integrator** as the head. Kernel/stride/padding, bias, GroupNorm(1,C), and dropout defaults match the reference. Head output channels remain the pipeline's three overlapping BraTS regions.

Spatial padding goes on the bottom/right before each slice, to a multiple of the product of the three pooling strides (8 by default). Final logits are cropped to the input dimensions. Pool and transposed-convolution settings must invert each other exactly. Convolution/head settings must preserve resolution. Channels are configurable, but the graph and its two skip connections are fixed. There are no Mamba, VSS, SS2D, FUE, extra stem, residual decoder, or additional skip modules.


The main YAML enables three supervised outputs with `deep_supervision.resolution: native`
and weights `[1.0, 0.5, 0.25]` (normalized). Auxiliary logits stay at half/quarter
H/W resolution; nearest-exact targets retain the full slice axis and match the
model padding. Missing `resolution` keeps legacy `full` behavior for old configs
and checkpoints. See [DEEP_SUPERVISION.md](DEEP_SUPERVISION.md) for the loss and
fresh-run requirements.

## Enable spiking independently in encoder and decoder

Two boolean switches under `snn.paper_unet` control the activation family:

```yaml
snn:
  paper_unet:
    encoder_spiking: false
    decoder_spiking: false
```

Both default to `true`, preserving existing configurations, weights and neuron choices. Set either to `false` to replace all neuron activations in that section with `nn.GELU()` after the existing GroupNorm and before dropout. The encoder switch controls `conv_block1`–`conv_block3`; the decoder switch controls all eight remaining Conv/Deconv blocks, including skip fusion. Per-block neuron overrides apply only while their section's spiking is enabled. No neuron parameters or memory states are created for a GELU block.

With encoder spiking disabled, no backward selective-neuron scan or bidirectional neuron fusion is built or executed, even if `bidirectional: true`. The final head keeps its existing non-spiking PLIF voltage integrator; the switches do not replace it with GELU. Deep supervision remains available. Flags are saved in plans and reconstructed for inference and checkpoint loading. Changing either switch changes the architecture: select a fresh `run_dir` and train a new run.

## Choose neurons

With `neurons.model: plif`, overrides of `conv_block1`, `conv_block2`, and `conv_block3` choose the three encoder neurons while retaining eight PLIF decoder blocks. Use `selective_multitimescale_lif` for plain SiLIF or `MultiMemoryLinearAttentionLIF` for the attention variant; commenting out these overrides gives an all-PLIF model.

All spiking ConvBlocks and DeconvBlocks use one independent activation after normalization and before dropout. Set:

```yaml
snn:
  paper_unet:
    neurons:
      model: selective_multitimescale_lif
      overrides:
        conv_block2: plif
        deconv_block3: plif
```

Allowed models: `plif`, `selective_multitimescale_lif`, `MultiMemoryLinearAttentionLIF`. Select attention by writing, for example, `conv_block2: MultiMemoryLinearAttentionLIF` in the same overrides mapping. These selections can be mixed, including in decoder blocks. Explicit overrides take priority over the legacy `neurons.selective.model` selector. Supported block names:

- `conv_block1`, `conv_block2`, `conv_block3`
- `deconv_block1`, `deconv1_conv`, `concat1_conv`
- `deconv_block2`, `deconv2_conv`, `concat2_conv`
- `deconv_block3`, `deconv3_conv`

`neurons.plif` configures the somatic LIF parameters. `neurons.multi_memory` holds the memory, decay, spatial mixing, projection and diagnostic settings shared by plain selective and linear attention neurons. `neurons.selective_readout` holds only the plain selective neuron gate settings; attention ignores that section. For both multi-memory neuron types, `neurons.plif` configures the existing somatic PLIF's tau, surrogate and detach-reset settings. The selective spatial neuron's validated equations, initialization and dynamics are unchanged. It is distinct from Mamba's selective scan. `head.plif` configures the final voltage integrator independently. `class_conv` cannot be overridden by a spiking neuron; its `spiking` and `normalization` switches must remain false.

`use_fptt: true` selects the existing FPTT regularizer and running-parameter updates. `false` selects TBPTT. `k` sets the gradient window length. The YAML exposes all four supported FPTT coefficients. Windows retain the neuron states and advance absolute slice offsets; each new patch/sequence starts at `t0=0`, resetting all memories including the head. `detach_states()` retains values and truncates all recurrent graphs.

Set `snn.paper_unet.neurons.multi_memory.temporal_weight_decay: false` to exempt only the selective neurons' learned `temporal_raw` parameters from optimizer weight decay. Keep `temporal_fptt_regularization: true` to retain their FPTT loss terms, auxiliary running-parameter updates, and epoch-end replacement by averaged weights when `use_fptt: true`. Setting that flag to `false` also excludes them from these FPTT operations; with both flags false, the memory coefficients remain trainable through the task loss. Other parameters keep the native `3e-5` decay and their original FPTT behavior. The example YAML disables temporal weight decay and keeps the temporal FPTT option enabled. Missing options default to `true`, preserving legacy behavior. Changing either flag for active selective neurons requires a new run directory; incompatible optimizer policies are rejected on resume.

The trainer recognizes selective cells through a stable capability marker, because nnU-Net external trainer discovery removes and reloads Python modules, invalidating class-identity checks. At initialization it prints and checks each cell's actual optimizer decay, FPTT participation, and trainability; epoch snapshots use the same reload-safe discovery. Existing running jobs retain their already-loaded code and optimizer.

At every training epoch end, after the FPTT running weights are applied, the trainer prints each selective block's fast/medium/slow `raw`, `lambda`, and nominal `tau` in slices as `mean [min, max]`, plus `delta_init` (mean lambda minus its configured initialization). This snapshot runs even when sampled recurrent diagnostics are disabled. Only rank zero prints it. Scalar histories are saved under `temporal/<block>/<compartment>/...` in the checkpoint logger. Tau describes nominal temporal decay; spatial mixing and readout affect the effective memory. Parameter changes alone do not establish that the memory improves segmentation.

The full effective configuration is serialized into `plans.snn_config`, `run_config.json`, and checkpoint initialization plans, with an additional `snn_config` checkpoint metadata copy for this variant. Native trainer loading rejects changed topology, neuron settings, temporal axis, or FPTT settings. Resume can extend the epoch horizon and updates both metadata copies. Periodic, native, final custom and full-fold evaluation use the saved validation patch for this variant. Raw prediction uses nnU-Net's saved training patch geometry. Inference rebuilds from saved plans and weights; do not edit architecture settings to resume an existing run.

## Repeatable fresh training

Optional progressive bidirectional encoder recurrence is documented in [BIDIRECTIONAL_UNET.md](BIDIRECTIONAL_UNET.md), with the computational comparison in [BIDIRECTIONAL_COSTS.md](BIDIRECTIONAL_COSTS.md). Its `snn.bidirectional` switch defaults to false; true applies `snn.bidirectional_fusion` only to encoder blocks using plain `selective_multitimescale_lif`, and enforces whole-patch BPTT with FPTT disabled. Fusion choices are `OR`, `conv_PLIF`, and `Shared_SOMA` (the original `or`/`conv_plif` spellings also work). PLIF, linear attention and decoder blocks remain forward-only. With bidirectionality disabled the fusion setting is inactive. Shared_SOMA fuses memories before pooling and requires adaptive selective readout only in its active blocks; other fusions combine post-pooling outputs. The `conv_plif` module uses Conv1x1 → GroupNorm → PLIF, reusing the configured normalization settings. OR bypasses that entire module and applies standard encoder dropout after fusion.

The YAML launcher enables strict deterministic training by default for new SNN experiments. The example exposes the policy explicitly:

```yaml
snn:
  seed: 42
  deterministic: true
  augmentation_workers: 8
```

Python, NumPy and Torch are seeded before model initialization. Each augmentation worker receives independent reproducible Python/NumPy/Torch seeds; training and validation use separate streams. Batches are consumed in worker order rather than completion order. The native augmentation remains enabled, including Gaussian blur, whose convolution algorithm is selected without timing benchmarks. Training uses the specified worker count and validation uses half that count (at least one). Keep this count fixed when comparing runs.

Strict mode also uses ordered out-of-place additions for the SSN compartment prefix sum: PyTorch 2.5.1 rejects floating CUDA `cumsum` under strict determinism, including in the old epoch logger. Coefficients and logging now share `temporal_time_constants()`. The mathematical parameterization and gradients are preserved; the nonstrict legacy calculation is unchanged. The regression tests check coefficient/gradient equivalence and logging, and reproduce the CUDA rejection before verifying the corrected GPU path.

The launcher sets `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` before starting the training interpreter. cuDNN benchmarking and TF32 are disabled, and Torch deterministic algorithms are required: operations for which Torch cannot provide a deterministic implementation raise an error. This can reduce throughput; the actual GPU epoch cost needs measurement. Custom CUDA extensions, including Mamba selective scan, are not certified by these Torch flags.

The guarantee is scoped to two **fresh runs** with identical code, preprocessed data/split, effective configuration, worker count, GPU count/model and library versions. A different seed deliberately gives a different run. Cross-hardware or cross-version bitwise equivalence is not promised. Resuming an interrupted job does not restore augmentation queues or all RNG states and is not guaranteed to reproduce an uninterrupted run exactly.

The policy is saved in plans/configuration/checkpoints; startup logs show `Reproducibility:` with seed, rank, backend flags and software versions. Resume rejects changes to seed, deterministic mode or worker count. Existing saved plans without these fields retain legacy nondeterministic behavior. Set `deterministic: false` for the previous pipeline. The direct CLI accepts `--seed`, `--deterministic`/`--no-deterministic` and `--augmentation-workers`; absent CLI switches inherit saved plans.

`test_reproducibility.py` runs independent CPU processes through native trainer discovery, augmentation, optimizer updates, FPTT and checkpoint saving to compare actual patch hashes, losses, weights and learned temporal coefficients. The measured comparison is saved in `reproducibility_results/comparison.json`: both deterministic runs match exactly on all these values, whereas both legacy runs differ. Final validation losses were `0.3696393371 / 0.3596045673` before and `0.3628873825 / 0.3628873825` after. These synthetic checks establish repeatability; they do not measure BraTS accuracy or GPU throughput.

## Commands

From the repository root, with the existing Python environment and preprocessed Dataset001/SNNPlans available:

```bash
export nnUNet_raw=/gpfs/scratch1/shared/apiaghiardelli/nnUNet_raw
export nnUNet_preprocessed=/gpfs/scratch1/shared/apiaghiardelli/nnUNet_preprocessed
export nnUNet_extTrainer="$PWD/snn_nnunet"
export nnUNet_compile=false

# New training; run inside a GPU allocation. run_dir must be new/empty.
.venv/bin/python -m snn_nnunet.run_model --config unet_config.yaml
```

The existing batch script also accepts this YAML:

```bash
sbatch --export=ALL,CONFIG_PATH="$PWD/unet_config.yaml" train_validate_snn_brats24_fold0.sbatch
```

Its existing SLURM resource request allocates one GPU, and the launcher uses the allocated visible GPUs inside SLURM, saving the effective count. Outside SLURM, the launcher uses the YAML `gpus` value.

Read run/fold/dataset/epoch settings from the same YAML for resume and evaluation:

```bash
RUN_DIR=$(.venv/bin/python -c 'from pathlib import Path; from snn_nnunet.run_model import _read_experiment; print(_read_experiment(Path("unet_config.yaml"))["run_dir"])')
FOLD=$(.venv/bin/python -c 'from pathlib import Path; from snn_nnunet.run_model import _read_experiment; print(_read_experiment(Path("unet_config.yaml"))["fold"])')
DATASET_ID=$(.venv/bin/python -c 'from pathlib import Path; from snn_nnunet.run_model import _read_experiment; print(_read_experiment(Path("unet_config.yaml"))["dataset_id"])')
EPOCHS=$(.venv/bin/python -c 'from pathlib import Path; from snn_nnunet.run_model import _read_experiment; print(_read_experiment(Path("unet_config.yaml"))["snn"]["epochs"])')

# CPU verification of a saved resumable checkpoint; no training.
.venv/bin/python -m snn_nnunet.resume_training \
  --run-dir "$RUN_DIR" --fold "$FOLD" --epochs "$EPOCHS" --check-only

# Resume, including optional extension to the total epochs specified in YAML.
.venv/bin/python -m snn_nnunet.resume_training \
  --run-dir "$RUN_DIR" --fold "$FOLD" --epochs "$EPOCHS"

# Full-fold evaluation of best checkpoint, existing Dice/postprocessing workflow.
.venv/bin/python -m snn_nnunet.evaluate_snn_fold \
  --dataset-id "$DATASET_ID" --fold "$FOLD" --run-dir "$RUN_DIR" --best

# Prediction from raw nnU-Net channel files (replace input/output directories).
.venv/bin/python -m snn_nnunet.cli predict \
  --dataset-id "$DATASET_ID" --run-dir "$RUN_DIR" --folds "$FOLD" \
  --checkpoint best --input /path/to/imagesTs --output /path/to/predictions
```

The resume helper requires a latest, final or periodic epoch checkpoint; it intentionally does not resume optimizer state from `checkpoint_best.pth` alone. Dataset preparation, postprocessing and label conversion retain their existing behavior.

## Verification

Verified on CPU: **150 tests passed**, two CUDA-specific tests skipped, with four existing SciPy/batchgenerators deprecation warnings. Both CUDA-specific tests then passed on an A100-SXM4-40GB in job 28848614 (the selected GPU-job suite had three passing tests, including the CPU regression). The configured native unidirectional selective model completed one batch of training, epoch-end temporal logging and validation with B=2, patch 81×128×128, k=81, FPTT off and strict determinism on. Peak allocated CUDA memory was 31.924 GiB. See `analysis_results/debug_28848032/verification.json`. This is a synthetic first-batch check, not a complete training or comparative throughput benchmark. `compileall` and `git diff --check` also passed.

Run focused tests with `OMP_NUM_THREADS=1 .venv/bin/python -m pytest test_paper_unet.py -q` and the full available suite with `OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q`.

Tests exercise the actual native trainer initialization, FPTT/TBPTT optimizer steps, validation loss, checkpoint save/load, epoch extension and FP32 sliding-window prediction from a saved synthetic model. They also compare default PLIF outputs exactly with `main:model.py`, check the two neuron families and named block overrides, gradients, temporal window equivalence, reset/detach, odd spatial sizes, layout axes, incompatible options and unknown keys. The test file and guide are at the repository root because the existing repository ignores `tests/` and `docs/`. All synthetic data and checkpoints are temporary; existing experiment outputs are untouched.

The GPU smoke check covers the configured unidirectional selective model. Bidirectional CUDA, multi-GPU execution and full real-data fold evaluation remain unverified. No complete training was started. Dataset preprocessing and comparative GPU performance requirements retain their separate validation scope.
