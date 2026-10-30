# Selective Spatial Multi-Timescale LIF

The optional cell is `SelectiveMultiTimescaleLIF` in `snn_nnunet/recurrent_neuron.py`.
It replaces `_SpikMambaEncoderAdapter.post_plif` after the existing VSS residual
addition and `post_norm` GroupNorm. Both the first and additional VSS adapters
receive independently constructed cells. Depths `[1,1,1]` give three cells;
`[1,1,2]` gives four. The stem, downsampling, patch embedding/tokenization,
within-slice Spatial Mamba, Temporal Mamba, FUE, decoder, skips and continuous
segmentation head retain their original paths.

The current Spatial Mamba implementation ends its SS2D fusion in GroupNorm and
output projection, without the GELU shown in the request's conceptual diagram.
This feature preserves the implementation audited in this checkout. The original
PLIFNode is composed as `cell.soma`: learned `w`, ATan surrogate, threshold 1,
hard reset to zero, and `detach_reset=True` are reused without new reset equations.
The global `neurons.model` still selects PLIF or SiLIF everywhere else, including
the decoder. A globally SiLIF model can therefore use this cell's PLIF soma at VSS
outputs while retaining SiLIF decoder neurons.

## Configuration and experiment

Use `brats24_config.yaml` for the full neuron. It selects `k: 4` and a fresh
run directory. The earlier baseline YAML, now removed, used `k: 1`. Dataset 1,
fold 0, 600 epochs, LR .001, temporal axis 0, FPTT, both `[162,128,128]`
patches, validation interval 50 and the other model/training settings are preserved.
Optimizer, schedule, loss, preprocessing, augmentation and volume aggregation
continue to come from the existing native pipeline. No pretrained weights are
requested. The launcher refuses a nonempty run directory.

The grouped interface is:

```yaml
snn:
  neurons:
    model: plif
  encoder:
    hierarchical: true
    stage:
      spiking: true
      recurrent_neuron:
        model: selective_multitimescale_lif
        compartments: 3
        lambda_init: [0.50, 0.80, 0.95]
        spatial_mixing:
          enabled: true
          kernel_size: 3
          padding: zeros
          eta_init: 0.1
          kernel_init: near_identity
          kernel_center_init: 0.95
          kernel_surround_init: 0.001
          max_kernel_l1_norm: 1.0
        selective_readout:
          enabled: true
          gate: sigmoid
          gate_kernel_size: 1
          gate_init: zeros
          static_gate_init: 0.5
          fusion: weighted_sum
        memory_projection:
          kernel_size: 1
          bias: false
          init: xavier_uniform
        memory_dtype: float32
        direction: forward
        diagnostics:
          every_n_windows: 100
          near_lambda_gap: 0.001
          growth_ratio_threshold: 100.0
```

All optional leaves have these defaults. Unsupported equations, unknown keys,
non-increasing or out-of-range lambdas, invalid eta/static gates, unsupported
kernels/padding/readout/fusion, and an initialization exceeding the L1 budget are
rejected. Initializers that lose strict ordering or reach an endpoint in FP32 are
also rejected, without a minimum lambda gap. Only hierarchical `orig` models support this option. There is no list
of selected stages. With `stage.spiking: false`, the original GELU output remains
active and no recurrent cell is instantiated. Omitting the option or setting it
to null preserves the baseline architecture and parameter names. A baseline YAML
also clears recurrence inherited from dataset plans; an explicit null normalizes
to absence in saved plans. Direct CLI users can pass `--recurrent-neuron` with a
JSON mapping, or JSON `null` to select the baseline.

For ablations, copy the full experiment to a new YAML with a distinct `run_dir`
and change only these two booleans under `recurrent_neuron`:

| Experiment | `spatial_mixing.enabled` | `selective_readout.enabled` |
|---|---|---|
| Full cell | `true` | `true` |
| No spatial recurrence | `false` | `true` |
| Static learned readout | `true` | `false` |
| Neither spatial mixing nor adaptive readout | `false` | `false` |

Every combination retains all three memories and the PLIF soma. Static gates are
learned sigmoid logits per compartment/channel, initialized at .5 and broadcast
across spatial positions. Disabled branches have no disconnected trainable weights.
No source edits or auxiliary losses are needed.

## Recurrence and numerical behavior

Per-channel positive time constants are cumulative softplus increments, with
`tau_fast = 1e-6 + softplus(raw_fast)` and `lambda = exp(-1/tau)`. Initialization
converts the requested lambdas in FP64 and uses stable inverse softplus
`x + log(-expm1(-x))` before creating trainable FP32 parameters. There is no
additional minimum separation of the learned lambdas. Their ordering and near
collapse are monitored; as with any finite-precision parametrization, extreme
optimization can cause floating-point saturation and is reported by diagnostics.

Each compartment has a separate zero-padded 3x3 depthwise convolution. Its signed
raw kernel is divided by `max(1, L1(raw)/bound * (1 + 4*eps_fp32))`. The small
roundoff allowance prevents FP32 reductions from exceeding the bound. The default
center is .95 and eight alternating signed neighbors have magnitude .001, giving
L1=.958 without strong smoothing. Normalization remains differentiable throughout
training; there is no optimizer projection or parameter freezing. Eta is a learned
sigmoid per compartment/channel initialized at .1.

For each slice F, independently for each r:

```
mixed_r = (1-eta_r) * previous_r + eta_r * DWConv_r(previous_r)
memory_r = lambda_r * mixed_r + (1-lambda_r) * F
gate_r = sigmoid(Conv1x1_r(concat(F, memory_r)))
R = sum_r(gate_r * memory_r)
I = F + Conv1x1_memory_projection(R)
spikes, soma_voltage = original_PLIF(I)
```

Without spatial mixing, `mixed_r = previous_r`. Without selective readout,
`gate_r = sigmoid(static_logits_r)`. Dynamic gate convolutions start at zero
weights/biases, hence .5 gates. R is an unnormalized sum. The bias-free projection
uses unscaled Xavier uniform. No additional gain, gate or normalization is applied.

All dendritic calculations, spatial mixing, decay operations and stored memories
are FP32 with autocast disabled locally. Feature casts remain differentiable.
The effective current and unchanged PLIF soma both stay FP32. The user's
follow-up requires full FP32 throughout the network: reduced-precision inputs,
parameter conversions and checkpoints are rejected, and outer autocast and CUDA
TF32 are disabled. Module device moves remain supported. No conversion to BF16 or
FP16 is performed; attempted module conversions fail before weights or states change.

For a bounded input with infinity norm at most A, zero padding and kernel L1 <= 1
imply `||DWConv(M)||inf <= ||M||inf`. Since eta is in (0,1), spatial mixing is
nonexpansive. With 0 < lambda < 1 the dendritic update is a convex combination of
a nonexpansively transformed previous state and the input. Therefore memory norm
never exceeds `max(||M_initial||inf, A)`, even when learned parameters change
between steps while satisfying the bounds. For fixed parameters it is a strict
contraction with factor at most lambda_max. Sigmoid readout is bounded, so
`||R||inf <= 3A`; the finite learned projection then gives a bounded effective
current for fixed weights. Somatic spike/reset dynamics never alter this argument.

## State lifecycle, FPTT, checkpoints

The buffers `fast_memory`, `medium_memory`, `slow_memory` have shape BCHW and
start at zero; `somatic_membrane` exposes the original soma's voltage. They are
nonpersistent buffers and do not become model weights or checkpoint contents.

`t0=0` starts a new independent patch, resetting dendrites and triggering the
original soma initialization. Within a patch, each batch element keeps its own
aligned states, and absolute offsets must advance consecutively. Calling the same
nonzero offset twice, or skipping ahead, raises an error. A changed batch/spatial
shape initializes a fresh cell sequence. Device transfers must occur before a
sequence or be followed by `t0=0`: the reused PLIF's voltage/spike attributes are
not registered buffers and cannot migrate an active sequence automatically.
Clients must still use `t0=0`
for unrelated patches of the same shape; shape alone cannot identify a patient.
Explicit `cell.reset()` begins a fresh sequence. Somatic firing/reset never resets
the dendrites. `cell.detach()` truncates gradients while retaining state values.

The native trainer traverses non-overlapping `range(0, length, k)` windows,
including a short final window. It detaches graphs after each optimizer step in
both existing modes. The next window of the same crop uses its absolute offset
and retained values. Every new training crop starts at zero. No batch-index-based
sequence inference or cross-crop/patient carryover is added.

FPTT remains the repository's actual algorithm:

1. `init_running_params` registers **all named parameters** in running averages
   and auxiliary lambdas, including every cell parameter family.
2. The original task loss and original FPTT parameter regularizer are differentiated
   on each window; optimizer, gradient clipping and loss definitions are unchanged.
3. `update_running_params` runs after every successful parameter update.
4. State graphs are detached, with values preserved.
5. The original epoch-end `reset_running_params` copies averaged parameters.

When FPTT is disabled, the existing plain TBPTT path runs without these auxiliary
parameter operations or regularizer. No special neuron hooks, manual parameter
registration, or replacement of FPTT is introduced.

Validation keeps the existing ordered chunk adapter and FP32 nnU-Net predictor.
Each tile and each mirror forward starts its own sequence at zero. Temporal chunks
inside that patch do not overlap or revisit slices. Overlapping volume tiles are
independent sequences, so their recurrence is reset before reprocessing shared
voxels. Native Gaussian logit reconstruction and metrics remain unchanged.

Native checkpoints still store all parameters and original FPTT auxiliary tensors.
Strict baseline PLIF/SiLIF loading retains old key names. New-cell and baseline
checkpoints are different architectures and cannot be interchanged. Trainer resume
also compares recurrence settings (excluding diagnostic cadence) before loading.
No old-to-new checkpoint transfer is implemented or requested.

## Diagnostics

Configure `diagnostics.every_n_windows` (0 disables, default 100). Sampling includes
the first window. No spatial maps or graphs are retained for logging. The last
slice of each sampled window supplies activity summaries; scalar sums/counts are
averaged across sampled windows, patches and DDP ranks at epoch end. There is one
host transfer for the sampled metric vector. Unsampled forwards do not compute
activity diagnostics; existing nonfinite loss/gradient checks remain enabled.

The existing native logger/checkpoint stores `recurrent/<module path>/...` values:
mean/min/max and sampled q10/q50/q90 of each lambda and gate, lambda gap statistics,
strict ordering and near-equality fraction, eta summaries, effective kernel L1
maxima, gate saturation fraction, static coefficients, memory/input/projection/
effective-current RMS, projection/input RMS ratio, soma voltage RMS and firing
rate, nonfinite-state and growth flags, gradient norms/missing/nonfinite counts,
and consecutive zero-gradient diagnostic samples for every parameter family.
Quantiles use at most approximately 1024 evenly sampled entries. Text logs print
compact firing/current/gap summaries and anomaly indicators. Persistent zero
counts describe sampled windows, not every optimizer update. Diagnostic thresholds
only control reporting; they add no losses or constraints to the model.

## Verification and launch

Run the complete suite from the repository:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

Tests cover equations, initializations, constraints, all four ablations and all
parameter families, AMP FP32 states, resets/offsets/short tails, all VSS counts,
ordered Temporal Mamba sequence integration, PLIF/SiLIF strict checkpoint roundtrips,
YAML/CLI/plans reconstruction, actual trainer FPTT and TBPTT steps with native
BCE+Dice loss, logger serialization, and native overlapping Gaussian volume tiles.
A separate CPU smoke test uses the actual Mamba `selective_scan_ref` with the
experiment widths `[64,128,256]`, `k=4`, native training loss and FPTT, followed by
inference; its synthetic spatial crop is reduced to `16x16`. Most CPU architecture tests inject a lightweight scan in place of the external
CUDA scan; they do not validate that CUDA kernel. Two session-only tests compare
pre-change audited model source, loading its baseline state_dict strictly and
checking bit-identical logits for PLIF and SiLIF. They skip elsewhere if the audit
snapshot in /tmp is unavailable. These are synthetic checkpoints, not archived
trained experiment checkpoints.

CUDA is unavailable on this node (`torch 2.5.1+cu124`, `cuda.is_available=False`).
`nnUNet_raw` and `nnUNet_preprocessed` are unset. A real BraTS data-loader training
step, CUDA selective-scan execution, real-patient validation and peak GPU memory
measurement therefore remain unverified. Synthetic CPU validation preserves the
native tiling and reconstruction code; it cannot establish segmentation quality,
long-run optimization behavior or full-patch GPU capacity. No 600-epoch run was
started and no experimental checkpoint/results were written.

After setting the existing `nnUNet_raw` and `nnUNet_preprocessed` paths on a
compatible GPU node with prepared Dataset001/SNNPlans data, launch from this repo:

```bash
.venv/bin/python -m snn_nnunet.run_model --config brats24_config.yaml
```

Use the fresh run directory in that YAML; do not point it at an existing run.
The launcher enforces its existing nonempty-directory protection. The plan and
session verification ledger are in
`docs/superpowers/plans/2026-10-29-selective-recurrent-neuron.md` (the repository
ignores docs/, so this root-level feature document is the versionable report).

## Changed files and verified results

Created:

- `snn_nnunet/recurrent_config.py`: defaults, schema validation, stable initializers.
- `snn_nnunet/recurrent_neuron.py`: constrained spatial operator, reusable cell, states, diagnostics.
- `brats24_config.yaml`: new scratch experiment, requested k=4.
- `test_selective_recurrent_neuron.py`: unit, gradient, numerical and lifecycle tests.
- `test_selective_recurrent_integration.py`: architecture, baseline, YAML/CLI/plans and chunk tests.
- `test_selective_recurrent_training.py`: native train_step, FPTT/TBPTT, volume tiles, checkpoint and smoke tests.
- `SELECTIVE_RECURRENT_NEURON.md`: complete feature report and instructions.
- `docs/superpowers/plans/2026-10-29-selective-recurrent-neuron.md`: audit, plan, execution ledger (git-ignored by existing repository rules).

Modified by this feature:

- `snn_nnunet/model.py`: optional activation selection, independent instances for every VSS, constructor plumbing.
- `snn_nnunet/config_layout.py`: optional grouped YAML field mapping.
- `snn_nnunet/run_model.py`: YAML validation and recurrence CLI forwarding, including explicit null for baseline runs.
- `snn_nnunet/cli.py`: JSON option and saved-plans forwarding.
- `snn_nnunet/network_adapter.py`: config validation and canonical absence for baseline plans.
- `snn_nnunet/trainer.py`: sampled native diagnostics and recurrence checkpoint compatibility guard.

Pre-existing edits in `README.md`, `spikmamba.py` and other dirty baseline files
were preserved. The earlier baseline YAML was removed at the user's request. No modifications to the FPTT helper,
original neuron implementations, inference tiling, original experiment, losses,
or data/augmentation pipeline were needed.

Verification on 2026-10-29:

- Baseline before implementation: **82 passed**, four dependency warnings.
- Final `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q -ra`:
  **147 passed, 1 skipped, 4 warnings in 31.52 s**. The skipped test is
  `test_actual_experiment_cuda_scan_fptt_amp_smoke`: CUDA unavailable.
- Actual-width CPU reference SS2D + FPTT training and inference passed.
- PLIF and SiLIF audit-snapshot baseline state_dict loading passed with bit-identical logits.
- `python -m compileall -q` for the package and three new test modules: exit 0.
- `git diff --check`: exit 0. CLI help exposes `--recurrent-neuron`.
- Read-only independent code review completed. Its one Important finding was
  FP32-infeasible initializers; three regression cases failed before the fix
  and the complete final suite passed afterward. No outstanding review findings.
- Earlier development failures (missing implementation, FP32 kernel rounding,
  test deepcopy of non-leaf tensors, dtype/membrane diagnostics, YAML k value,
  null-plan compatibility, lossless state casting, extreme initializers) were
  resolved and are not remaining failures.

Implementation decisions: work in the current dirty checkout to retain audited
architecture changes; use the explicitly requested k=4 despite source YAML k=1;
treat shape/device changes as a fresh cell sequence; reject nonconsecutive
nonzero offsets; reject FP32-infeasible initializers without imposing a decay
separation. These choices preserve the approved recurrence. Real data/GPU,
archived trained checkpoints, DDP execution and long-run accuracy remain
unverified, as described above.

## Full FP32 policy requested in follow-up

The native full precision supported by this Mamba architecture is FP32. The entire
SNN now enforces it through `snn_nnunet/precision.py`, including bare core forwards,
the nnU-Net adapter, the recurrent neuron and native volume inference.

- All continuous activations, neuron currents/membranes, learned parameters,
  gradients, optimizer state and FPTT auxiliary tensors remain FP32.
- Active spiking outputs retain their existing discrete values, represented as
  FP32 tensors. Decoder/head behavior and spiking equations are unchanged.
- FP16/BF16 autocast is disabled inside forwards, including when called directly
  under an external autocast context.
- CUDA matrix multiplication and cuDNN convolution TF32 are disabled; matmul
  precision is `highest`. These PyTorch backend flags are process-wide and the
  policy is reasserted at model construction and forward.
- `.half()`, `.bfloat16()` and `.to(dtype=...)` requesting any dtype other than
  FP32 are rejected before any weights or state values are changed. FP32 device
  moves before initialization, or followed by a `t0=0` reset, are supported. This
  replaces the earlier buffer-only conversion guard.
- Network inputs and loaded model state_dict tensors must already be FP32.
  Upcasting previously rounded inputs or checkpoints cannot restore their lost
  information and is therefore rejected, rather than done silently. FP64 network
  execution is not supported by this policy/backend.

Follow-up files: created `snn_nnunet/precision.py` and
`test_snn_full_precision.py`; modified the core base classes/forward guards in
`model.py`, adapter/cell guards, `inference.py`, prior unit tests and this report.
No new YAML switch is needed. Plan:
`docs/superpowers/plans/2026-10-29-full-fp32-policy.md`.

New tests inspect continuous-layer and PLIF input/output dtypes under external
CPU autocast, parameter/state preservation on rejected dtype changes, rejection
of lower-precision inputs/checkpoints, forward keyword API compatibility, TF32
configuration flags, and FP32 gradients/optimizer/FPTT tensors. Actual execution
on CUDA remains unverified because no CUDA GPU is available on this node.

Final follow-up verification (2026-10-29): **174 passed, 1 skipped, 4 dependency
warnings in 31.05 s**, using the complete pytest suite. The skip remains the CUDA
smoke test, because CUDA is unavailable. Compileall and git diff --check passed.
The independent read-only review found no Critical or Important issues. Its one
Minor finding was the existing inability to migrate an active original-PLIF
sequence across devices; documentation now explicitly requires transfer before
initialization or a subsequent t0=0 reset. Extending original somatic state device
migration is deferred because it is outside this precision change.
