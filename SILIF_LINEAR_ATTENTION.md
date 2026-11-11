# MultiMemoryLinearAttentionLIF

This provides a **unidirectional multi-memory LIF neuron with linear attention**. Select `MultiMemoryLinearAttentionLIF` for individual blocks in `unet_config.yaml`, using the same overrides as plain `selective_multitimescale_lif`.

## Select and launch

There is no experiment `mode` selector. Neuron type and bidirectional fusion are independent:

```yaml
snn:
  variant: paper_unet
  bidirectional: true
  bidirectional_fusion: Shared_SOMA  # also OR / conv_PLIF
  paper_unet:
    neurons:
      model: plif
      overrides:
        conv_block1: selective_multitimescale_lif
        conv_block2: MultiMemoryLinearAttentionLIF
        conv_block3: selective_multitimescale_lif
        deconv1_conv: MultiMemoryLinearAttentionLIF
```

Only plain SiLIF encoder blocks get backward scans and fusion when `bidirectional: true`. Attention and PLIF always scan forward, and every decoder block stays forward-only. An attention block can still receive features containing future context from an earlier bidirectional SiLIF block; its own recurrence has no backward scan. With `bidirectional: false`, the fusion setting is inactive everywhere. To make all three encoders use attention, set all three overrides to `MultiMemoryLinearAttentionLIF`.

Shared memory parameters live under `neurons.multi_memory`. The sibling `neurons.selective_readout` section applies only to `selective_multitimescale_lif`; its enabled, gate and fusion settings do not apply to the attention neuron. Attention does not require adaptive selective readout to be enabled. The new neuron retains **no adaptive/static gate parameters**. `neurons.model: MultiMemoryLinearAttentionLIF` also selects attention as the default for all spiking blocks, while named overrides take priority. The old global `neurons.selective.model` form and the old name `SiLIF_LinearAttention` remain readable as compatibility aliases; explicit overrides take priority.

```bash
.venv/bin/python -m snn_nnunet.run_model --config unet_config.yaml
```

Choose a fresh run directory when changing the architecture. No full training was started by this implementation.

For the separate existing `orig` VSS backbone in `brats24_config.yaml`, the selector remains:

```yaml
snn:
  encoder:
    stage:
      recurrent_neuron:
        model: MultiMemoryLinearAttentionLIF
```

That VSS setting applies to all configured multi-memory blocks within its encoder stages. Its training protocol is independent of the paper U-Net protocol in `unet_config.yaml`.

## Actual input provenance and unchanged interfaces

`_SpikMambaEncoderAdapter` in `model.py` already computes continuous `pre_spike = post_norm(x)` and passes it directly to its SiLIF output neuron. `PaperBlock` already supplies Conv2d/ConvTranspose2d → GroupNorm output. Both inputs are FP32 `[B,C,H,W]`.

Thus **F_pre = F_mem = F_soma = the original neuron input**. No new feature branch, conversion of spikes into a continuous estimate, or change to any upstream binary pathway is needed. Even when a convolution receives spikes, its normalized output entering SiLIF is continuous.

The neuron interface remains `forward(features, time_step) -> (spikes, voltage)`. Output spikes have the original `[B,C,H,W]` shape. No encoder/decoder topology or native adapter layout changes were required.

For the actual primary `128×128×128` patch:

| Stage | F_pre, each memory, and final readout | Q and each K/V | Number of memory tokens |
|---|---|---|---:|
| 1 | `[B,64,64,64]` | `[B,4096,32]` | 12,288 |
| 2 | `[B,128,32,32]` | `[B,1024,32]` | 3,072 |
| 3 | `[B,256,16,16]` | `[B,256,32]` | 768 |

Each attention module has its own parameters. Shapes follow actual block channels/resolution for other configurations.

## Mathematics and state lifetime

The original ordered decay parameterization, initial values, bounded depthwise kernels, learned mixing coefficients and update operation order are unchanged:

\[
\widetilde M_{z-1}^r=(1-\eta_r)M_{z-1}^r+\eta_r DWConv^r(M_{z-1}^r),
\qquad M_z^r=\lambda_r\widetilde M_{z-1}^r+(1-\lambda_r)F_z.
\]

Readout occurs **after these writes**. With dk=dv=32, one head, and φ(x)=ELU(x)+1:

\[
Q=W_QF_z,\quad K_r=W_K^rM_z^r+e_r,\quad V_r=W_V^rM_z^r,
\]
\[
S=\sum_r\phi(K_r)^\top V_r,\quad n=\sum_r\sum_j\phi(K_{r,j}),
\quad A_i=\frac{\phi(Q_i)S}{\phi(Q_i)n+10^{-6}},\quad R^{LA}=W_OA.
\]

Q/K/V flatten spatial dimensions to `[B,HW,32]`. Each memory has independent K and V projections and a learnable 32-element key-scale embedding. There is no spatial positional encoding, relative bias, extra adaptive gate, local-memory residual or second readout.

Production forms only `[B,32,32]` sufficient statistics, `[B,32]` key sums and linear-size token tensors. It never forms `[B,HW,3HW]`. Every Query can retrieve from all positions of all three updated memories.

The attention neuron normalizes the final memory-to-soma projection before adding the original features:

\[
I_z^{eff}=F_z+\gamma\odot\operatorname{GroupNorm}_{1,C}(W_R(R_z^{LA})).
\]

`W_O` is the new attention output projection; `W_R` is the existing memory-to-soma projection. They are deliberately distinct. The output normalization is `GroupNorm(1, channels, eps=1e-5, affine=False)`, independently per sample and slice over channels and spatial positions. It acts after `W_R`, only on the residual. Gamma is applied exactly once after this existing normalization, immediately before the residual addition. Memory updates and plain SiLIF/Shared_SOMA keep their existing computation. Recurrent diagnostics measure the scaled normalized projection actually added to the input. The existing PLIF receives that current with the original learnable tau, threshold, hard reset to zero, detach-reset policy and ATan surrogate. Memories never depend on somatic resets.

Implementation shares the original neuron forward. `_aggregate_memories` retains the old gate/sum operations for plain SiLIF, while `_project_memories` retains its original convolution. The attention subclass overrides aggregation and adds output GroupNorm to the projection hook. Its diagnostics report memory/soma and attention-gradient statistics without inventing obsolete gate activations.

The original three transient memory buffers, `_next_step`, `reset()` and `detach()` are retained. Subject starts reset at time_step=0; nonconsecutive offsets are rejected. Native TBPTT detaches states at window boundaries, and FPTT includes attention parameters in its existing auxiliary state and regularizer. There is no new temporal state or cross-subject attention cache. Anatomy remains forward-causal; global retrieval is spatial within the current updated memories.

Builder choices not otherwise specified: bias-free Q/K/V/O 1x1 convolutions, Xavier-uniform weights, zero initial scale embeddings, epsilon=1e-6. Attention initialization preserves the parent RNG streams, keeping common model weights identical at a fixed seed; no baseline initialization is altered.

## Fixed gamma and per-channel LayerScale

In `unet_config.yaml`, the existing `snn.paper_unet.neurons.multi_memory` section supports:

```yaml
attention_gamma_mode: learnable_per_channel
attention_gamma_init: 0.1
attention_gamma: 0.05  # Used only when attention_gamma_mode: fixed.
```

Every attention cell independently owns an FP32 `gamma` parameter shaped `[1, C, 1, 1]`, broadcast over its `[B, C, H, W]` slice. All coefficients start at `attention_gamma_init`. The current three encoder cells have shapes `[1,32,1,1]`, `[1,64,1,1]`, `[1,128,1,1]`: 224 additional learned coefficients. Any attention cells selected in the decoder also own independent parameters. Learned coefficients are constrained to the closed interval `[0, 1]`: the native trainer projects them after each optimizer step, and FPTT projects them after its epoch-end parameter reset. The forward path also bounds the effective gain. This preserves the direct parameterization, initial gain and optimizer learning rate; no sigmoid or additional normalization is introduced. `attention_gamma_init` must be in `[0, 1]` in learned mode. Fixed-scalar ablations retain their existing behavior.

`fixed` uses the existing Python scalar `attention_gamma` and adds no parameter/checkpoint keys. Missing mode in older saved configurations resolves to `fixed`; missing scalar resolves to the historical 1.0. The supplied YAML now explicitly selects `learnable_per_channel` with init 0.05. For the previous ablation, choose `fixed` with scalar 0.05; for the original unit scaling use `fixed` with scalar 1.0.

Learnable gamma is enumerated by the existing optimizer and saved in the model state dict. The trainer checks optimizer membership at initialization and after restoration. Optimizer hyperparameters, weight decay and FPTT policy are unchanged; FPTT includes gamma with the other learned attention parameters. Checkpoints restore learned values and optimizer momentum, rather than reinitializing gamma. Out-of-range gamma values from older unconstrained checkpoints are projected into `[0, 1]` on load without editing the saved tensors. Start a fresh run directory for a controlled comparison with unconstrained training. Changing mode changes the parameter layout, so start a new run when switching fixed/learnable modes. Normal data handling and five-volume overfit mode are independent of this choice.

Training logs include actual epoch-end mean/min/max under `attention_gamma/<block>/...`, even if recurrent diagnostics are disabled. At the existing `diagnostics.every_n_windows` cadence (100 by default), `recurrent/<block>/...` also records:

- `input/rms`: local features.
- `projection/rms`: the complete scaled attention residual.
- `projection/input_ratio`: scaled residual RMS / local feature RMS.
- `gamma/mean`, `gamma/min`, `gamma/max` and `gradient/gamma/norm` when a gradient is available.
- `gradient/gamma/missing`: unavailable gradients are distinguished from zero gradients.

## Precision, correctness and checkpoints

The repository deliberately enforces FP32 rather than enabling AMP. The new module keeps this policy: outer autocast is disabled during attention; FP16/BF16 input/parameter conversion is rejected. CPU BF16-autocast tests still return FP32 results identical to non-autocast execution. No global training precision settings were relaxed. Actual GPU mixed-precision execution was not measured.

Tests compare efficient attention with a small explicit quadratic **test-only** kernel reference. Output tolerances are rtol=2e-5/atol=2e-6; gradient tolerances are rtol=1e-4/atol=3e-6, covering Query/input, all memories, independent K/V weights, embeddings and output weights. Zero and strongly negative inputs remain finite in the tested FP32 cases. Epsilon protects zero denominators; it is not a guarantee against arbitrary exploding FP32 projections.

Other checks include global influence of distant spatial positions from every scale, absence of quadratic intermediate shapes, actual continuous Query provenance, read-after-write, identical memory trajectories and common initial parameters, unchanged soma settings, causality, state resets/detachment, all configured stages/depths, strict state-dict reloads, config roundtrips and native FPTT/TBPTT optimizer steps. The CPU `orig` interface test explicitly substitutes only the unavailable CUDA selective-scan primitive; it is not a benchmark or numerical validation of CUDA Mamba.

Eight pre-change fingerprints cover unidirectional original SiLIF and all three fusion architectures in training/evaluation, with all SiLIF decoder placements exercised. Together with the four OR/Conv+PLIF golden regressions on plain SiLIF encoders, they test original outputs, gradients, parameter keys, subject resets and strict checkpoint loading with rtol=atol=0 in the recorded PyTorch CPU setup. New attention checkpoints have different readout keys and must be resumed with the new selector; no silent old-to-new checkpoint migration is performed. Checkpoints with unchanged plain SiLIF topology keep their keys. Old bidirectional PLIF checkpoints describe the previous architecture and require that previous implementation; PLIF now intentionally stays unidirectional.

The output GroupNorm has no parameters or persistent buffers, so it adds no state-dict keys. Existing attention weights still load strictly in fixed mode; learnable mode adds per-cell `.gamma` entries and requires a matching checkpoint mode. Historical unnormalized forward results differ because the residual is now normalized. Use a new run directory for the normalized experiment; reproducing historical unnormalized attention results requires the previous implementation. Improved segmentation convergence has not yet been established by a GPU training run.

## Continuous MACs and spike-driven ACs

All attention work is continuous, including Query projection. A zero-valued continuous memory is never relabeled as a spike input. Per spatial site, per slice and sample:

| Readout operation | Continuous MACs |
|---|---:|
| Original three adaptive gates | `6 C²` |
| New Q + three K + three V + output projection | `256 C` |
| Three KᵀV statistics | `3×32² = 3072` |
| Query retrieval | `32² = 1024` |
| Query normalization dot product | `32` |
| Total new attention readout | `256 C + 4128` |
| Existing memory-to-soma W_R, unchanged | `C²` |
| Existing three depthwise memory convolutions, unchanged | `27 C` when enabled |

Original gate readout parameters are `6C²+3C`. New attention parameters are `256C+96`; memory and soma parameters are common. New readout spike-driven ACs are **zero**. The sign/magnitude of overhead depends on C and spatial dimensions; smaller theoretical counts do not prove faster GPU execution.

`profile_linear_attention.py` counts the actual primary backbone's total/trainable parameters without executing Mamba, then reports exact continuous costs for every primary SiLIF block at the configured patch geometry. It does **not** label those neuron costs as whole-backbone totals: remaining Mamba/SS2D and mixed-input synaptic costs are not included in that primary ledger.

For an additional fully accounted example, the report contains a separately labeled unidirectional `paper_unet` with encoder SiLIF and decoder PLIF, patch `81×128×128`. Its continuous GMAC and theoretical binary-input GAC at unit activity are separate fields. Actual firing rates reduce event counts; training's inverted-dropout-scaled features require scaled/continuous arithmetic. No energy or sparse hardware speed claim follows from these counts.

Forward MAC tables exclude normalization, PLIF/reset/surrogate, coefficient functions, kernel normalization, backward and optimizer. Elementwise fields are **partial tensor-site counts**, not exhaustive FLOPs; cross-memory statistic/vector additions and individual ELU+1 operations are not separately itemized.

## Measurements and limits

Regenerate the scoped report with:

```bash
.venv/bin/python -m snn_nnunet.profile_linear_attention
# On a CUDA host, measure the same readout workloads with synchronization:
.venv/bin/python -m snn_nnunet.profile_linear_attention --measure-cuda
```

Readout timing uses identical FP32 features/memories, channel dimensions, batch size, hardware, warmups and repetitions for baseline and attention. Forward and forward+backward are reported separately. Timed tiny `16×16` feature maps are **not** full MRI volume inference. The timed scope excludes memory updates, W_R, output GroupNorm, soma, optimizer and data pipeline. Samples/second therefore describes this readout workload, not whole-training throughput.

CUDA is unavailable in this session. Full-model GPU peak memory, full training throughput and complete MRI inference latency are unmeasured. No dataset experiment or comparison with the stated 0.758 Dice baseline was run. Linear-size attention avoids the quadratic token matrix, but full-volume BPTT still retains recurrent graphs and projected activations; peak memory and actual training stability require the GPU experiment.

Measured numbers, test results and changed-file inventory are in `analysis_results/linear_attention/report.json` and `verification.json`.

## Recorded results before output GroupNorm

Verification of the original attention extension: **102 passed**, with four pre-existing SciPy deprecation warnings. All 16 pre-change golden cases remain bitwise equal in their recorded CPU/PyTorch scope. The independent review's CPU-fixture failure was corrected and the complete suite rerun.

| Primary BraTS orig experiment | Baseline | Linear Attention |
|---|---:|---:|
| Total model parameters | 2,401,586 | 1,999,122 |
| Trainable parameters | 2,401,580 | 1,999,116 |
| Continuous GMAC in SiLIF blocks only, B1/128³ patch | 46.682604 | 25.900876 |

These GMAC values include spatial memory mixing, memory projection and the respective readout. They are not full-network operation totals. New readout spike-driven GAC is zero; primary whole-backbone spike activity was not measured.

Median readout CPU timing, B1 and 16×16 feature maps, one thread, two warmups/seven repetitions:

| C | Baseline forward ms | Attention forward ms | Baseline forward+backward ms | Attention forward+backward ms |
|---|---:|---:|---:|---:|
| 64 | 0.793 | 1.618 | 2.502 | 4.923 |
| 128 | 1.838 | 1.927 | 5.723 | 5.844 |
| 256 | 5.736 | 2.079 | 17.325 | 9.204 |

These CPU results show that overhead depends on channel width. They do not establish full-volume or GPU acceleration.

Auxiliary paper_unet comparison (not the primary BraTS orig backbone):

| Variant | Continuous GMAC | Potential spike-driven GAC at unit activity |
|---|---:|---:|
| baseline | 32.073449 | 381.696344 |
| attention | 33.827881 | 381.696344 |

## Files changed for this task

- `snn_nnunet/linear_attention_neuron.py`: new attention and optional neuron subclass.
- `snn_nnunet/recurrent_neuron.py`: minimal aggregation hook and factory.
- `snn_nnunet/recurrent_config.py`: additional model selector, old default preserved.
- `snn_nnunet/model.py`, `paper_unet.py`: factory routing at existing SiLIF construction sites.
- `snn_nnunet/paper_config.py`: per-block neuron selection and selective-only bidirectional validation.
- `snn_nnunet/profile_paper_unet.py`: additional readout parameter/continuous-MAC accounting.
- `snn_nnunet/profile_linear_attention.py`: scoped primary/auxiliary reports and benchmark.
- `unet_config.yaml`: flat experiment with independent per-block attention and fusion controls.
- `test_linear_attention.py`, `test_linear_attention_legacy.py`: new checks and pre-change regressions.
- This document and `analysis_results/linear_attention/*`: review and numerical evidence.
- Local plan: `docs/superpowers/plans/2026-11-05-silif-linear-attention.md` (docs/ is already ignored by the repository).

Existing Shared_SOMA changes in the dirty checkout were preserved. No preprocessing, mapping, loss, augmentation, metric, postprocessing, LR schedule, epoch count, baseline YAML or surrogate implementation was modified.
