# Faithful VM-UNet SS2D Integration Design

## Objective

Replace the repository's simplified selective `SS2D` and its duplicating VSS
wrapper with a faithful integration of the VSS/SS2D implementation used by
VM-UNet. Preserve the surrounding spiking U-Net, including its BCHW interface,
PLIF neurons, TBPTT/FPTT behavior, encoder/decoder, and insertion point after
`conv_block3`.

The resulting model is not a reproduction of the complete VM-UNet network. It
is the existing SNN U-Net with the official VM-UNet VSS/SS2D block integrated
at its deepest encoder stage.

## Upstream reference

The implementation source is the official VM-UNet repository:

- Paper: `https://arxiv.org/abs/2402.02491`
- Code: `https://github.com/JCruan519/VM-UNet`
- Reference module: `models/vmunet/vmamba.py`
- License: Apache-2.0

The port must retain an attribution comment and must not be described as a new
or independently derived SS2D implementation.

## Scope

### Included

- The official SS2D parameterization and initialization.
- `d_state=16` by default.
- Internal expansion `expand=2`.
- Automatic time-step rank `ceil(d_model / 16)`.
- Input projection into the data branch `x` and gate branch `z`.
- Depthwise 2D convolution followed by SiLU.
- Four-direction cross-scan construction.
- Direction-specific projections of low-rank time steps, `B`, and `C`.
- Direction-specific time-step projection and bias.
- Stable state transition `A = -exp(A_log)`.
- Direct skip parameter `D`.
- The original `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`.
- Four-direction output alignment and summation.
- Output LayerNorm, SiLU gate, output projection, and optional dropout.
- A VM-UNet-compatible VSS wrapper with pre-normalization, residual connection,
  and optional stochastic depth behavior if required by the copied wrapper.
- BCHW/BHWC conversion at the boundary between the SNN and VSS block.
- CPU structural tests and a CUDA integration smoke test.

### Excluded

- Replacing the full SNN U-Net with VM-UNet.
- Changing convolutional encoder or decoder stages.
- Moving the VSS block away from its current position after `conv_block3`.
- Changing PLIF behavior, temporal state handling, TBPTT, or FPTT.
- Replacing the official selective-scan kernel with the repository's Python
  recurrence.
- Retaining the earlier scalar-state constraint; faithful VM-UNet SS2D uses
  `d_state=16`.
- Claiming exact zero-order-hold discretization beyond what the official
  `selective_scan_fn` actually implements.

## Dependency contract

Faithfulness is defined against the dependency versions documented by the
official VM-UNet repository:

```text
Python 3.8/3.9-compatible environment
PyTorch 1.13.0 with CUDA 11.7/11.8
causal_conv1d==1.0.0
mamba_ssm==1.0.1
timm==0.4.12
einops
```

The current login environment uses Python 3.9 and PyTorch 2.1.2+cu121, does not
have `mamba_ssm` installed, and exposes no CUDA device. The official-version
environment must therefore be created or selected for SLURM GPU execution
rather than silently installing incompatible packages into the current shared
environment.

Import failure must raise an actionable error naming the missing package and
required version. It must not be swallowed with a bare `except` that leaves
`selective_scan_fn` undefined.

## Architecture

### End-to-end placement

```text
input window (B, k, 4, H, W)
  -> ConvBlock1 -> pool1
  -> ConvBlock2 -> pool2
  -> ConvBlock3 (B, 128, H/4, W/4)
  -> SSMBlock2D
       -> BCHW-to-BHWC adapter
       -> VM-UNet VSS block
       -> BHWC-to-BCHW adapter
       -> existing GroupNorm
       -> existing PLIF neuron
  -> max pool
  -> existing decoder and skip connections
  -> segmentation logits
```

`SSMBlock2D` continues to consume and produce `(B, C, H, W)` and continues to
apply its normalization and PLIF neuron after the continuous VSS result.

### VSS wrapper

The current repository wrapper contains input projection, depthwise
convolution, gate projection, SS2D normalization, and output projection. The
official VM-UNet `SS2D` already owns these transformations. Leaving both sets
would duplicate the main branch and gate and would not be faithful.

The wrapper must therefore become:

```text
residual = x
x = LayerNorm(x)
x = SS2D(x)
x = residual + DropPath(x)
```

where tensors inside the wrapper use BHWC layout, matching the upstream
implementation. With drop-path probability zero, `DropPath` is an identity.

### SS2D input branch

For input `x` shaped `(B, H, W, d_model)`:

```text
xz = in_proj(x)
x, z = split(xz)
x = BCHW(x)
x = SiLU(depthwise_conv2d(x))
```

The inner width is:

```text
d_inner = expand * d_model
```

with default `expand=2`.

### Four-direction cross scan

For spatial length `L = H * W`, construct four sequences:

1. row-major flattened image;
2. row-major flattened transposed image;
3. reverse of sequence 1;
4. reverse of sequence 2.

Their packed shape is `(B, 4, d_inner, L)`. This is the upstream VMamba-style
cross scan and is different from maintaining an independent recurrent state
for each image row or column.

### Selective parameters

Each direction has its own input projection producing:

```text
dt_low_rank: (B, 4, dt_rank, L)
B:           (B, 4, d_state, L)
C:           (B, 4, d_state, L)
```

The direction-specific time-step projection expands `dt_low_rank` to
`d_inner`. Its bias is initialized so that `softplus(dt)` lies between
`dt_min=0.001` and `dt_max=0.1`, with floor `1e-4`.

The state and skip parameters are:

```text
A = -exp(A_logs): (4 * d_inner, d_state)
D:                (4 * d_inner)
```

`A_logs` uses the upstream S4D real initialization based on
`log(1), ..., log(d_state)`. `D` initializes to one. Both parameters retain the
upstream no-weight-decay markers.

### Selective scan and output alignment

Call the official kernel with:

```text
u=xs
delta=dts
A=As
B=Bs
C=Cs
D=Ds
z=None
delta_bias=dt_projs_bias
delta_softplus=True
return_last_state=False
```

The four outputs are restored into common row-major spatial order by flipping
the two reverse sequences and transposing the two width-height sequences.
They are summed, not averaged, matching the attached official implementation.

The combined tensor then follows:

```text
y = LayerNorm(y)
y = y * SiLU(z)
y = out_proj(y)
y = dropout(y), when configured
```

and returns `(B, H, W, d_model)`.

## Public interfaces

The repository-facing constructors remain simple:

```python
SS2D(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 3,
    expand: int = 2,
    dt_rank: int | str = "auto",
    dropout: float = 0.0,
)

VSSBlock2D(
    channels: int,
    d_state: int = 16,
    dropout: float = 0.0,
    drop_path: float = 0.0,
)
```

`VSSBlock2D.forward` continues to accept and return BCHW tensors for local
compatibility; it performs layout conversion internally around the upstream
BHWC implementation. `SSMBlock2D.forward(x, time_step)` remains unchanged.

## Precision and device behavior

The scan inputs, time steps, `B`, `C`, `A`, `D`, and time-step bias follow the
upstream conversion to float32 before calling `selective_scan_fn`. The returned
scan is expected to be float32 before output normalization and projection.

No CPU execution claim is made for the compiled CUDA kernel. Pure structural
tests may replace the imported scan function with a shape-compatible test
double. The real integration test runs on a CUDA compute node with the pinned
environment.

## Error handling

- Non-4D input to `VSSBlock2D` raises a `ValueError` containing the received
  shape.
- Channel mismatch produces a clear error before the selective-scan call.
- Missing `mamba_ssm==1.0.1` produces an `ImportError` with installation and
  environment guidance.
- Kernel build or CUDA incompatibility is treated as an environment failure,
  not silently replaced with the simplified Python scan.
- Non-finite scan output fails the CUDA smoke test.

## Compatibility and checkpoint policy

The new parameter names and shapes are intentionally incompatible with
checkpoints produced by the simplified scalar-state `SS2D`. Old checkpoints
must be evaluated with their originating Git revision. New VM-UNet-SS2D runs
must start from a fresh initialization unless a separately validated upstream
weight-mapping procedure is introduced.

Each trained checkpoint must record:

- the exact Git commit;
- `mamba_ssm`, PyTorch, and CUDA versions;
- `d_state`, `expand`, `dt_rank`, and convolution width;
- confirmation that the official kernel, rather than a test double, was used.

## Validation strategy

### CPU structural tests

1. `dt_init` produces positive softplus time steps inside the configured
   interval at initialization.
2. `A_log_init` has shape `(4 * d_inner, d_state)` and maps to strictly
   negative `A`.
3. `D_init` has shape `(4 * d_inner)` and initializes to ones.
4. Cross-scan constructs row-major, transposed, and two reverse sequences in
   the upstream order.
5. Output alignment exactly reverses the cross-scan transformations.
6. A shape-compatible scan test double verifies all kernel argument shapes,
   `delta_softplus=True`, and absence of an internal `z` argument.
7. `SS2D` preserves BHWC shape and backpropagates through projections.
8. `VSSBlock2D` preserves BCHW shape and backpropagates through the layout
   adapter and residual.
9. `SSMBlock2D` continues to preserve BCHW shape and apply its PLIF neuron.

### CUDA integration tests

1. Import `selective_scan_fn` from `mamba_ssm==1.0.1` on the allocated GPU.
2. Run SS2D forward and backward on a small tensor.
3. Compare the compiled kernel with `selective_scan_ref` for the same packed
   inputs using documented floating-point tolerances.
4. Run one `SNNBraTS` forward/backward batch and assert finite loss and
   gradients.
5. Run a one-subject, one-epoch BraTS smoke experiment before scheduling a
   complete experiment.

## Documentation language

The supervisor-facing description must be:

> SNN U-Net with a VSS/SS2D block ported from the official VM-UNet
> implementation, using `mamba_ssm==1.0.1` selective scan.

It must not claim that the complete network is VM-UNet or that the official
kernel implements the full theoretical zero-order-hold expression shown in
the S6 pseudocode.

