# SpikMamba2D in ConvBlock Design

## Objective

Replace the current continuous `SS2D` module with an autonomous spiking
`SpikMamba2D` module and execute it inside selected `ConvBlock` instances.
The new module must implement this ordered pipeline:

```text
Linear_m
  -> PLIF_1
  -> CrossScan
  -> depthwise Conv1D_scan
  -> PLIF_2
  -> delta/B/C projections
  -> Selective Scan
  -> Merge by aligned sum
  -> PLIF_SSM
  -> Hadamard product with F
  -> residual addition with F
```

Here `F` is the output of `ConvBlock.conv`. `SpikMamba2D` runs before the
existing `GroupNorm`, output PLIF neuron, and dropout.

## Scope

### Included

- Replace class `SS2D` with class `SpikMamba2D`.
- Rename the `ConvBlock` feature flag from `ss2d` to `spikMamba`.
- Set `spikMamba=False` by default.
- Enable `spikMamba=True` explicitly for `SNNBraTS.conv_block1`,
  `SNNBraTS.conv_block2`, and `SNNBraTS.conv_block3`.
- Keep SpikMamba disabled for decoder and classifier blocks unless explicitly
  enabled.
- Preserve the existing four-direction selective-scan parameterization and
  initialization.
- Use three independent local `PLIFNode` instances.
- Preserve the BCHW shape throughout the public module boundary.
- Preserve the user's unrelated local worktree changes.

### Excluded

- Changing encoder or decoder channel widths.
- Adding SpikMamba to decoder or classifier blocks.
- Sharing membrane state or learnable tau between the three internal PLIF
  neurons.
- Changing the existing `ConvBlock` normalization, output PLIF, or dropout
  semantics.
- Adding a fallback recurrence in production when `mamba_ssm` is unavailable.
- Providing compatibility with checkpoints created by the old `SS2D` class.

## Public interfaces

```python
class SpikMamba2D(nn.Module):
    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        dt_rank: int | str = "auto",
        conv_kernel_size: int = 3,
        init_tau: float = 2.0,
        selective_scan=None,
        device=None,
        dtype=None,
    ): ...

    def forward(self, F: torch.Tensor, time_step: int) -> torch.Tensor: ...
```

`SpikMamba2D.forward` accepts and returns `(B, C, H, W)`. The explicit
`selective_scan` argument supports a differentiable CPU test double. If it is
not provided, the module uses
`mamba_ssm.ops.selective_scan_interface.selective_scan_fn`.

`ConvBlock` changes from:

```python
ConvBlock(..., ss2d=True)
```

to:

```python
ConvBlock(..., spikMamba=False)
```

When enabled, `ConvBlock` owns one module in `self.spik_mamba`.

## Architecture and data flow

### ConvBlock placement

For input `x`, `ConvBlock.forward(x, time_step)` executes:

```text
F = conv(x)
if spikMamba:
    F = spik_mamba(F, time_step)
if normalization:
    F = norm(F)
if spiking:
    F = output_plif(F, time_step).spikes
else:
    F = output_plif(F, time_step)
if dropout > 0:
    F = dropout(F)
return F
```

### Linear_m and first PLIF

Save `residual = F` before any transformation. Convert `F` from BCHW to BHWC,
apply `nn.Linear(channels, channels)`, and convert the result back to BCHW.
The projection does not expand or contract the channel dimension.

Pass the projected tensor to an independent `PLIFNode` named `lif_1`, using
the same `time_step` received from `ConvBlock`. Use the emitted spikes as the
next stage's input.

### CrossScan

For spatial length `L = H * W`, construct:

1. row-major sequence;
2. row-major sequence of the transposed spatial axes;
3. reverse of sequence 1;
4. reverse of sequence 2.

The result has shape `(B, 4, C, L)`.

### Conv1D_scan and second PLIF

Pack CrossScan output as `(B, 4*C, L)`. Apply:

```python
nn.Conv1d(
    in_channels=4 * channels,
    out_channels=4 * channels,
    kernel_size=3,
    padding=1,
    groups=4 * channels,
)
```

The convolution preserves length and uses one independent filter per
direction-channel pair. Reshape the output back to `(B, 4, C, L)` and pass it
through the independent `lif_2` PLIF neuron with the same `time_step`. Use its
emitted spikes for the selective projections and scan input.

### Selective projections and scan

Retain four direction-specific projections. For `d_state=N` and
`dt_rank=R`, project the spiking scan tensor into:

```text
dt_low_rank: (B, 4, R, L)
B:           (B, 4, N, L)
C:           (B, 4, N, L)
```

Project `dt_low_rank` to `(B, 4, channels, L)`. Preserve the current
initialization:

- `dt_rank="auto"` means `ceil(channels / 16)`;
- initial `softplus(dt_bias)` lies in `[0.001, 0.1]` with floor `1e-4`;
- `A = -exp(A_logs)` with shape `(4*channels, d_state)`;
- `D` initializes to ones with shape `(4*channels)`;
- `A_logs` and `D` retain their no-weight-decay markers.

Call selective scan with packed `u` and `delta` shaped `(B, 4*C, L)`, and:

```text
z=None
delta_bias=dt_projs_bias
delta_softplus=True
return_last_state=False
```

Convert kernel inputs to float32 as in the current implementation.

### Merge, gate, and residual

Flip the two reverse outputs and transpose the two vertical outputs back into
common row-major order. Sum all four aligned tensors and reshape the result to
`(B, C, H, W)`; do not average them.

Pass the merged tensor through the independent `lif_ssm` PLIF neuron with the
same `time_step`. If `S` denotes the emitted spike tensor, return exactly:

```python
residual + S * residual
```

Both the Hadamard operand and the residual operand are the original `F`
captured at the module input.

## Temporal-state behavior

`lif_1`, `lif_2`, and `lif_ssm` are separate `PLIFNode` objects. Each has its
own membrane state and learnable tau. All receive the absolute `time_step`
already used by the surrounding SNN. At `time_step=0`, they follow the
repository's existing PLIF initialization/reset behavior; later time steps
continue their respective state sequences.

## SNNBraTS integration

The three encoder blocks are explicitly configured as:

```python
self.conv_block1 = ConvBlock(4, 32, padding=1, dropout=0.1,
                             spikMamba=True)
self.conv_block2 = ConvBlock(32, 64, padding=1, dropout=0.1,
                             spikMamba=True)
self.conv_block3 = ConvBlock(64, 128, padding=1, dropout=0.1,
                             spikMamba=True)
```

`SNNBraTS` passes the constructor's injected selective-scan function and SSM
configuration to all three enabled blocks so CPU tests can avoid the compiled
CUDA kernel. Existing decoder and classifier construction remains functionally
unchanged and uses the default `spikMamba=False`.

## Error handling

- A non-4D `F` raises `ValueError` and reports the received shape.
- A channel mismatch raises `ValueError` before Linear_m or selective scan.
- `conv_kernel_size` must be a positive odd integer so same-length symmetric
  padding is defined; invalid values raise `ValueError` during construction.
- If no scan function is injected and `mamba_ssm` is unavailable, forward
  raises an actionable `ImportError` naming `mamba_ssm`.
- Shape checks occur before reshaping packed directions, preventing obscure
  kernel errors.

## Compatibility

Replacing `SS2D` and renaming the flag intentionally changes parameter names
and tensor shapes. Old SS2D checkpoints must remain tied to the Git revision
that created them. New training runs start from fresh initialization unless a
separate weight migration is designed and validated.

## Validation strategy

Development follows red-green-refactor TDD.

### Unit tests

1. Construction creates `Linear_m`, three distinct `PLIFNode` objects, and a
   depthwise Conv1D with `in_channels=out_channels=groups=4*C`, kernel size 3,
   and padding 1.
2. Invalid convolution kernel sizes fail during construction.
3. CrossScan produces the exact row-major, transposed, reverse row-major, and
   reverse-transposed order for a small known tensor.
4. A recording scan double verifies every packed kernel argument shape and
   scan option.
5. Merge restores all four directions and sums rather than averages.
6. Instrumented PLIF test doubles verify that `lif_1`, `lif_2`, and `lif_ssm`
   each receive the same `time_step` exactly once per module forward.
7. A controlled scan/PLIF setup verifies the exact final expression
   `F + S*F`.
8. A differentiable CPU scan double verifies BCHW shape preservation and
   nonzero input gradients.

### ConvBlock and model integration tests

1. `ConvBlock` does not create a SpikMamba module by default.
2. `ConvBlock(..., spikMamba=True)` creates `SpikMamba2D` and preserves BCHW
   output shape.
3. An instrumented SpikMamba double verifies execution after `conv` and before
   normalization, the output PLIF, and dropout.
4. `SNNBraTS` owns `SpikMamba2D` in all three encoder blocks and none in the
   decoder/classifier blocks.
5. A small full-network CPU forward with an injected selective-scan double
   preserves the expected segmentation-logit shape.

### Production smoke test

In the pinned GPU environment containing `mamba_ssm`, run forward and backward
through `SpikMamba2D` and a small `SNNBraTS` batch, asserting finite outputs,
loss, and gradients. The CPU test double is not a production fallback.
