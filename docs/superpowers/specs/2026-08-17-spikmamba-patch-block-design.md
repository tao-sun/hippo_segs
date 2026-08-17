# Patch-based SpikMamba Block Design

## Objective

Implement an autonomous patch-based SpikMamba experiment for the existing
spiking medical-image segmentation repository. The implementation must keep
the existing `SpikMamba2D`, `SS2D` baseline behavior, `ConvBlock`, and
`SNNBraTS` architectures unchanged.

The experiment consists of exactly three public modules:

```text
Spiking2DPatchEmbedding
SpikeMambaLayer
SpikMambaBlock
```

The block contains no SpikeSLA, attention, SS2D, CrossScan, reverse scan, or
four-direction scan.

## Deliberate adaptations from the paper

The design follows the structure of SpikMamba equations (8), (17)-(21), (10),
and (11), with three explicit user-selected adaptations:

1. Patch embedding uses a per-frame `Conv2d`, not `Conv3d`. A Conv2D with
   kernel and stride `patch_size` is equivalent to the paper's Conv3D with
   temporal kernel and stride one when frames are processed individually.
2. Every spike layer uses the repository's `PLIFNode`, not SpikingJelly's
   fixed-tau `MultiStepLIFNode`.
3. Mamba's sequence axis is the row-major spatial patch sequence within each
   frame, not the sequence of frames. Temporal state across frames remains in
   the stateful PLIF neurons and is driven by the absolute `time_step` already
   used by `SNNBraTS`.

These choices intentionally make the module an adaptation of SpikMamba for
this segmentation codebase rather than a bit-for-bit reproduction of the
classification implementation.

## Scope

### Included

- A self-contained production module in root-level `spikmamba.py`, matching
  this repository's flat model layout.
- A Conv2D spiking patch embedding with configurable patch size and default
  `patch_size=4`.
- Learnable spatial and temporal positional embeddings.
- A one-direction causal Mamba selective scan over row-major patch tokens.
- The Mamba default expansion and SSM parameterization used by the installed
  `mamba_ssm` implementation.
- Four independent `PLIFNode` instances: `SL_patch`, `SL_m1`, `SL_m2`, and
  `SL_ssm`.
- The two residual connections and the FFN from the SpikMamba block.
- Dependency injection for `selective_scan_fn` so CPU unit tests can use a
  differentiable reference implementation.
- Unit tests in `tests/test_spikmamba.py`.

### Excluded

- Integration into `SNNBraTS`, `ConvBlock`, an encoder, or a decoder.
- A dense segmentation reconstruction or upsampling adapter.
- Modification or removal of the existing `SpikMamba2D` baseline.
- SpikeSLA, Q/K/V projections, window partitioning, linear attention, or any
  other attention implementation.
- SS2D, VMamba, CrossScan, vertical scans, reverse scans, or direction merges.
- An additional recurrent state outside the existing `PLIFNode` state and the
  continuous hidden recurrence internal to the selective scan.

## Public interfaces

```python
class Spiking2DPatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        image_size: tuple[int, int] = (160, 192),
        patch_size: int = 4,
        max_time_steps: int = 256,
        init_tau: float = 2.0,
        device=None,
        dtype=None,
    ) -> None: ...

    def forward(self, x: torch.Tensor, time_step: int) -> torch.Tensor: ...


class SpikeMambaLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        init_tau: float = 2.0,
        selective_scan=None,
        conv_bias: bool = True,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        device=None,
        dtype=None,
    ) -> None: ...

    def forward(self, patches: torch.Tensor, time_step: int) -> torch.Tensor: ...


class SpikMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **mamba_kwargs,
    ) -> None: ...

    def forward(self, patches: torch.Tensor, time_step: int) -> torch.Tensor: ...
```

All public forwards accept one frame's absolute `time_step`. Patch embedding
accepts BCHW and returns `B D H_patch W_patch`. The Mamba layer and block both
accept and return `B D H_patch W_patch`.

## Spiking2DPatchEmbedding

For input `x` shaped `(B, C_in, H, W)`, require `H` and `W` to be divisible
by `patch_size`. With `p = patch_size`, calculate:

```text
Conv2D(kernel=p, stride=p): (B, C_in, H, W)
                         -> (B, D, H/p, W/p)
BatchNorm2D:             -> (B, D, H/p, W/p)
SL_patch(time_step):     -> (B, D, H/p, W/p)
spatial PE + temporal PE -> (B, D, H/p, W/p)
```

The convolution uses `bias=True`, matching the official patch projection, and
performs patch extraction and embedding projection in one operation. Each
output grid position is the learned embedding of one non-overlapping `p x p`
input region; no explicit `unfold` tensor is needed.

`SL_patch` is constructed exactly as:

```python
PLIFNode(
    init_tau=init_tau,
    surrogate_function=surrogate.ATan(),
    detach_reset=True,
)
```

The learnable spatial positional embedding is initialized for the base grid
derived from `image_size`. It is bilinearly interpolated with
`align_corners=False` if a valid runtime frame has a different patch-grid
size. A learnable temporal table of shape
`(max_time_steps, D)` supplies one channel vector for the absolute
`time_step`; the vector is broadcast across the patch grid. Spatial weights
use truncated-normal initialization with standard deviation `0.02`, matching
the official implementation's absolute positional embedding initialization.
The temporal table starts at zero, also matching the official implementation.

## SpikeMambaLayer

### Patch sequence

For `patches` shaped `(B, D, H_patch, W_patch)`, define
`L = H_patch * W_patch` and convert to row-major tokens:

```text
(B, D, H_patch, W_patch)
  -> flatten spatial axes
(B, D, L)
  -> transpose
(B, L, D)
```

The last position of a row is followed by the first position of the next row.
There is one forward causal scan only. No transposition of height and width,
flip, reverse pass, or multi-direction packing occurs.

### Linear, spikes, and causal Conv1D

Use the installed Mamba defaults unless explicitly overridden:

```text
d_inner = expand * D       (default expand=2)
dt_rank = ceil(D / 16)     when dt_rank="auto"
d_state = 16
d_conv = 4
```

The first path is:

```text
tokens                 (B, L, D)
Linear_m               (B, L, d_inner)
SL_m1                  (B, L, d_inner)
transpose              (B, d_inner, L)
depthwise Conv1D_m      (B, d_inner, L + d_conv - 1)
causal crop to L        (B, d_inner, L)
SL_m2                  (B, d_inner, L)
```

`Linear_m` uses `bias=False` by default. `Conv1D_m` has
`in_channels=out_channels=groups=d_inner`, `kernel_size=d_conv`,
`padding=d_conv-1`, and `bias=True` by default. Cropping the right tail
preserves length while ensuring an output position never depends on a later
patch token.

`SL_m1` and `SL_m2` are distinct `PLIFNode` instances constructed with the
same explicit configuration as `SL_patch`.

### Continuous selective SSM

Project `SL_m2` output with one bias-free linear layer from `d_inner` to
`dt_rank + 2*d_state`, then split into input-dependent low-rank Delta, B, and
C. Project low-rank Delta to `d_inner` with a trainable bias.

Preserve Mamba initialization and parameterization:

- initialize the Delta projection weight with scale
  `dt_rank**-0.5 * dt_scale`;
- initialize `softplus(delta_bias)` within `[dt_min, dt_max]` and respect
  `dt_init_floor`;
- store trainable `A_log` in float32, initialize rows with
  `log(1, ..., d_state)`, and use `A = -exp(A_log)`;
- initialize the float32 skip parameter `D` to ones;
- mark `A_log` and `D` as excluded from weight decay.

Call selective scan with:

```text
u:          (B, d_inner, L)
delta:      (B, d_inner, L)
A:          (d_inner, d_state)
B:          (B, d_state, L)
C:          (B, d_state, L)
D:          (d_inner,)
z:          None
delta_bias: (d_inner,)
delta_softplus=True
return_last_state=False
```

The SSM hidden-state recurrence remains continuous. There is no threshold,
surrogate activation, or PLIF inside the recurrence.

Convert scan output from `(B, d_inner, L)` to `(B, L, d_inner)`, apply a
linear output projection `d_inner -> D`, then apply the independent `SL_ssm`
PLIF. The output projection is the dimension-restoring part of the Mamba SSM
path. Finally perform equation (21)'s gate against the original token tensor:

```python
gated_tokens = ssm_spikes * original_tokens
```

Reshape the result back to `(B, D, H_patch, W_patch)`. The layer does not add
a residual; the enclosing block owns that operation.

## SpikMambaBlock

Attention is removed by defining `P_local = P`. The block implements exactly
two residual stages:

```text
P_local  = P
P_global = SpikeMambaLayer(P_local, time_step) + P_local
P_out    = FFN(P_global) + P_global
```

The FFN remains an internal attribute rather than a separate public class. It
operates independently on every row-major patch token and follows the active
official MLP path:

```text
LayerNorm(D)
Linear(D, mlp_ratio * D)
GELU
Dropout
Linear(mlp_ratio * D, D)
Dropout
```

The official repository defines LIF fields for its MLP but comments out their
use in the forward path. Therefore this design does not invent additional
spike layers in the FFN. All actual spike layers in this implementation use
the requested `PLIFNode`.

## Temporal-state behavior

The four PLIF modules own all spike membrane state. The implementation passes
the exact scalar `time_step` received from the caller to each relevant PLIF
once per forward. It does not maintain a duplicate frame counter and does not
reset state independently of existing `PLIFNode` semantics.

At `time_step=0`, the repository's PLIF implementation initializes its state.
At later time steps, it reuses its membrane state. Independent sequences are
therefore separated in the same way as all current `SNNBraTS` spiking blocks:
the caller begins the new sequence with `time_step=0`.

The selective scan's continuous hidden recurrence is local to one call and
runs over the spatial patch positions of that frame. It is not retained across
frames.

## Error handling

- Reject non-BCHW inputs with a `ValueError` containing the received shape.
- Reject non-positive `patch_size`, `max_time_steps`, `dim`, `d_state`,
  `d_conv`, or `expand` values during construction.
- Reject a base or runtime image size not divisible by `patch_size` rather
  than silently dropping boundary pixels.
- Reject channel mismatches before projection.
- Reject `time_step < 0` or `time_step >= max_time_steps` in patch embedding.
- Reject invalid `dt_rank` values and unsupported `dt_init` modes.
- If no scan is injected and `mamba_ssm` is unavailable, raise an actionable
  `ImportError` naming the missing dependency.

## File organization

```text
spikmamba.py
  Spiking2DPatchEmbedding
  SpikeMambaLayer
  SpikMambaBlock

tests/test_spikmamba.py
  patch embedding tests
  sequence-axis and SSM contract tests
  residual and FFN tests
  state and gradient tests
  forbidden-component tests
```

No import from `model.SpikMamba2D`, no edit to `model.py`, and no production
dependency on a test helper is permitted.

## Validation strategy

Development follows red-green-refactor TDD.

### Patch embedding tests

1. Construction uses Conv2D kernel and stride four by default, BatchNorm2D,
   and the repository's `PLIFNode`.
2. A `(2, 4, 16, 20)` input produces `(2, D, 4, 5)`.
3. A recording PLIF receives the caller's exact `time_step`.
4. The spatial positional embedding interpolates to another valid grid, and
   different temporal indices select different learned temporal vectors.
5. Invalid ranks, dimensions, divisibility, channels, or temporal indices
   raise focused errors.

### Mamba layer tests

1. `SL_m1`, `SL_m2`, and `SL_ssm` are three independent `PLIFNode` objects.
2. `Linear_m` expands `D -> 2D` by default.
3. Conv1D is depthwise, uses kernel four and padding three, and crops back to
   exactly `L`.
4. A known `(H_patch, W_patch)` tensor proves exact row-major token order.
5. A recording scan double proves that the scan sequence length is exactly
   `H_patch * W_patch` and verifies every scan argument and option.
6. Instrumented modules prove that no height/width transpose, flip, reverse
   scan, direction dimension, or direction merge is used.
7. The SSM output projection, `SL_ssm`, Hadamard gate, and reshape preserve the
   patch-grid shape.
8. A differentiable continuous reference scan produces finite forward output
   and finite nonzero input and parameter gradients.

### Block tests

1. The module tree contains no attention, Q/K/V, SpikeSLA, SS2D, CrossScan,
   or four-direction component.
2. A controlled Mamba-layer double verifies
   `P_global = SpikeMambaLayer(P) + P` exactly.
3. A controlled FFN verifies `P_out = FFN(P_global) + P_global` exactly.
4. Real patch embedding plus a real block with a CPU-compatible injected scan
   preserves all documented shapes and supports backward propagation.
5. Consecutive calls with increasing `time_step` retain distinct PLIF states;
   a new call at `time_step=0` follows the repository's reset convention.

### Completion report

Before completion, report:

- the mapping from equations (8), (17)-(21), (10), and (11) to code;
- the final architecture from frame input through FFN residual;
- explicit absence of SpikeSLA, attention, SS2D, CrossScan, and
  four-direction scanning;
- focused and full test results;
- a dummy forward/backward result;
- all intermediate tensor shapes for one concrete example.
