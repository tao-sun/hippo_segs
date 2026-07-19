# Selective SS2D design

## Objective

Replace the static recurrence inside `SS2D` with an input-selective Mamba-style
state-space recurrence while preserving the public `(B, C, H, W)` interface and
the surrounding `VSSBlock2D` architecture.

## Architecture

`SS2D` keeps four independent spatial directions: left-to-right,
right-to-left, top-to-bottom, and bottom-to-top. For each direction and spatial
position, learned projections of the current input estimate `delta`, `B`, and
`C`. The state transition parameter `A` remains learned but input-independent,
negative, and therefore stable. A learned `D` parameter supplies the direct
skip connection.

For each position, the recurrence is:

```text
A_bar = exp(delta(x_t) * A)
h_t = A_bar * h_(t-1) + delta(x_t) * B(x_t) * x_t
y_t = C(x_t) * h_t + D * x_t
```

Positive time steps are enforced with `softplus`. The four directional outputs
are averaged, matching the current aggregation behavior.

## Scope and compatibility

- The implementation uses ordinary PyTorch and introduces no external Mamba or
  CUDA-kernel dependency.
- Input and output remain BCHW tensors with identical shapes.
- `VSSBlock2D` and its gate, normalization, projections, residual, and dropout
  behavior remain unchanged.
- This is a minimal Mamba-style selective SSM, not a complete optimized VMamba
  reproduction with low-rank time-step projection or fused scan kernels.
- The stale `SimpleSSM2D` test is removed or updated because that class no
  longer exists.
- The `SSMBlock2D.ssm_module` injection issue is outside this change unless a
  test requires resolving it to exercise the requested block.

## Validation

Tests will verify that:

1. `SS2D` preserves BCHW shape and supports backpropagation.
2. Estimated `delta`, `B`, and `C` have the expected shapes and change when the
   input changes.
3. `VSSBlock2D` continues to preserve shape and backpropagate.
4. Invalid non-4D input still raises a clear error.

