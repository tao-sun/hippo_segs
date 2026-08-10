# SpikMamba2D ConvBlock Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `SS2D` with the approved stateful `SpikMamba2D` pipeline and run it before normalization in all three `SNNBraTS` encoder `ConvBlock` instances.

**Architecture:** `SpikMamba2D` remains a standalone BCHW module. It projects the convolution feature `F`, spikes it, cross-scans it, applies direction-channel depthwise Conv1D and a second PLIF, runs the existing four-direction selective scan, merges by aligned sum, applies a third PLIF, and returns `F + spike_ssm * F`. `ConvBlock` owns it only when the new `spikMamba` flag is explicitly true.

**Tech Stack:** Python 3.9, PyTorch, local `PLIFNode`, `mamba_ssm` selective scan in production, `unittest` tests executed through pytest.

## Global Constraints

- Follow the approved design in `docs/superpowers/specs/2026-08-10-spikmamba2d-convblock-design.md`.
- Keep the public tensor contract BCHW to BCHW.
- Use `nn.Linear(C, C)` for `Linear_m`; do not expand channels.
- Use three separate `PLIFNode` instances with `init_tau=2.0` by default.
- Use depthwise `Conv1d(4*C, 4*C, kernel_size=3, padding=1, groups=4*C)`.
- Merge the four aligned scan directions by summation, never averaging.
- Compute the final output exactly as `F + lif_ssm(scan_sum) * F`.
- Rename the feature flag to `spikMamba` and set its default to `False`.
- Enable SpikMamba explicitly on `SNNBraTS.conv_block1`, `conv_block2`, and `conv_block3` only.
- Preserve the user's pre-existing modifications in `model.py`, `experiments_snn_fptt.yaml`, `snn_fptt.job`, and `__pycache__/model.cpython-39.pyc`.
- Do not stage or commit implementation files while the overlapping pre-existing `model.py` edit remains in the worktree; use diff checkpoints instead so user-owned changes are not silently absorbed into an implementation commit.
- Use `.venv/bin/python -m pytest`; bare `pytest` and `/usr/bin/python -m pytest` are unavailable.
- The current test baseline fails during collection because `tests/test_ssm_module.py` imports the removed `SSMBlock2D`; replace those stale tests with the approved SpikMamba2D contracts.

---

## File map

- Modify `model.py`: define `SpikMamba2D`, integrate it in `ConvBlock`, and enable it in the three `SNNBraTS` encoder blocks.
- Modify `tests/test_ssm_module.py`: replace obsolete SS2D/SSMBlock2D tests with focused SpikMamba2D, ConvBlock, and SNNBraTS tests plus CPU selective-scan doubles.
- No production fallback module or new dependency file is created.

### Task 1: Establish the SpikMamba2D constructor and validation

**Files:**
- Modify: `tests/test_ssm_module.py`
- Modify: `model.py:52-110`
- Modify: `model.py:136-221`

**Interfaces:**
- Produces: `SpikMamba2D(channels, d_state=16, dt_rank="auto", conv_kernel_size=3, init_tau=2.0, selective_scan=None, device=None, dtype=None)`.
- Produces: attributes `linear_m`, `lif_1`, `scan_conv1d`, `lif_2`, `lif_ssm`, `x_proj_weight`, `dt_projs_weight`, `dt_projs_bias`, `A_logs`, and `Ds`.
- Consumes later: all later tasks instantiate the class and inject a CPU scan double through `selective_scan`.

- [ ] **Step 1: Replace stale imports and write failing constructor tests**

Replace the test import with:

```python
from model import ConvBlock, SNNBraTS, SpikMamba2D
from spike_neurons import PLIFNode
```

Retain the differentiable `selective_scan_test_double` and `RecordingScan`, then replace `SS2DTest` and the obsolete `SSMBlock2D` tests with this first test class:

```python
class SpikMamba2DConstructionTest(unittest.TestCase):
    def test_constructor_builds_independent_spiking_scan_stages(self):
        module = SpikMamba2D(
            channels=8,
            d_state=4,
            selective_scan=selective_scan_test_double,
        )

        self.assertEqual(module.channels, 8)
        self.assertEqual(module.dt_rank, 1)
        self.assertEqual(module.linear_m.in_features, 8)
        self.assertEqual(module.linear_m.out_features, 8)
        self.assertIsInstance(module.lif_1, PLIFNode)
        self.assertIsInstance(module.lif_2, PLIFNode)
        self.assertIsInstance(module.lif_ssm, PLIFNode)
        self.assertIsNot(module.lif_1, module.lif_2)
        self.assertIsNot(module.lif_1, module.lif_ssm)
        self.assertIsNot(module.lif_2, module.lif_ssm)
        self.assertEqual(module.scan_conv1d.in_channels, 32)
        self.assertEqual(module.scan_conv1d.out_channels, 32)
        self.assertEqual(module.scan_conv1d.groups, 32)
        self.assertEqual(module.scan_conv1d.kernel_size, (3,))
        self.assertEqual(module.scan_conv1d.padding, (1,))
        self.assertEqual(module.A_logs.shape, (32, 4))
        self.assertEqual(module.Ds.shape, (32,))
        self.assertIs(module.selective_scan, selective_scan_test_double)

    def test_constructor_rejects_kernel_without_symmetric_same_padding(self):
        for kernel_size in (0, 2, -1):
            with self.subTest(kernel_size=kernel_size):
                with self.assertRaisesRegex(
                    ValueError, "positive odd integer"
                ):
                    SpikMamba2D(
                        channels=4,
                        conv_kernel_size=kernel_size,
                        selective_scan=selective_scan_test_double,
                    )

    def test_selective_parameters_keep_stable_initialization(self):
        module = SpikMamba2D(
            channels=8,
            d_state=4,
            selective_scan=selective_scan_test_double,
        )

        initialized_dt = F.softplus(module.dt_projs_bias)
        self.assertGreaterEqual(float(initialized_dt.min()), 0.001 - 1e-6)
        self.assertLessEqual(float(initialized_dt.max()), 0.1 + 1e-6)
        self.assertTrue(torch.all(-torch.exp(module.A_logs) < 0))
        self.assertTrue(torch.equal(module.Ds, torch.ones_like(module.Ds)))
```

- [ ] **Step 2: Run the constructor tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_ssm_module.py::SpikMamba2DConstructionTest
```

Expected: collection fails because `SpikMamba2D` does not exist yet. This is the intended failure, replacing the unrelated stale `SSMBlock2D` import error.

- [ ] **Step 3: Rename SS2D and implement only constructor-owned stages**

Rename `SS2D` to `SpikMamba2D`. Extend its constructor signature with `conv_kernel_size`, `init_tau`, and an explicit `selective_scan` argument. Validate the kernel before allocating layers:

```python
if not isinstance(conv_kernel_size, int) or conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
    raise ValueError("conv_kernel_size must be a positive odd integer")
```

Add:

```python
self.linear_m = nn.Linear(self.channels, self.channels, **factory_kwargs)
self.lif_1 = PLIFNode(
    init_tau=init_tau,
    surrogate_function=surrogate.ATan(),
    detach_reset=True,
)
self.scan_conv1d = nn.Conv1d(
    4 * self.channels,
    4 * self.channels,
    kernel_size=conv_kernel_size,
    padding=conv_kernel_size // 2,
    groups=4 * self.channels,
    device=device,
    dtype=dtype,
)
self.lif_2 = PLIFNode(
    init_tau=init_tau,
    surrogate_function=surrogate.ATan(),
    detach_reset=True,
)
self.lif_ssm = PLIFNode(
    init_tau=init_tau,
    surrogate_function=surrogate.ATan(),
    detach_reset=True,
)
self.selective_scan = (
    selective_scan if selective_scan is not None else selective_scan_fn
)
```

Keep the current `dt_init`, `A_log_init`, `D_init`, and direction-specific projection initialization unchanged.

- [ ] **Step 4: Run constructor tests and verify GREEN**

Run the command from Step 2. Expected: all three constructor tests pass with no warnings.

- [ ] **Step 5: Inspect the focused diff checkpoint**

Run:

```bash
git diff --check -- model.py tests/test_ssm_module.py
git diff -- model.py tests/test_ssm_module.py
```

Confirm the pre-existing ConvBlock comments and unrelated files have not been lost. Do not stage or commit because `model.py` already contains overlapping user edits.

### Task 2: Implement spiking cross-scan preprocessing and selective merge

**Files:**
- Modify: `tests/test_ssm_module.py`
- Modify: `model.py:222-340`

**Interfaces:**
- Consumes: constructor attributes from Task 1.
- Produces: `_prepare_scans(F, time_step) -> Tensor[B,4,C,L]`.
- Produces: `_run_selective_scan(scans) -> Tensor[B,4,C,L]`.
- Produces: `_merge_scan_outputs(out_y, height, width) -> Tensor[B,C,H,W]`.

- [ ] **Step 1: Add deterministic PLIF test doubles and failing preprocessing test**

Add to the test utilities:

```python
class PassthroughPLIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_steps = []

    def forward(self, x, time_step):
        self.time_steps.append(time_step)
        return x, x
```

Add the behavior test with a hand-derived expected tensor:

```python
class SpikMamba2DScanTest(unittest.TestCase):
    def test_prepare_scans_preserves_direction_order_and_filters_independently(self):
        module = SpikMamba2D(
            channels=1,
            d_state=1,
            selective_scan=selective_scan_test_double,
        )
        module.linear_m = nn.Identity()
        module.lif_1 = PassthroughPLIF()
        module.lif_2 = PassthroughPLIF()
        with torch.no_grad():
            module.scan_conv1d.weight.zero_()
            module.scan_conv1d.bias.zero_()
            module.scan_conv1d.weight[:, 0, 1] = torch.tensor([1., 2., 3., 4.])
        F_in = torch.tensor([[[[1., 2., 3.], [4., 5., 6.]]]])

        scans = module._prepare_scans(F_in, time_step=7)

        expected = torch.tensor([[[[1., 2., 3., 4., 5., 6.]],
                                  [[2., 8., 4., 10., 6., 12.]],
                                  [[18., 15., 12., 9., 6., 3.]],
                                  [[24., 12., 20., 8., 16., 4.]]]])
        self.assertTrue(torch.equal(scans, expected))
        self.assertEqual(module.lif_1.time_steps, [7])
        self.assertEqual(module.lif_2.time_steps, [7])
```

- [ ] **Step 2: Run the preprocessing test and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_ssm_module.py::SpikMamba2DScanTest::test_prepare_scans_preserves_direction_order_and_filters_independently
```

Expected: failure because `_prepare_scans` does not exist.

- [ ] **Step 3: Implement Linear_m, PLIF_1, CrossScan, Conv1D_scan, and PLIF_2**

Keep `_cross_scan` unchanged. Add:

```python
def _prepare_scans(self, F: torch.Tensor, time_step: int) -> torch.Tensor:
    batch, channels, height, width = F.shape
    projected = self.linear_m(F.permute(0, 2, 3, 1))
    projected = projected.permute(0, 3, 1, 2).contiguous()
    projected, _ = self.lif_1(projected, time_step)
    scans = self._cross_scan(projected)
    length = height * width
    scans = self.scan_conv1d(scans.view(batch, 4 * channels, length))
    scans = scans.view(batch, 4, channels, length)
    scans, _ = self.lif_2(scans, time_step)
    return scans
```

- [ ] **Step 4: Run the preprocessing test and verify GREEN**

Run the command from Step 2. Expected: pass.

- [ ] **Step 5: Write failing selective-kernel and merge tests**

Add:

```python
    def test_selective_scan_uses_packed_four_direction_contract(self):
        recorder = RecordingScan()
        module = SpikMamba2D(channels=4, d_state=3, selective_scan=recorder)
        scans = torch.randn(2, 4, 4, 6)

        raw = module._run_selective_scan(scans)

        self.assertEqual(raw.shape, (2, 4, 4, 6))
        self.assertEqual(recorder.call["u"], torch.Size((2, 16, 6)))
        self.assertEqual(recorder.call["delta"], torch.Size((2, 16, 6)))
        self.assertEqual(recorder.call["A"], torch.Size((16, 3)))
        self.assertEqual(recorder.call["B"], torch.Size((2, 4, 3, 6)))
        self.assertEqual(recorder.call["C"], torch.Size((2, 4, 3, 6)))
        self.assertEqual(recorder.call["D"], torch.Size((16,)))
        self.assertEqual(recorder.call["delta_bias"], torch.Size((16,)))
        self.assertIsNone(recorder.call["z"])
        self.assertTrue(recorder.call["delta_softplus"])
        self.assertFalse(recorder.call["return_last_state"])

    def test_merge_restores_directions_and_sums_them(self):
        module = SpikMamba2D(
            channels=1,
            d_state=1,
            selective_scan=selective_scan_test_double,
        )
        raw = torch.tensor([[[[1., 2., 3., 4., 5., 6.]],
                             [[1., 4., 2., 5., 3., 6.]],
                             [[6., 5., 4., 3., 2., 1.]],
                             [[6., 3., 5., 2., 4., 1.]]]])

        merged = module._merge_scan_outputs(raw, height=2, width=3)

        expected = torch.tensor([[[[4., 8., 12.], [16., 20., 24.]]]])
        self.assertTrue(torch.equal(merged, expected))
```

- [ ] **Step 6: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_ssm_module.py::SpikMamba2DScanTest
```

Expected: the preprocessing test passes; the two new tests fail because `_run_selective_scan` and `_merge_scan_outputs` do not exist.

- [ ] **Step 7: Split the old forward_core into selective scan and merge helpers**

Change the old `forward_core` so `_run_selective_scan` consumes already-spiking `(B,4,C,L)` scans, performs the direction-specific projections, calls the kernel, and returns raw `(B,4,C,L)` output. Before the call:

```python
if self.selective_scan is None:
    raise ImportError(
        "SpikMamba2D requires mamba_ssm or an injected selective_scan function"
    ) from _MAMBA_IMPORT_ERROR
```

Keep float32 packing and the exact kernel keyword arguments. Add:

```python
def _merge_scan_outputs(self, out_y, height, width):
    aligned = self._align_scan_outputs(out_y, height, width)
    batch, _, channels, _ = out_y.shape
    return sum(aligned).view(batch, channels, height, width)
```

- [ ] **Step 8: Run all scan tests and verify GREEN**

Run the command from Step 6. Expected: all scan tests pass.

- [ ] **Step 9: Inspect the focused diff checkpoint**

Run `git diff --check -- model.py tests/test_ssm_module.py`, inspect the diff, and leave it unstaged.

### Task 3: Implement PLIF_SSM gating, residual, errors, and gradients

**Files:**
- Modify: `tests/test_ssm_module.py`
- Modify: `model.py:330-355`

**Interfaces:**
- Consumes: `_prepare_scans`, `_run_selective_scan`, and `_merge_scan_outputs` from Task 2.
- Produces: `SpikMamba2D.forward(F, time_step) -> Tensor[B,C,H,W]`.

- [ ] **Step 1: Add scan doubles and failing forward contract tests**

Add:

```python
def ones_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
              delta_softplus=False, return_last_state=False):
    return torch.ones_like(u)


class SpikMamba2DForwardTest(unittest.TestCase):
    def test_forward_uses_original_F_for_hadamard_and_residual(self):
        module = SpikMamba2D(channels=2, d_state=1, selective_scan=ones_scan)
        module.lif_ssm = PassthroughPLIF()
        F_in = torch.tensor([[[[1., 2.]], [[3., 4.]]]])

        output = module(F_in, time_step=5)

        self.assertTrue(torch.equal(output, 5 * F_in))
        self.assertEqual(module.lif_ssm.time_steps, [5])

    def test_forward_passes_same_time_step_to_all_three_plifs(self):
        module = SpikMamba2D(channels=2, d_state=1, selective_scan=ones_scan)
        module.lif_1 = PassthroughPLIF()
        module.lif_2 = PassthroughPLIF()
        module.lif_ssm = PassthroughPLIF()

        module(torch.randn(1, 2, 2, 3), time_step=11)

        self.assertEqual(module.lif_1.time_steps, [11])
        self.assertEqual(module.lif_2.time_steps, [11])
        self.assertEqual(module.lif_ssm.time_steps, [11])

    def test_forward_validates_layout_channels_and_scan_dependency(self):
        module = SpikMamba2D(channels=2, d_state=1, selective_scan=ones_scan)
        with self.assertRaisesRegex(ValueError, "BCHW"):
            module(torch.randn(1, 2, 4), time_step=0)
        with self.assertRaisesRegex(ValueError, "expects 2 channels"):
            module(torch.randn(1, 3, 2, 2), time_step=0)
        module.selective_scan = None
        with self.assertRaisesRegex(ImportError, "mamba_ssm"):
            module(torch.randn(1, 2, 2, 2), time_step=0)

    def test_forward_preserves_shape_and_backpropagates(self):
        torch.manual_seed(0)
        module = SpikMamba2D(
            channels=4,
            d_state=2,
            selective_scan=selective_scan_test_double,
        )
        F_in = torch.randn(2, 4, 3, 4, requires_grad=True)

        output = module(F_in, time_step=0)
        output.square().mean().backward()

        self.assertEqual(output.shape, F_in.shape)
        self.assertIsNotNone(F_in.grad)
        self.assertGreater(float(F_in.grad.abs().sum()), 0.0)
        self.assertIsNotNone(module.linear_m.weight.grad)
        self.assertIsNotNone(module.scan_conv1d.weight.grad)
```

- [ ] **Step 2: Run forward tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_ssm_module.py::SpikMamba2DForwardTest
```

Expected: failures because the old `forward` does not accept/use `time_step`, internal PLIF stages, or final gating.

- [ ] **Step 3: Implement the approved forward pipeline**

Replace the old `forward` with:

```python
def forward(self, F: torch.Tensor, time_step: int) -> torch.Tensor:
    if F.ndim != 4:
        raise ValueError(
            f"SpikMamba2D expects BCHW input, got shape={tuple(F.shape)}"
        )
    batch, channels, height, width = F.shape
    if channels != self.channels:
        raise ValueError(
            f"SpikMamba2D expects {self.channels} channels, got {channels}"
        )
    if self.selective_scan is None:
        raise ImportError(
            "SpikMamba2D requires mamba_ssm or an injected selective_scan function"
        ) from _MAMBA_IMPORT_ERROR

    residual = F
    scans = self._prepare_scans(F, time_step)
    raw = self._run_selective_scan(scans)
    merged = self._merge_scan_outputs(raw, height, width)
    spikes, _ = self.lif_ssm(merged, time_step)
    return residual + spikes * residual
```

Keep dependency validation in `forward`; `_run_selective_scan` may retain the same guard for safe direct use.

- [ ] **Step 4: Run forward tests and verify GREEN**

Run the command from Step 2. Expected: all forward tests pass.

- [ ] **Step 5: Run all SpikMamba unit tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_ssm_module.py -k 'SpikMamba2D'
```

Expected: all construction, scan, and forward tests pass.

- [ ] **Step 6: Inspect the focused diff checkpoint**

Run `git diff --check -- model.py tests/test_ssm_module.py`, inspect the diff, and leave it unstaged.

### Task 4: Integrate SpikMamba2D into ConvBlock and the three encoder stages

**Files:**
- Modify: `tests/test_ssm_module.py`
- Modify: `model.py:52-108`
- Modify: `model.py:350-390`

**Interfaces:**
- Consumes: `SpikMamba2D.forward(F, time_step)` from Task 3.
- Produces: `ConvBlock(..., spikMamba=False, selective_scan=None, ssm_d_state=16, ssm_dt_rank="auto")`.
- Produces: `ConvBlock.spik_mamba` only when enabled.
- Produces: `SNNBraTS` with SpikMamba enabled on all three encoder blocks.

- [ ] **Step 1: Add deterministic block doubles and failing ConvBlock ordering tests**

Add:

```python
class AddOneSpikMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, F, time_step):
        self.calls.append((F.detach().clone(), time_step))
        return F + 1


class TimesTwo(nn.Module):
    def forward(self, x):
        return 2 * x


class AddThreePLIF(nn.Module):
    def forward(self, x, time_step):
        return x + 3, x


class ConvBlockSpikMambaIntegrationTest(unittest.TestCase):
    def test_spikmamba_is_opt_in(self):
        block = ConvBlock(1, 1, kernel_size=1, dropout=0.0)
        self.assertFalse(block.spikMamba)
        self.assertFalse(hasattr(block, "spik_mamba"))

    def test_spikmamba_runs_after_conv_and_before_norm_and_output_plif(self):
        block = ConvBlock(
            1,
            1,
            kernel_size=1,
            dropout=0.0,
            spikMamba=True,
            selective_scan=ones_scan,
        )
        with torch.no_grad():
            block.conv.weight.fill_(1.0)
        recorder = AddOneSpikMamba()
        block.spik_mamba = recorder
        block.norm = TimesTwo()
        block.spike_neurons = AddThreePLIF()

        output = block(torch.ones(1, 1, 1, 1), time_step=9)

        self.assertTrue(torch.equal(output, torch.tensor([[[[7.]]]])))
        self.assertTrue(torch.equal(recorder.calls[0][0], torch.ones(1, 1, 1, 1)))
        self.assertEqual(recorder.calls[0][1], 9)
```

- [ ] **Step 2: Run ConvBlock integration tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_ssm_module.py::ConvBlockSpikMambaIntegrationTest
```

Expected: constructor errors because `spikMamba` is not implemented and default state is absent.

- [ ] **Step 3: Implement ConvBlock opt-in construction and forward ordering**

Change the constructor tail to:

```python
normalization=True,
spiking=True,
spikMamba=False,
selective_scan=None,
ssm_d_state=16,
ssm_dt_rank="auto",
```

Store `self.spikMamba = bool(spikMamba)`. When enabled, construct:

```python
self.spik_mamba = SpikMamba2D(
    channels=out_channels,
    d_state=ssm_d_state,
    dt_rank=ssm_dt_rank,
    init_tau=init_tau,
    selective_scan=selective_scan,
)
```

In `forward`, immediately after `out = self.conv(x)`, add:

```python
if self.spikMamba:
    out = self.spik_mamba(out, time_step)
```

Remove the obsolete commented SS2D construction/forward block while preserving all unrelated user edits.

- [ ] **Step 4: Run ConvBlock integration tests and verify GREEN**

Run the command from Step 2. Expected: both tests pass.

- [ ] **Step 5: Add failing SNNBraTS placement and end-to-end tests**

Add:

```python
class SNNBraTSSpikMambaIntegrationTest(unittest.TestCase):
    def test_all_three_encoder_blocks_use_injected_spikmamba(self):
        recorder = RecordingScan()
        model = SNNBraTS(
            out_channels=4,
            selective_scan=recorder,
            ssm_d_state=2,
        )

        for block in (model.conv_block1, model.conv_block2, model.conv_block3):
            self.assertTrue(block.spikMamba)
            self.assertIsInstance(block.spik_mamba, SpikMamba2D)
            self.assertIs(block.spik_mamba.selective_scan, recorder)
            self.assertEqual(block.spik_mamba.d_state, 2)
        for block in (
            model.deconv1_conv,
            model.concat1_conv,
            model.deconv2_conv,
            model.concat2_conv,
            model.deconv3_conv,
            model.class_conv,
        ):
            self.assertFalse(block.spikMamba)
            self.assertFalse(hasattr(block, "spik_mamba"))

    def test_small_network_forward_preserves_segmentation_shape(self):
        model = SNNBraTS(
            out_channels=4,
            selective_scan=ones_scan,
            ssm_d_state=2,
        ).eval()
        x = torch.randn(1, 1, 4, 16, 16)

        with torch.no_grad():
            output = model(x, t0=0)

        self.assertEqual(output.shape, (1, 4, 1, 16, 16))
```

- [ ] **Step 6: Run SNNBraTS integration tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_ssm_module.py::SNNBraTSSpikMambaIntegrationTest
```

Expected: failures because the model still uses the removed `ss2d` argument and does not enable/pass the injected scan to all three encoder blocks.

- [ ] **Step 7: Enable all three encoder blocks and remove old flag call sites**

Construct `conv_block1`, `conv_block2`, and `conv_block3` with:

```python
spikMamba=True,
selective_scan=selective_scan,
ssm_d_state=ssm_d_state,
ssm_dt_rank=ssm_dt_rank,
```

Remove every `ss2d=False` argument from decoder/classifier construction because the new default is false. Search with:

```bash
rg -n "ss2d|SS2D" model.py tests/test_ssm_module.py
```

Expected: no old production symbol or flag remains.

- [ ] **Step 8: Run integration tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_ssm_module.py::ConvBlockSpikMambaIntegrationTest tests/test_ssm_module.py::SNNBraTSSpikMambaIntegrationTest
```

Expected: all four integration tests pass.

- [ ] **Step 9: Inspect the focused diff checkpoint**

Run `git diff --check -- model.py tests/test_ssm_module.py`, inspect the complete implementation diff, and leave it unstaged.

### Task 5: Regression and production-readiness verification

**Files:**
- Verify: `model.py`
- Verify: `tests/test_ssm_module.py`
- Preserve: all unrelated dirty files

**Interfaces:**
- Consumes: completed SpikMamba2D and integrations from Tasks 1-4.
- Produces: evidence that focused and repository test suites pass in the local CPU environment, plus a clearly identified CUDA-only smoke-test handoff if no GPU is available.

- [ ] **Step 1: Run the complete focused suite**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_ssm_module.py
```

Expected: all SpikMamba2D, ConvBlock, and SNNBraTS tests pass with no collection errors.

- [ ] **Step 2: Run the repository test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all repository tests pass. If a pre-existing unrelated test fails, capture the exact failure and prove `tests/test_ssm_module.py` remains green; do not change unrelated behavior without authorization.

- [ ] **Step 3: Run static and stale-symbol checks**

Run:

```bash
.venv/bin/python -m py_compile model.py tests/test_ssm_module.py
rg -n "ss2d|SS2D|SSMBlock2D" model.py tests/test_ssm_module.py
git diff --check -- model.py tests/test_ssm_module.py
```

Expected: compilation succeeds, the stale-symbol search has no matches, and diff check is clean.

- [ ] **Step 4: Review mutation coverage**

Confirm the tests would fail for each realistic defect:

- moving SpikMamba after GroupNorm;
- using a shared PLIF instance;
- replacing depthwise Conv1D with channel mixing;
- passing the pre-Conv1D tensor into projections;
- averaging rather than summing directions;
- gating with projected features rather than original `F`;
- dropping the residual;
- failing to forward `time_step`;
- enabling only one or two encoder blocks;
- silently running without `mamba_ssm` or an injected scan.

- [ ] **Step 5: Inspect final worktree without staging user changes**

Run:

```bash
git status --short
git diff --stat
git diff -- model.py tests/test_ssm_module.py
```

Report the implementation files separately from the pre-existing user changes. Do not commit the overlapping implementation unless the user explicitly authorizes a combined or carefully reconstructed commit.

- [ ] **Step 6: Record the CUDA smoke-test handoff**

If the current node has no CUDA or compiled `mamba_ssm`, report that limitation. On the configured SLURM GPU environment, run a small real-kernel `SpikMamba2D` and `SNNBraTS` forward/backward and assert finite outputs, loss, and gradients before starting a full training run.
