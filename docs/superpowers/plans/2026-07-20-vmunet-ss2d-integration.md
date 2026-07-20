# Faithful VM-UNet SS2D Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simplified spatial SSM with the official VM-UNet VSS/SS2D implementation using `mamba_ssm==1.0.1`, while preserving the surrounding spiking U-Net interface and behavior.

**Architecture:** Port the upstream BHWC SS2D, including its projections, four-direction cross scan, initializations, gate, and output projection. Reduce the local VSS wrapper to pre-normalization plus residual so those operations are not duplicated, and adapt BCHW at the SNN boundary. Inject the selective-scan callable for CPU structural tests while requiring the official CUDA kernel in production.

**Tech Stack:** Python 3.9, PyTorch, `mamba_ssm==1.0.1`, `causal_conv1d==1.0.0`, einops, timm, pytest/unittest, CUDA 11.7/11.8.

## Global Constraints

- Preserve `SSMBlock2D.forward(x, time_step)` and its BCHW input/output contract.
- Preserve the SNN encoder, decoder, PLIF behavior, TBPTT/FPTT, and insertion after `conv_block3`.
- Port VM-UNet SS2D with `d_state=16`, `expand=2`, and automatic `dt_rank=ceil(d_model/16)`.
- Use `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` in production.
- Do not silently fall back to the old Python recurrence.
- Sum the four aligned scan outputs, matching upstream; do not average them.
- Keep upstream Apache-2.0 attribution in source documentation.
- Old simplified-SS2D checkpoints are intentionally incompatible and must remain tied to their original commit.
- Implement with tests first and observe every new behavioral test fail before production changes.

---

### Task 1: Define SS2D initialization and dependency behavior

**Files:**
- Modify: `tests/test_ssm_module.py`
- Modify: `model.py`

**Interfaces:**
- Consumes: `SS2D(d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dropout=0.0, selective_scan=None)`.
- Produces: `dt_init`, `A_log_init`, `D_init`, dimension attributes, and an actionable kernel import path.

- [ ] **Step 1: Write failing initialization tests**

Add tests asserting that for `SS2D(d_model=8, d_state=4, expand=2)`: `d_inner == 16`, `dt_rank == 1`, `A_logs.shape == (64, 4)`, `Ds.shape == (64,)`, `-exp(A_logs)` is negative, `Ds` is all ones, and softplus of each time-step bias lies within `[0.001, 0.1]` up to floating-point tolerance.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/test_ssm_module.py -k "initialization or dimensions"
```

Expected: failure because the current `SS2D` accepts `channels`, has no `d_inner`, `dt_rank`, `A_logs`, or `Ds`, and lacks upstream initializers.

- [ ] **Step 3: Implement the upstream constructor and initializers**

Port `dt_init`, `A_log_init`, and `D_init` from the attached official VM-UNet module. Add an optional `selective_scan` constructor argument used only for CPU tests; when omitted, resolve the official imported kernel and raise an actionable `ImportError` if unavailable.

- [ ] **Step 4: Run the initialization tests and verify GREEN**

Run:

```bash
pytest -q tests/test_ssm_module.py -k "initialization or dimensions"
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the initialization unit**

```bash
git add model.py tests/test_ssm_module.py
git commit -m "feat: add VM-UNet SS2D parameters"
```

---

### Task 2: Implement four-direction cross scan and selective core

**Files:**
- Modify: `tests/test_ssm_module.py`
- Modify: `model.py`

**Interfaces:**
- Consumes: BCHW inner features `(B, d_inner, H, W)` and the initialized SS2D parameters.
- Produces: `_cross_scan(x) -> (B, 4, d_inner, H*W)`, `_align_scan_outputs(out_y, H, W) -> tuple[Tensor, Tensor, Tensor, Tensor]`, and `forward_core(x)`.

- [ ] **Step 1: Write a failing exact-order cross-scan test**

Use a tensor containing sequential values and assert that the four packed sequences are row-major, transposed-row-major, reverse-row-major, and reverse-transposed-row-major in that exact order.

- [ ] **Step 2: Run the cross-scan test and verify RED**

```bash
pytest -q tests/test_ssm_module.py -k cross_scan
```

Expected: failure because `_cross_scan` does not exist.

- [ ] **Step 3: Implement `_cross_scan` and output alignment**

Implement the upstream reshape/transpose/flip operations without changing sequence order. Keep these transformations in named helpers so they can be tested without the CUDA kernel.

- [ ] **Step 4: Run cross-scan tests and verify GREEN**

```bash
pytest -q tests/test_ssm_module.py -k "cross_scan or align_scan"
```

Expected: all selected tests pass.

- [ ] **Step 5: Write a failing selective-kernel contract test**

Provide a recording test double that accepts the same keyword arguments as `selective_scan_fn`, returns a deterministic `(B, 4*d_inner, L)` tensor, and records shapes. Assert `u/delta=(B,4*d_inner,L)`, `A=(4*d_inner,d_state)`, `B/C=(B,4,d_state,L)`, `D/delta_bias=(4*d_inner,)`, `z is None`, `delta_softplus is True`, and `return_last_state is False`.

- [ ] **Step 6: Run the kernel-contract test and verify RED**

```bash
pytest -q tests/test_ssm_module.py -k selective_scan_contract
```

Expected: failure because `forward_core` does not implement the official packed call.

- [ ] **Step 7: Implement `forward_core`**

Port the upstream direction-specific `x_proj_weight`, low-rank `dt_projs_weight`, float32 conversion, packed kernel call, and four-output alignment. Return the four aligned `(B, d_inner, L)` tensors.

- [ ] **Step 8: Run core tests and verify GREEN**

```bash
pytest -q tests/test_ssm_module.py -k "cross_scan or align_scan or selective_scan_contract"
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit the selective core**

```bash
git add model.py tests/test_ssm_module.py
git commit -m "feat: port VM-UNet four-direction selective scan"
```

---

### Task 3: Complete SS2D and integrate the faithful VSS wrapper

**Files:**
- Modify: `tests/test_ssm_module.py`
- Modify: `model.py`

**Interfaces:**
- Consumes: `SS2D.forward_core` and BCHW tensors from `SSMBlock2D`.
- Produces: BHWC `SS2D.forward`, BCHW `VSSBlock2D.forward`, and unchanged BCHW `SSMBlock2D.forward`.

- [ ] **Step 1: Write failing SS2D forward tests**

With the scan test double, assert BHWC shape preservation, that the four aligned outputs are summed, that gradients reach input and projection parameters, and that invalid rank or final channel size raises a descriptive `ValueError`.

- [ ] **Step 2: Run SS2D forward tests and verify RED**

```bash
pytest -q tests/test_ssm_module.py -k ss2d_forward
```

Expected: failure because the old SS2D accepts BCHW and lacks the official input/gate/output path.

- [ ] **Step 3: Implement official SS2D forward**

Implement BHWC `in_proj`, split `x/z`, BCHW depthwise convolution and SiLU, `forward_core`, sum of four outputs, BHWC restoration, output LayerNorm, `SiLU(z)` gating, output projection, and optional dropout.

- [ ] **Step 4: Run SS2D forward tests and verify GREEN**

```bash
pytest -q tests/test_ssm_module.py -k ss2d_forward
```

Expected: all selected tests pass.

- [ ] **Step 5: Write failing VSS integration tests**

Assert that `VSSBlock2D(channels=8, selective_scan=fake_scan)` preserves BCHW shape, backpropagates, contains a single SS2D-owned input projection and depthwise convolution, and applies a residual. Retain the existing `SSMBlock2D` shape/backpropagation test using dependency injection.

- [ ] **Step 6: Run VSS integration tests and verify RED**

```bash
pytest -q tests/test_ssm_module.py -k "vss or ssm_block"
```

Expected: failure because the current wrapper duplicates projections/convolution/gating and expects the old SS2D BCHW interface.

- [ ] **Step 7: Implement the VM-UNet-compatible wrapper**

Reduce the wrapper to BCHW-to-BHWC conversion, LayerNorm, official SS2D, optional DropPath, residual addition, and BHWC-to-BCHW conversion. Ensure `SSMBlock2D` continues to construct it by default and honor an explicitly supplied `ssm_module` instead of overwriting it.

- [ ] **Step 8: Run all SSM module tests and verify GREEN**

```bash
pytest -q tests/test_ssm_module.py
```

Expected: all tests pass with no CUDA requirement because they inject the scan test double.

- [ ] **Step 9: Commit the integrated block**

```bash
git add model.py tests/test_ssm_module.py
git commit -m "feat: integrate faithful VM-UNet VSS block"
```

---

### Task 4: Document dependencies and verify the official CUDA kernel

**Files:**
- Create or Modify: `requirements-vmunet-ss2d.txt`
- Modify: `snn_fptt.job`
- Create: `tests/test_vmunet_ss2d_cuda.py`
- Modify: `README.md` if present; otherwise create `docs/vmunet-ss2d.md`

**Interfaces:**
- Consumes: the integrated VSS/SS2D model and a CUDA SLURM node.
- Produces: reproducible environment instructions and compiled-kernel verification.

- [ ] **Step 1: Add the pinned dependency manifest**

Record exactly:

```text
torch==1.13.0
timm==0.4.12
einops
causal_conv1d==1.0.0
mamba_ssm==1.0.1
```

Document that the matching CUDA 11.7/11.8 PyTorch wheel and compiled extensions must be installed in an isolated environment.

- [ ] **Step 2: Write the CUDA comparison test**

Mark the test skipped unless CUDA and `mamba_ssm` are available. Instantiate official `selective_scan_fn` and `selective_scan_ref` on identical small packed tensors, compare forward outputs with explicit tolerances, then run SS2D forward/backward and assert finite outputs and gradients.

- [ ] **Step 3: Run the local test suite**

```bash
pytest -q
```

Expected: CPU tests pass; the compiled-kernel test reports SKIPPED on the login node rather than failing.

- [ ] **Step 4: Run the CUDA test through SLURM**

Execute in the pinned environment on a CUDA node:

```bash
pytest -q tests/test_vmunet_ss2d_cuda.py -rs
```

Expected: compiled-versus-reference comparison passes, SS2D forward/backward is finite, and no test is skipped.

- [ ] **Step 5: Run one-model smoke integration**

Instantiate `SNNBraTS`, run one small input window and backward pass, and confirm output shape `(B, out_channels, k, H, W)` plus finite gradients. Then run the existing one-subject, one-epoch configuration before a full experiment.

- [ ] **Step 6: Commit dependency and CUDA validation files**

```bash
git add requirements-vmunet-ss2d.txt snn_fptt.job tests/test_vmunet_ss2d_cuda.py README.md docs/vmunet-ss2d.md
git commit -m "test: validate VM-UNet selective scan integration"
```

Stage only paths that exist and were intentionally changed.

---

## Completion criteria

- The official VM-UNet SS2D parameterization and four-direction scan are present.
- Production execution requires `mamba_ssm==1.0.1`; there is no silent simplified fallback.
- The VSS wrapper does not duplicate SS2D projections, depthwise convolution, normalization, or gate.
- CPU structural tests pass with a scan test double.
- The official compiled kernel agrees with `selective_scan_ref` on a CUDA node.
- `SNNBraTS` preserves its output contract and completes a finite backward pass.
- Source attribution and checkpoint incompatibility are documented.

