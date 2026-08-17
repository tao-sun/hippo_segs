# Patch-based SpikMamba Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build autonomous Conv2D patch embedding, one-direction patch-sequence SpikeMamba, and residual FFN modules without changing the existing segmentation models.

**Architecture:** `Spiking2DPatchEmbedding` creates a non-overlapping 4x4 patch grid for one BCHW frame and applies PLIF plus learned spatial/temporal position embeddings. `SpikeMambaLayer` flattens the grid row-major, executes expanded causal Conv1D and a continuous selective SSM with three independent PLIF stages, and gates the original tokens. `SpikMambaBlock` owns the Mamba and FFN residuals.

**Tech Stack:** Python 3.9, PyTorch 2.5.1, `mamba_ssm` 2.2.6, repository `PLIFNode`, pytest 8.

## Global Constraints

- Create only `spikmamba.py` and `tests/test_spikmamba.py`; do not edit `model.py`, `ConvBlock`, `SNNBraTS`, or `SpikMamba2D`.
- Export exactly `Spiking2DPatchEmbedding`, `SpikeMambaLayer`, and `SpikMambaBlock`.
- Default to `patch_size=4`; Conv2D kernel and stride both equal the patch size.
- Construct every spike stage with `PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True)`.
- Scan only `H_patch * W_patch` row-major tokens, causally and forward once.
- Keep the SSM recurrence continuous; no PLIF or threshold occurs inside it.
- Include no SpikeSLA, attention, Q/K/V, SS2D, VMamba, CrossScan, reverse scan, direction dimension, or direction merge.
- Keep the FFN internal: LayerNorm, Linear `D -> 4D`, GELU, dropout, Linear `4D -> D`, dropout.
- Keep Python 3.9 compatibility with `typing.Union`.
- Preserve unrelated dirty changes in `model.py`, `tests/test_ssm_module.py`, and cache files.
- Follow RED-GREEN-REFACTOR for every production class.

---

### Task 1: Spiking Conv2D Patch Embedding

**Files:**
- Create: `tests/test_spikmamba.py`
- Create: `spikmamba.py`

**Interfaces:**
- Consumes: `PLIFNode.forward(dv, time_step)`.
- Produces: `Spiking2DPatchEmbedding.forward(x: Tensor, time_step: int) -> Tensor`, BCHW to `(B,D,H/4,W/4)`.

- [ ] **Step 1: Write failing patch-embedding tests**

Create `tests/test_spikmamba.py`:

```python
import torch
import torch.nn as nn
import pytest

import surrogate
from spike_neurons import PLIFNode
from spikmamba import Spiking2DPatchEmbedding


class RecordingPLIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_steps = []

    def forward(self, x, time_step):
        self.time_steps.append(time_step)
        return x, x


def test_patch_embedding_construction_and_shape():
    module = Spiking2DPatchEmbedding(
        4, 8, image_size=(16, 20)
    )
    assert module.proj.kernel_size == (4, 4)
    assert module.proj.stride == (4, 4)
    assert module.proj.bias is not None
    assert isinstance(module.norm, nn.BatchNorm2d)
    assert isinstance(module.sl_patch, PLIFNode)
    assert module.sl_patch.detach_reset is True
    assert isinstance(module.sl_patch.surrogate_function, surrogate.ATan)
    assert module.spatial_pos_embed.shape == (1, 8, 4, 5)
    assert module.temporal_pos_embed.shape == (256, 8)

    recorder = RecordingPLIF()
    module.sl_patch = recorder
    module.eval()
    output = module(torch.randn(2, 4, 16, 20), time_step=7)
    assert output.shape == (2, 8, 4, 5)
    assert recorder.time_steps == [7]


def test_patch_embedding_interpolates_space_and_selects_time():
    module = Spiking2DPatchEmbedding(
        1, 2, image_size=(8, 8), max_time_steps=4
    )
    module.sl_patch = RecordingPLIF()
    module.eval()
    with torch.no_grad():
        module.spatial_pos_embed.zero_()
        module.temporal_pos_embed.zero_()
        module.temporal_pos_embed[2].fill_(1.5)
    x = torch.randn(1, 1, 8, 12)
    at_two = module(x, 2)
    at_three = module(x, 3)
    assert at_two.shape == (1, 2, 2, 3)
    torch.testing.assert_close(
        at_two - at_three, torch.full_like(at_two, 1.5)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 0, "embed_dim": 4},
        {"in_channels": 1, "embed_dim": 0},
        {"in_channels": 1, "embed_dim": 4, "patch_size": 0},
        {"in_channels": 1, "embed_dim": 4, "max_time_steps": 0},
        {"in_channels": 1, "embed_dim": 4, "image_size": (9, 8)},
    ],
)
def test_patch_embedding_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        Spiking2DPatchEmbedding(**kwargs)


def test_patch_embedding_rejects_bad_input():
    module = Spiking2DPatchEmbedding(
        4, 8, image_size=(16, 20), max_time_steps=3
    )
    with pytest.raises(ValueError, match="BCHW"):
        module(torch.randn(2, 4, 16), 0)
    with pytest.raises(ValueError, match="4 channels"):
        module(torch.randn(2, 3, 16, 20), 0)
    with pytest.raises(ValueError, match="divisible"):
        module(torch.randn(2, 4, 15, 20), 0)
    with pytest.raises(ValueError, match="time_step"):
        module(torch.randn(2, 4, 16, 20), 3)


def test_patch_plif_continues_then_restarts_at_time_zero():
    module = Spiking2DPatchEmbedding(
        1, 1, image_size=(8, 8), max_time_steps=4
    ).eval()
    with torch.no_grad():
        module.proj.weight.zero_()
        module.proj.bias.fill_(0.5)
        module.spatial_pos_embed.zero_()
        module.temporal_pos_embed.zero_()
    x = torch.zeros(1, 1, 8, 8)
    module(x, 0)
    first = module.sl_patch.v.detach().clone()
    module(x, 1)
    continued = module.sl_patch.v.detach().clone()
    module(x, 0)
    restarted = module.sl_patch.v.detach().clone()
    assert not torch.allclose(first, continued)
    torch.testing.assert_close(first, restarted)
```

- [ ] **Step 2: Run RED**

Run `.venv/bin/python -m pytest tests/test_spikmamba.py -q`.

Expected: `ModuleNotFoundError: No module named 'spikmamba'`.

- [ ] **Step 3: Implement patch embedding**

Create `spikmamba.py`:

```python
"""Patch-based spiking Mamba components for per-frame BCHW features."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import surrogate
from spike_neurons import PLIFNode

__all__ = ["Spiking2DPatchEmbedding"]


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _make_plif(init_tau: float, device=None, dtype=None) -> PLIFNode:
    node = PLIFNode(
        init_tau=init_tau,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
    )
    if device is not None or dtype is not None:
        node = node.to(device=device, dtype=dtype)
    return node


class Spiking2DPatchEmbedding(nn.Module):
    """Project one BCHW frame into a spiking patch grid."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        image_size: Tuple[int, int] = (160, 192),
        patch_size: int = 4,
        max_time_steps: int = 256,
        init_tau: float = 2.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        self.embed_dim = _positive_int("embed_dim", embed_dim)
        self.patch_size = _positive_int("patch_size", patch_size)
        self.max_time_steps = _positive_int(
            "max_time_steps", max_time_steps
        )
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError("image_size must be a (height, width) tuple")
        height = _positive_int("image_size[0]", image_size[0])
        width = _positive_int("image_size[1]", image_size[1])
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")

        kwargs = {"device": device, "dtype": dtype}
        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            self.patch_size,
            stride=self.patch_size,
            bias=True,
            **kwargs,
        )
        self.norm = nn.BatchNorm2d(self.embed_dim, **kwargs)
        self.sl_patch = _make_plif(
            init_tau, device=device, dtype=dtype
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.empty(
                1,
                self.embed_dim,
                height // self.patch_size,
                width // self.patch_size,
                **kwargs,
            )
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(self.max_time_steps, self.embed_dim, **kwargs)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, time_step: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"expected BCHW input, got shape={tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {x.shape[1]}"
            )
        height, width = x.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("input size must be divisible by patch_size")
        if (
            isinstance(time_step, bool)
            or not isinstance(time_step, int)
            or not 0 <= time_step < self.max_time_steps
        ):
            raise ValueError("time_step is outside the configured range")

        patches = self.norm(self.proj(x))
        patches, _ = self.sl_patch(patches, time_step)
        spatial = self.spatial_pos_embed
        if spatial.shape[-2:] != patches.shape[-2:]:
            spatial = F.interpolate(
                spatial,
                size=patches.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        temporal = self.temporal_pos_embed[time_step].view(
            1, self.embed_dim, 1, 1
        )
        return patches + spatial + temporal
```

- [ ] **Step 4: Run GREEN**

Run `.venv/bin/python -m pytest tests/test_spikmamba.py -q`.

Expected: all Task 1 tests pass without warnings.

- [ ] **Step 5: Commit**

```bash
git add spikmamba.py tests/test_spikmamba.py
git commit -m "feat: add spiking 2D patch embedding"
```

---

### Task 2: Continuous One-direction SpikeMamba Layer

**Files:**
- Modify: `tests/test_spikmamba.py`
- Modify: `spikmamba.py`

**Interfaces:**
- Consumes: patch grid `(B,D,H_patch,W_patch)`, absolute `time_step`, and a selective-scan callable.
- Produces: same-shaped gated patch grid before the block residual.

- [ ] **Step 1: Append failing Mamba tests**

Change the test import to include `SpikeMambaLayer`, then append:

```python
class IdentityPLIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_steps = []

    def forward(self, x, time_step):
        self.time_steps.append(time_step)
        return x, x


class OnesPLIF(nn.Module):
    def forward(self, x, time_step):
        return torch.ones_like(x), x


class RecordingScan:
    def __init__(self):
        self.call = None
        self.calls = 0
        self.u_values = None

    def __call__(
        self, u, delta, A, B, C, D=None, z=None,
        delta_bias=None, delta_softplus=False,
        return_last_state=False,
    ):
        self.calls += 1
        self.u_values = u.detach().clone()
        self.call = {
            "u": u.shape,
            "delta": delta.shape,
            "A": A.shape,
            "B": B.shape,
            "C": C.shape,
            "D": D.shape,
            "z": z,
            "delta_bias": delta_bias.shape,
            "delta_softplus": delta_softplus,
            "return_last_state": return_last_state,
        }
        return u


def test_mamba_construction_and_independent_plifs():
    module = SpikeMambaLayer(
        4, d_state=3, selective_scan=RecordingScan()
    )
    assert module.d_inner == 8
    assert module.dt_rank == 1
    assert (module.linear_m.in_features,
            module.linear_m.out_features) == (4, 8)
    assert module.conv1d_m.in_channels == 8
    assert module.conv1d_m.out_channels == 8
    assert module.conv1d_m.groups == 8
    assert module.conv1d_m.kernel_size == (4,)
    assert module.conv1d_m.padding == (3,)
    assert all(
        isinstance(stage, PLIFNode)
        for stage in (module.sl_m1, module.sl_m2, module.sl_ssm)
    )
    assert all(
        stage.detach_reset is True
        and isinstance(stage.surrogate_function, surrogate.ATan)
        for stage in (module.sl_m1, module.sl_m2, module.sl_ssm)
    )
    assert len({
        id(module.sl_m1), id(module.sl_m2), id(module.sl_ssm)
    }) == 3


def test_patch_grid_uses_exact_row_major_order():
    patches = torch.tensor(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]]
    )
    tokens = SpikeMambaLayer._to_tokens(patches)
    assert tokens.shape == (1, 6, 1)
    torch.testing.assert_close(
        tokens[0, :, 0], torch.arange(6, dtype=torch.float32)
    )


def test_forward_scans_one_row_major_sequence_without_flip_or_reverse():
    recorder = RecordingScan()
    module = SpikeMambaLayer(
        1,
        d_state=1,
        d_conv=1,
        expand=1,
        selective_scan=recorder,
    )
    module.sl_m1 = IdentityPLIF()
    module.sl_m2 = IdentityPLIF()
    module.sl_ssm = IdentityPLIF()
    with torch.no_grad():
        module.linear_m.weight.fill_(1.0)
        module.conv1d_m.weight.fill_(1.0)
        module.conv1d_m.bias.zero_()

    patches = torch.arange(6, dtype=torch.float32).view(1, 1, 2, 3)
    module(patches, time_step=2)

    assert recorder.calls == 1
    torch.testing.assert_close(
        recorder.u_values[0, 0],
        torch.arange(6, dtype=torch.float32),
    )


def test_scan_length_is_patch_count_with_no_direction_axis():
    recorder = RecordingScan()
    module = SpikeMambaLayer(
        4, d_state=3, selective_scan=recorder
    )
    module.sl_m1 = IdentityPLIF()
    module.sl_m2 = IdentityPLIF()
    module.sl_ssm = IdentityPLIF()
    output = module(torch.randn(2, 4, 2, 3), time_step=7)
    assert output.shape == (2, 4, 2, 3)
    assert recorder.call == {
        "u": torch.Size([2, 8, 6]),
        "delta": torch.Size([2, 8, 6]),
        "A": torch.Size([8, 3]),
        "B": torch.Size([2, 3, 6]),
        "C": torch.Size([2, 3, 6]),
        "D": torch.Size([8]),
        "z": None,
        "delta_bias": torch.Size([8]),
        "delta_softplus": True,
        "return_last_state": False,
    }
    assert module.sl_m1.time_steps == [7]
    assert module.sl_m2.time_steps == [7]
    assert module.sl_ssm.time_steps == [7]


def test_mamba_plifs_continue_then_restart_at_time_zero():
    module = SpikeMambaLayer(
        2,
        d_state=2,
        expand=1,
        selective_scan=RecordingScan(),
    )
    stimulus = torch.full((1, 2, 3), 0.5)

    for stage in (module.sl_m1, module.sl_m2, module.sl_ssm):
        stage(stimulus, 0)
        first = stage.v.detach().clone()
        stage(stimulus, 1)
        continued = stage.v.detach().clone()
        stage(stimulus, 0)
        restarted = stage.v.detach().clone()

        assert not torch.allclose(first, continued)
        torch.testing.assert_close(first, restarted)


def test_conv1d_is_causal_and_preserves_length():
    module = SpikeMambaLayer(
        2, d_state=2, d_conv=3, expand=1,
        selective_scan=RecordingScan(),
    )
    with torch.no_grad():
        module.conv1d_m.weight.fill_(1)
        module.conv1d_m.bias.zero_()
    sequence = torch.zeros(1, 2, 5)
    sequence[:, :, -1] = 1
    output = module._causal_conv(sequence)
    assert output.shape == sequence.shape
    torch.testing.assert_close(output[:, :, :4], torch.zeros(1, 2, 4))
    torch.testing.assert_close(output[:, :, 4], torch.ones(1, 2))


def test_ssm_spikes_gate_original_tokens():
    module = SpikeMambaLayer(
        3, d_state=2, expand=1,
        selective_scan=RecordingScan(),
    )
    module.sl_m1 = IdentityPLIF()
    module.sl_m2 = IdentityPLIF()
    module.sl_ssm = OnesPLIF()
    patches = torch.randn(2, 3, 2, 2)
    torch.testing.assert_close(module(patches, 4), patches)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 0},
        {"dim": 4, "d_state": 0},
        {"dim": 4, "d_conv": 0},
        {"dim": 4, "expand": 0},
        {"dim": 4, "dt_rank": 0},
        {"dim": 4, "dt_init": "bad"},
    ],
)
def test_mamba_rejects_bad_configuration(kwargs):
    with pytest.raises((ValueError, NotImplementedError)):
        SpikeMambaLayer(
            selective_scan=RecordingScan(), **kwargs
        )


def test_mamba_rejects_bad_grid_and_missing_scan():
    module = SpikeMambaLayer(
        4, d_state=2, selective_scan=RecordingScan()
    )
    with pytest.raises(ValueError, match="BDHW"):
        module(torch.randn(1, 4, 8), 0)
    with pytest.raises(ValueError, match="4 channels"):
        module(torch.randn(1, 3, 2, 2), 0)
    module.selective_scan = None
    with pytest.raises(ImportError, match="mamba_ssm"):
        module(torch.randn(1, 4, 2, 2), 0)
```

- [ ] **Step 2: Run RED**

Run `.venv/bin/python -m pytest tests/test_spikmamba.py -q`.

Expected: `ImportError: cannot import name 'SpikeMambaLayer'`.

- [ ] **Step 3: Implement the Mamba layer**

Add `import math`, change the typing import to `Tuple, Union`, import `selective_scan_fn` behind this guarded block, and export the new class:

```python
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _MAMBA_IMPORT_ERROR = None
except ImportError as exc:
    selective_scan_fn = None
    _MAMBA_IMPORT_ERROR = exc
```

Append:

```python
class SpikeMambaLayer(nn.Module):
    """Continuous causal SSM over row-major patch tokens."""

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
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.d_state = _positive_int("d_state", d_state)
        self.d_conv = _positive_int("d_conv", d_conv)
        self.expand = _positive_int("expand", expand)
        self.d_inner = self.expand * self.dim
        self.dt_rank = (
            math.ceil(self.dim / 16)
            if dt_rank == "auto"
            else _positive_int("dt_rank", dt_rank)
        )
        if dt_min <= 0 or dt_max <= 0 or dt_min > dt_max:
            raise ValueError("require 0 < dt_min <= dt_max")
        if dt_init_floor <= 0:
            raise ValueError("dt_init_floor must be positive")
        if dt_init not in {"constant", "random"}:
            raise NotImplementedError(f"unsupported dt_init={dt_init!r}")

        kwargs = {"device": device, "dtype": dtype}
        self.linear_m = nn.Linear(
            self.dim, self.d_inner, bias=bias, **kwargs
        )
        self.sl_m1 = _make_plif(
            init_tau, device=device, dtype=dtype
        )
        self.conv1d_m = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
            **kwargs,
        )
        self.sl_m2 = _make_plif(
            init_tau, device=device, dtype=dtype
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
            **kwargs,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **kwargs
        )

        std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(self.d_inner, **kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)
        self.dt_proj.bias._no_reinit = True

        base = torch.arange(
            1, self.d_state + 1,
            dtype=torch.float32, device=device,
        )
        self.A_log = nn.Parameter(
            torch.log(base.unsqueeze(0).repeat(self.d_inner, 1))
        )
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(
            torch.ones(self.d_inner, dtype=torch.float32, device=device)
        )
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(
            self.d_inner, self.dim, bias=bias, **kwargs
        )
        self.sl_ssm = _make_plif(
            init_tau, device=device, dtype=dtype
        )
        self.selective_scan = (
            selective_scan
            if selective_scan is not None
            else selective_scan_fn
        )

    @staticmethod
    def _to_tokens(patches: torch.Tensor) -> torch.Tensor:
        # B D H W -> B (H W) D, row-major.
        return patches.flatten(2).transpose(1, 2).contiguous()

    def _causal_conv(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.conv1d_m(sequence)[..., :sequence.shape[-1]]

    def forward(self, patches: torch.Tensor, time_step: int) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError(
                f"expected BDHW patches, got {tuple(patches.shape)}"
            )
        if patches.shape[1] != self.dim:
            raise ValueError(
                f"expected {self.dim} channels, got {patches.shape[1]}"
            )
        if self.selective_scan is None:
            raise ImportError(
                "mamba_ssm or an injected selective_scan is required"
            ) from _MAMBA_IMPORT_ERROR

        batch, _, height, width = patches.shape
        original = self._to_tokens(patches)
        x = self.linear_m(original)
        x, _ = self.sl_m1(x, time_step)
        x = self._causal_conv(x.transpose(1, 2).contiguous())
        x, _ = self.sl_m2(x, time_step)

        projected = self.x_proj(x.transpose(1, 2))
        delta_low, B, C = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.linear(
            delta_low, self.dt_proj.weight, bias=None
        ).transpose(1, 2).contiguous()
        B = B.transpose(1, 2).contiguous()
        C = C.transpose(1, 2).contiguous()
        A = -torch.exp(self.A_log.float())

        # Continuous recurrence: no spike operation occurs in this call.
        y = self.selective_scan(
            x,
            delta,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        y = self.out_proj(
            y.transpose(1, 2).to(self.out_proj.weight.dtype)
        )
        ssm_spikes, _ = self.sl_ssm(y, time_step)
        gated = ssm_spikes * original
        return (
            gated.transpose(1, 2)
            .contiguous()
            .view(batch, self.dim, height, width)
        )
```

Update `__all__` to include `"SpikeMambaLayer"`.

- [ ] **Step 4: Run GREEN**

Run `.venv/bin/python -m pytest tests/test_spikmamba.py -q`.

Expected: all Task 1-2 tests pass; the scan contract is `(B,d_inner,H_patch*W_patch)` with no direction axis.

- [ ] **Step 5: Commit**

```bash
git add spikmamba.py tests/test_spikmamba.py
git commit -m "feat: add patch-sequence SpikeMamba layer"
```

---

### Task 3: Add the residual SpikMamba block and end-to-end numerical tests

**Files:**
- Modify: `tests/test_spikmamba.py`
- Modify: `spikmamba.py`

- [ ] **Step 1: Write the failing block and integration tests**

Extend the test imports so the module object and every public class are available:

```python
import torch.nn.functional as F

import spikmamba
from spikmamba import (
    SpikeMambaLayer,
    Spiking2DPatchEmbedding,
    SpikMambaBlock,
)
```

Add deterministic replacement modules for isolating each residual branch:

```python
class ZeroMamba(nn.Module):
    def forward(self, patches: torch.Tensor, time_step: int) -> torch.Tensor:
        return torch.zeros_like(patches)


class DoubleMamba(nn.Module):
    def forward(self, patches: torch.Tensor, time_step: int) -> torch.Tensor:
        return 2.0 * patches


class ZeroFFN(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(tokens)


class DoubleFFN(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return 2.0 * tokens
```

Add a differentiable, CPU-compatible selective-scan reference with the same contract used by `SpikeMambaLayer`. This is a test oracle and a fallback only for tests; production still imports the fused Mamba implementation by default:

```python
def continuous_selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor = None,
    z: torch.Tensor = None,
    delta_bias: torch.Tensor = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    batch, channels, length = u.shape
    state_size = A.shape[-1]
    state = torch.zeros(
        batch,
        channels,
        state_size,
        device=u.device,
        dtype=u.dtype,
    )
    outputs = []

    for index in range(length):
        delta_i = delta[:, :, index]
        input_i = u[:, :, index]
        transition = torch.exp(delta_i.unsqueeze(-1) * A.unsqueeze(0))
        input_term = (
            delta_i.unsqueeze(-1)
            * B[:, :, index].unsqueeze(1)
            * input_i.unsqueeze(-1)
        )
        state = transition * state + input_term
        output_i = torch.einsum("bdn,bn->bd", state, C[:, :, index])
        if D is not None:
            output_i = output_i + D.unsqueeze(0) * input_i
        if z is not None:
            output_i = output_i * F.silu(z[:, :, index])
        outputs.append(output_i)

    output = torch.stack(outputs, dim=-1)
    if return_last_state:
        return output, state
    return output
```

Add the residual, FFN, public-API, validation, and end-to-end tests:

```python
def test_block_adds_the_mamba_residual_exactly():
    block = SpikMambaBlock(dim=4, d_state=2, expand=1)
    block.mamba_layer = DoubleMamba()
    block.ffn = ZeroFFN()

    patches = torch.randn(2, 4, 2, 3)
    output = block(patches, time_step=0)

    torch.testing.assert_close(output, 3.0 * patches)


def test_block_adds_the_ffn_residual_exactly():
    block = SpikMambaBlock(dim=4, d_state=2, expand=1)
    block.mamba_layer = ZeroMamba()
    block.ffn_norm = nn.Identity()
    block.ffn = DoubleFFN()

    patches = torch.randn(2, 4, 2, 3)
    output = block(patches, time_step=0)

    torch.testing.assert_close(output, 3.0 * patches)


def test_block_ffn_uses_four_times_channel_width_and_gelu():
    block = SpikMambaBlock(dim=6, mlp_ratio=4.0, d_state=2, expand=1)

    assert isinstance(block.ffn_norm, nn.LayerNorm)
    assert block.ffn_norm.normalized_shape == (6,)
    assert isinstance(block.ffn[0], nn.Linear)
    assert block.ffn[0].in_features == 6
    assert block.ffn[0].out_features == 24
    assert isinstance(block.ffn[1], nn.GELU)
    assert isinstance(block.ffn[3], nn.Linear)
    assert block.ffn[3].in_features == 24
    assert block.ffn[3].out_features == 6


def test_public_api_and_module_tree_contain_no_attention_or_cross_scan():
    assert spikmamba.__all__ == [
        "Spiking2DPatchEmbedding",
        "SpikeMambaLayer",
        "SpikMambaBlock",
    ]

    block = SpikMambaBlock(dim=4, d_state=2, expand=1)
    module_names = [type(module).__name__.lower() for module in block.modules()]

    assert not any(isinstance(module, nn.MultiheadAttention) for module in block.modules())
    forbidden_name_fragments = (
        "attention",
        "spikesla",
        "ss2d",
        "vmamba",
        "crossscan",
        "cross_scan",
        "reverse",
        "directionmerge",
        "direction_merge",
    )
    assert not any(
        fragment in name
        for name in module_names
        for fragment in forbidden_name_fragments
    )
    for attribute in (
        "q_proj",
        "k_proj",
        "v_proj",
        "cross_scan",
        "reverse_scan",
        "directions",
        "direction_merge",
    ):
        assert not hasattr(block.mamba_layer, attribute)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dim": 0}, "dim"),
        ({"dim": 4, "mlp_ratio": 0.0}, "mlp_ratio"),
        ({"dim": 4, "dropout": -0.1}, "dropout"),
        ({"dim": 4, "dropout": 1.0}, "dropout"),
    ],
)
def test_block_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SpikMambaBlock(**kwargs)


def test_embedding_and_block_forward_backward_on_cpu():
    torch.manual_seed(4)
    embedding = Spiking2DPatchEmbedding(
        in_channels=4,
        embed_dim=8,
        image_size=(8, 12),
        patch_size=4,
        max_time_steps=3,
    )
    block = SpikMambaBlock(
        dim=8,
        d_state=4,
        d_conv=4,
        expand=2,
        selective_scan=continuous_selective_scan_reference,
    )

    image = torch.randn(2, 4, 8, 12, requires_grad=True)
    patches = embedding(image, time_step=0)
    output = block(patches, time_step=0)
    loss = output.square().mean()
    loss.backward()

    assert patches.shape == (2, 8, 2, 3)
    assert output.shape == patches.shape
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert torch.count_nonzero(image.grad).item() > 0
    parameter_grads = [
        parameter.grad
        for parameter in list(embedding.parameters()) + list(block.parameters())
        if parameter.grad is not None
    ]
    assert parameter_grads
    assert all(torch.isfinite(grad).all() for grad in parameter_grads)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in parameter_grads)


def test_block_preserves_a_non_square_patch_grid():
    block = SpikMambaBlock(
        dim=4,
        d_state=2,
        d_conv=3,
        expand=1,
        selective_scan=continuous_selective_scan_reference,
    )
    patches = torch.randn(1, 4, 2, 5)

    output = block(patches, time_step=0)

    assert output.shape == (1, 4, 2, 5)
```

- [ ] **Step 2: Run the new tests and confirm the RED state**

Run:

```bash
.venv/bin/python -m pytest tests/test_spikmamba.py -q
```

Expected: collection fails because `SpikMambaBlock` is not yet exported.

- [ ] **Step 3: Implement the private FFN inside the public block**

Update the public export list and append the block to `spikmamba.py`:

```python
__all__ = [
    "Spiking2DPatchEmbedding",
    "SpikeMambaLayer",
    "SpikMambaBlock",
]


class SpikMambaBlock(nn.Module):
    """Mamba patch mixer followed by a token-wise FFN, both residual."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **mamba_kwargs,
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        if not isinstance(mlp_ratio, (int, float)) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        hidden_dim = int(self.dim * float(mlp_ratio))
        if hidden_dim < 1:
            raise ValueError("mlp_ratio produces an empty hidden dimension")

        self.mamba_layer = SpikeMambaLayer(dim=self.dim, **mamba_kwargs)
        self.ffn_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, patches: torch.Tensor, time_step: int) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError("patches must have shape [B, D, Hp, Wp]")
        if patches.shape[1] != self.dim:
            raise ValueError(
                f"expected {self.dim} patch channels, got {patches.shape[1]}"
            )

        global_features = patches + self.mamba_layer(patches, time_step)
        batch, channels, height, width = global_features.shape
        tokens = global_features.flatten(2).transpose(1, 2)
        ffn_tokens = self.ffn(self.ffn_norm(tokens))
        ffn_features = ffn_tokens.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )
        return global_features + ffn_features
```

The FFN remains an implementation detail of `SpikMambaBlock`; do not introduce a public `SpikMambaFFN` class.

- [ ] **Step 4: Run the complete focused suite and confirm GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_spikmamba.py -q
```

Expected: all tests in `tests/test_spikmamba.py` pass.

- [ ] **Step 5: Inspect the exact implementation diff**

Run:

```bash
git diff -- spikmamba.py tests/test_spikmamba.py
git diff --check -- spikmamba.py tests/test_spikmamba.py
```

Expected: only the two new implementation/test files are shown, and the
scoped whitespace check prints nothing. A repository-wide `git diff --check`
is intentionally avoided here because the user's existing `model.py` diff
already contains trailing whitespace outside this feature's scope.

- [ ] **Step 6: Commit the block and integration tests**

Run:

```bash
git add spikmamba.py tests/test_spikmamba.py
git diff --cached --name-only
git commit -m "feat: add residual SpikMamba patch block"
```

Expected staged files: only `spikmamba.py` and `tests/test_spikmamba.py`; existing changes in `model.py` and `tests/test_ssm_module.py` remain unstaged.

---

### Task 4: Run regression, smoke, and scope verification

**Files:**
- Verify: `spikmamba.py`
- Verify: `tests/test_spikmamba.py`
- Verify unchanged: `model.py`
- Verify unchanged: `tests/test_ssm_module.py`

- [ ] **Step 1: Run the focused suite verbosely**

Run:

```bash
.venv/bin/python -m pytest tests/test_spikmamba.py -vv
```

Expected: every patch-embedding, scan, residual, validation, shape, and gradient test passes.

- [ ] **Step 2: Run the repository test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: the full suite passes. If an unrelated pre-existing test fails, record the exact command, failure, and whether the same failure is reproducible from the pre-implementation baseline; do not change unrelated code to hide it.

- [ ] **Step 3: Run a direct CPU smoke test**

Run:

```bash
.venv/bin/python -c "import runpy, torch; from spikmamba import Spiking2DPatchEmbedding, SpikMambaBlock; scan = runpy.run_path('tests/test_spikmamba.py')['continuous_selective_scan_reference']; embedding = Spiking2DPatchEmbedding(4, 4, image_size=(8, 12), patch_size=4, max_time_steps=2); block = SpikMambaBlock(4, d_state=2, expand=1, selective_scan=scan); image = torch.randn(2, 4, 8, 12, requires_grad=True); patches = embedding(image, 0); output = block(patches, 0); output.sum().backward(); print(tuple(patches.shape), tuple(output.shape), bool(torch.isfinite(image.grad).all()))"
```

Expected output:

```text
(2, 4, 2, 3) (2, 4, 2, 3) True
```

This confirms that an `8 x 12` frame becomes a row-major sequence of six `4 x 4` spatial patches, while the public tensors remain in grid form.

- [ ] **Step 4: Verify formatting and scope**

Run:

```bash
git diff --check -- spikmamba.py tests/test_spikmamba.py
git status --short
git diff --name-only 0f553d13..HEAD
```

Expected:

- The scoped `git diff --check` prints nothing; the pre-existing whitespace
  finding in `model.py` remains outside this feature.
- Feature commits contain only the implementation plan, `spikmamba.py`, and `tests/test_spikmamba.py`.
- The user's pre-existing modifications to `model.py`, `tests/test_ssm_module.py`, and `__pycache__/model.cpython-39.pyc` remain present and were not included in the feature commits.

- [ ] **Step 5: Report the final architecture and evidence**

The completion report must state:

- Map equation (8) to Conv2D patch projection plus `SL_patch`; equations
  (17)-(20) to `Linear_m`, `SL_m1`, causal depthwise Conv1D, `SL_m2`,
  continuous selective scan, output projection, and `SL_ssm`; equation (21)
  to the Hadamard gate; equation (10) to the Mamba residual; and equation
  (11) to the FFN residual.
- Patches are created in `Spiking2DPatchEmbedding.proj` by `Conv2d(kernel_size=4, stride=4)`.
- A frame `[B, C, H, W]` becomes `[B, D, H/4, W/4]`, then `SpikeMambaLayer` flattens it row-major to `[B, (H/4)(W/4), D]`.
- Give this concrete default-expansion example: with
  `B=2, C_in=4, H=8, W=12, D=8, patch_size=4, d_state=4`, the shapes are
  input `[2,4,8,12]`; patch grid `[2,8,2,3]`; tokens `[2,6,8]`;
  `Linear_m/SL_m1` `[2,6,16]`; causal Conv1D/`SL_m2` `[2,16,6]`;
  Delta `[2,16,6]`; B and C each `[2,4,6]`; A `[16,4]`; scan output
  `[2,16,6]`; projected/gated tokens `[2,6,8]`; FFN hidden tokens
  `[2,6,32]`; and final grid `[2,8,2,3]`.
- The Mamba recurrence scans only that spatial patch sequence; PLIF state across frames is controlled by the supplied `time_step`.
- The layer uses `Linear_m -> PLIF -> causal depthwise Conv1d -> PLIF -> continuous selective scan -> Linear_out -> PLIF -> Hadamard gate`.
- `SpikMambaBlock` applies both the Mamba residual and the `D -> 4D -> D` FFN residual.
- The public API contains exactly `Spiking2DPatchEmbedding`, `SpikeMambaLayer`, and `SpikMambaBlock`.
- There is no attention, QKV path, CrossScan, reverse scan, four-direction merge, SS2D, or VMamba component.
- Include the exact focused/full test commands and their pass counts, or disclose any verified pre-existing failure precisely.
