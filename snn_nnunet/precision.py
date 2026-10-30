"""FP32-only execution policy for the SNN's continuous computations.

Mamba's full precision mode is FP32. No AMP, BF16/FP16 parameter conversion or
TF32 reduced-mantissa CUDA math is permitted. Spike values remain intentional.
Backend flags are process-wide and are enforced at construction and forward.
"""
from functools import wraps
from inspect import signature

import torch
from torch import nn


def configure_full_precision():
    if torch.get_float32_matmul_precision() != 'highest':
        torch.set_float32_matmul_precision('highest')
    if torch.backends.cuda.matmul.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
    if torch.backends.cudnn.allow_tf32:
        torch.backends.cudnn.allow_tf32 = False


def require_fp32(tensor, label):
    if tensor.dtype != torch.float32:
        raise TypeError(f'{label} must be FP32 (torch.float32); got {tensor.dtype}. '
                        'Reduced precision values cannot be restored by upcasting.')


def require_fp32_module(module):
    for name, value in module.named_parameters():
        if value.is_floating_point():
            require_fp32(value, f'Parameter {name}')
    for name, value in module.named_buffers():
        if value.is_floating_point():
            require_fp32(value, f'Buffer {name}')


def fp32_forward(forward):
    """Guard bare-core calls as well as native adapter calls, without casting."""
    input_name = tuple(signature(forward).parameters)[1]
    @wraps(forward)
    def run(self, *args, **kwargs):
        if not args and input_name not in kwargs:
            return forward(self, *args, **kwargs)
        x = args[0] if args else kwargs[input_name]
        require_fp32(x, 'Network input')
        require_fp32_module(self)
        configure_full_precision()
        with torch.autocast(x.device.type, enabled=False):
            return forward(self, *args, **kwargs)
    return run


class FP32Module(nn.Module):
    """Reject dtype conversion before it can mutate any parameter or state."""
    def __init__(self):
        super().__init__()
        configure_full_precision()

    def _apply(self, fn, recurse=True):
        require_fp32_module(self)
        # Probe an empty tensor: no model values are converted or rounded.
        probe = fn(torch.empty(0, dtype=torch.float32))
        require_fp32(probe, 'Requested module dtype')
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # A conversion to FP32 cannot recover an already rounded checkpoint.
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                require_fp32(value, f'Checkpoint tensor {name}')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
