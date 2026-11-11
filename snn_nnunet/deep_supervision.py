"""Paper U-Net losses on full-resolution or native-scale region targets."""
import math

import torch.nn.functional as F


def deep_supervision_loss(outputs, target, loss, weights, *, temporal_axis=0, spatial_stride=2):
    """Apply normalized native losses while preserving the sequence and ignore mask.

    Native half/quarter logits include the model's spatial padding. Pad the
    target to that same grid before nearest-exact reduction, so odd dimensions
    keep the decoder's coordinates. Equal-sized outputs retain legacy behavior.
    """
    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise ValueError('deep supervision requires main, half and quarter logits')
    if (not isinstance(weights, (list, tuple)) or len(weights) != 3
            or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights)
            or not math.isfinite(sum(weights))):
        raise ValueError('deep supervision requires three positive finite loss weights')
    if type(temporal_axis) is not int or temporal_axis not in (0, 1, 2):
        raise ValueError('temporal_axis must be 0, 1 or 2')
    if type(spatial_stride) is not int or spatial_stride < 2:
        raise ValueError('spatial_stride must be an integer >= 2')
    if target.ndim != 5 or any(output.ndim != 5 for output in outputs):
        raise ValueError('deep supervision requires [B, C, X, Y, Z] tensors')
    if outputs[0].shape[0] != target.shape[0] or outputs[0].shape[2:] != target.shape[2:]:
        raise ValueError('main logits must match the full-resolution target grid')
    if any(output.shape[:2] != outputs[0].shape[:2] for output in outputs):
        raise ValueError('deep supervision logits must share batch and class counts')
    if any(output.shape[2 + temporal_axis] != target.shape[2 + temporal_axis]
           for output in outputs):
        raise ValueError('deep supervision must preserve the temporal dimension')

    targets = [target]
    # A tiny padded native head may happen to equal the uncropped main shape.
    # Decide across the whole pyramid, rather than treating that head as legacy.
    native = any(output.shape[2:] != target.shape[2:] for output in outputs[1:])
    padded_grid = None
    for index, output in enumerate(outputs[1:], start=1):
        if not native:
            targets.append(target)
            continue
        scale = spatial_stride ** index
        grid = tuple(size if axis == temporal_axis else output.shape[2 + axis] * scale
                     for axis, size in enumerate(target.shape[2:]))
        if any(padded < original for padded, original in zip(grid, target.shape[2:])):
            raise ValueError('auxiliary logits are smaller than the native padded grid')
        if padded_grid is not None and grid != padded_grid:
            raise ValueError('auxiliary logits must share the same padded input grid')
        padded_grid = grid
        padding = tuple(value for axis in reversed(range(3))
                        for value in (0, grid[axis] - target.shape[2 + axis]))
        padded_target = F.pad(target, padding) if any(padding) else target
        if getattr(loss, 'use_ignore_label', False) and any(padding):
            # Original ignored voxels and added padding stay excluded at every scale.
            padded_target[:, -1:] = F.pad(target[:, -1:], padding, value=1)
        targets.append(F.interpolate(padded_target.float(), size=output.shape[2:],
                                     mode='nearest-exact').to(target.dtype))
    total = sum(weights)
    return sum((weight / total) * loss(output, scaled_target)
               for weight, output, scaled_target in zip(weights, outputs, targets))
