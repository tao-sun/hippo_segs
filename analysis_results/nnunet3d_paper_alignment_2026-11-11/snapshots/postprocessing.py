"""Three-dimensional post-processing for predicted BraTS masks only."""

from typing import Sequence

import numpy as np
from scipy import ndimage


def validate_postprocessing_parameters(
    apply_postprocessing: bool,
    min_component_sizes: Sequence[int],
    closing_radius: int,
) -> tuple[bool, tuple[int, int, int], int]:
    """Validate and normalize configurable BraTS post-processing settings."""
    if not isinstance(apply_postprocessing, bool):
        raise ValueError("apply_postprocessing must be a boolean")
    if (
        not isinstance(min_component_sizes, (list, tuple))
        or len(min_component_sizes) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in min_component_sizes
        )
    ):
        raise ValueError(
            "min_component_sizes must contain three integer values for ET, TC, and WT"
        )
    if isinstance(closing_radius, bool) or not isinstance(closing_radius, int):
        raise ValueError("closing_radius must be an integer")
    return apply_postprocessing, tuple(min_component_sizes), closing_radius


def _binary_mask_3d(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError("mask must have shape [X, Y, Z]")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("mask must be binary")
    return array.astype(bool, copy=False)


def remove_small_components_3d(
    mask: np.ndarray,
    min_size: int,
    connectivity: int = 26,
) -> np.ndarray:
    """Remove small disconnected regions from a predicted binary 3D mask."""
    binary = _binary_mask_3d(mask)
    if min_size <= 0:
        return binary.astype(np.uint8)
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be one of 6, 18, or 26")

    if connectivity == 26:
        structure = np.ones((3, 3, 3), dtype=bool)
    else:
        structure = ndimage.generate_binary_structure(
            rank=3,
            connectivity=1 if connectivity == 6 else 2,
        )
    labels, _ = ndimage.label(binary, structure=structure)
    component_sizes = np.bincount(labels.ravel())
    keep = component_sizes >= min_size
    keep[0] = False
    return keep[labels].astype(np.uint8)


def morphological_closing_3d(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Fill small holes and discontinuities in a predicted binary 3D mask."""
    binary = _binary_mask_3d(mask)
    if radius <= 0:
        return binary.astype(np.uint8)

    coordinates = np.arange(-radius, radius + 1)
    xx, yy, zz = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    structure = xx**2 + yy**2 + zz**2 <= radius**2
    padded = np.pad(binary, radius, mode="edge")
    closed = ndimage.binary_closing(padded, structure=structure)
    crop = (slice(radius, -radius),) * 3
    return closed[crop].astype(np.uint8)


def postprocess_brats_prediction(
    prediction: np.ndarray,
    min_component_sizes: Sequence[int] = (0, 0, 0),
    closing_radius: int = 1,
) -> np.ndarray:
    """Apply independent 3D CCA and closing to predicted ET/TC/WT masks."""
    array = np.asarray(prediction)
    if array.ndim != 4 or array.shape[0] != 3:
        raise ValueError("prediction must have shape [3, X, Y, Z]")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("prediction must be binary")
    if len(min_component_sizes) != 3:
        raise ValueError("min_component_sizes must contain ET, TC, and WT values")

    output = np.empty(array.shape, dtype=np.uint8)
    for channel, min_size in enumerate(min_component_sizes):
        filtered = remove_small_components_3d(array[channel], min_size=min_size)
        output[channel] = morphological_closing_3d(filtered, radius=closing_radius)

    if output.shape != array.shape or not np.isin(output, (0, 1)).all():
        raise RuntimeError("post-processing must preserve shape and binary values")
    return output
