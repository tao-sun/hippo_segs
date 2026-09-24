import numpy as np
import pytest

from postprocessing import (
    morphological_closing_3d,
    postprocess_brats_prediction,
    remove_small_components_3d,
)


def test_remove_small_components_keeps_large_and_removes_small_component():
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[1:6, 1:6, 1:5] = 1  # 100 voxels
    mask[9, 9, 8:11] = 1  # 3 voxels

    result = remove_small_components_3d(mask, min_size=10)

    assert result[1:6, 1:6, 1:5].sum() == 100
    assert result[9, 9, 8:11].sum() == 0


def test_remove_small_components_retains_every_component_at_or_above_threshold():
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[1:3, 1:3, 1:3] = True  # exactly 8 voxels
    mask[8:11, 8:11, 8:11] = True  # 27 voxels

    result = remove_small_components_3d(mask, min_size=8)

    assert result.sum() == 35
    assert result[1:3, 1:3, 1:3].all()
    assert result[8:11, 8:11, 8:11].all()


def test_remove_small_components_uses_26_connectivity_by_default():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1

    result = remove_small_components_3d(mask, min_size=2)

    assert result.sum() == 2


def test_remove_small_components_handles_empty_and_disabled_masks():
    empty = np.zeros((4, 4, 4), dtype=np.uint8)
    mask = empty.copy()
    mask[1, 1, 1] = 1

    assert not remove_small_components_3d(empty, min_size=10).any()
    assert np.array_equal(remove_small_components_3d(mask, min_size=0), mask)


def test_morphological_closing_fills_small_internal_3d_hole():
    mask = np.zeros((7, 7, 7), dtype=np.uint8)
    mask[1:6, 1:6, 1:6] = 1
    mask[3, 3, 3] = 0

    result = morphological_closing_3d(mask, radius=1)

    assert result[3, 3, 3]
    assert result.sum() == 125


def test_morphological_closing_preserves_foreground_at_volume_boundaries():
    mask = np.ones((5, 5, 5), dtype=np.uint8)

    result = morphological_closing_3d(mask, radius=1)

    assert np.array_equal(result, mask)


def test_morphological_closing_radius_zero_is_disabled():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1

    result = morphological_closing_3d(mask, radius=0)

    assert np.array_equal(result, mask)


def test_postprocess_brats_prediction_processes_channels_independently():
    prediction = np.zeros((3, 9, 9, 9), dtype=np.uint8)
    prediction[0, 1:4, 1:4, 1:4] = 1
    prediction[0, 7, 7, 7] = 1
    prediction[1, 1:3, 1:3, 1:3] = 1
    prediction[2, 5:8, 5:8, 5:8] = 1

    result = postprocess_brats_prediction(
        prediction,
        min_component_sizes=(2, 9, 0),
        closing_radius=0,
    )

    assert result.shape == prediction.shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 1})
    assert result[0].sum() == 27
    assert result[1].sum() == 0
    assert result[2].sum() == 27


def test_postprocess_brats_prediction_can_be_fully_disabled():
    prediction = np.zeros((3, 5, 5, 5), dtype=np.uint8)
    prediction[:, 2, 2, 2] = 1

    result = postprocess_brats_prediction(
        prediction,
        min_component_sizes=(0, 0, 0),
        closing_radius=0,
    )

    assert np.array_equal(result, prediction)


@pytest.mark.parametrize(
    "prediction",
    [
        np.zeros((2, 4, 4, 4), dtype=np.uint8),
        np.zeros((3, 4, 4), dtype=np.uint8),
    ],
)
def test_postprocess_brats_prediction_rejects_wrong_shape(prediction):
    with pytest.raises(ValueError, match=r"\[3, X, Y, Z\]"):
        postprocess_brats_prediction(prediction)


def test_postprocessing_rejects_nonbinary_input():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1, 1, 1] = 2

    with pytest.raises(ValueError, match="binary"):
        remove_small_components_3d(mask, min_size=1)

