"""Temporal slices keep their native order during augmentation and inference."""

import pytest
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from test_trainer import REGION_DATASET, make_plans
from snn_nnunet.trainer import nnUNetTrainerSNNFPTT


@pytest.mark.parametrize(
    ("axis", "expected"), [(0, (1, 2)), (1, (0, 2)), (2, (0, 1))]
)
def test_temporal_axis_removed_from_native_mirroring(axis, expected, tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_results", str(tmp_path))
    native_return = ("rotation", False, [128, 128, 128], (0, 1, 2))

    def native_mirroring(self):
        self.inference_allowed_mirroring_axes = (0, 1, 2)
        return native_return

    monkeypatch.setattr(nnUNetTrainer, "configure_rotation_dummyDA_mirroring_and_inital_patch_size", native_mirroring)
    trainer = nnUNetTrainerSNNFPTT(make_plans(axis), "3d_fullres", 0, REGION_DATASET, torch.device("cpu"))

    result = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

    assert result == ("rotation", False, [128, 128, 128], expected)
    assert trainer.inference_allowed_mirroring_axes == expected
