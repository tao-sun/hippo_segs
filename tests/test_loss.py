"""The trainer retains nnU-Net's region loss exactly."""

import torch
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

from test_trainer import REGION_DATASET, make_plans
from snn_nnunet.trainer import nnUNetTrainerSNNFPTT


def test_region_loss_is_native_dice_and_bce(tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_results", str(tmp_path))
    trainer = nnUNetTrainerSNNFPTT(make_plans(), "3d_fullres", 0, REGION_DATASET, torch.device("cpu"))
    loss = trainer._build_loss()
    assert type(loss) is DC_and_BCE_loss
    assert type(loss.dc) is MemoryEfficientSoftDiceLoss
    assert type(loss.ce) is torch.nn.BCEWithLogitsLoss
