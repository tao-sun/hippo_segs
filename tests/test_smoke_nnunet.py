"""Small, isolated native nnU-Net lifecycle smoke test."""

import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from conftest import TinySequentialCore


DATASET_NAME = "Dataset987_SyntheticSNN"
CASE_NAMES = tuple(f"Case{i:03d}" for i in range(5))


def _write_case(folder: Path, name: str, index: int, *, labeled: bool) -> None:
    rng = np.random.default_rng(index)
    shape = (16, 16, 16)
    for channel in range(4):
        data = rng.normal(100 + channel * 20, 10, shape).astype(np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), folder / f"{name}_{channel:04d}.nii.gz")
    if labeled:
        segmentation = np.zeros(shape, dtype=np.uint8)
        segmentation[3:12, 3:12, 3:12] = 1
        segmentation[5:10, 5:10, 5:10] = 2
        segmentation[7:9, 7:9, 7:9] = 3
        nib.save(nib.Nifti1Image(segmentation, np.eye(4)), folder.parent / "labelsTr" / f"{name}.nii.gz")


def _make_dataset(raw_root: Path) -> Path:
    dataset = raw_root / DATASET_NAME
    for directory in ("imagesTr", "imagesTs", "labelsTr"):
        (dataset / directory).mkdir(parents=True)
    for index, name in enumerate(CASE_NAMES):
        _write_case(dataset / "imagesTr", name, index, labeled=True)
    _write_case(dataset / "imagesTs", "CaseTest", 9, labeled=False)
    metadata = {
        "channel_names": {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"},
        "labels": {
            "background": 0,
            "whole_tumor": [1, 2, 3],
            "tumor_core": [1, 3],
            "enhancing_tumor": 3,
        },
        "regions_class_order": [2, 1, 3],
        "numTraining": len(CASE_NAMES),
        "file_ending": ".nii.gz",
        "name": "SyntheticSNN",
    }
    (dataset / "dataset.json").write_text(json.dumps(metadata))
    return dataset


@pytest.mark.integration
def test_native_synthetic_lifecycle(tmp_path):
    if os.environ.get("SNN_SMOKE_CHILD") == "1":
        _run_native_lifecycle(Path(os.environ["SNN_SMOKE_ROOT"]))
        return

    _make_dataset(tmp_path / "raw")
    environment = os.environ.copy()
    environment.update({
        "SNN_SMOKE_CHILD": "1",
        "SNN_SMOKE_ROOT": str(tmp_path),
        "nnUNet_raw": str(tmp_path / "raw"),
        "nnUNet_preprocessed": str(tmp_path / "preprocessed"),
        "nnUNet_results": str(tmp_path / "results"),
        "nnUNet_extTrainer": str(Path(__file__).resolve().parents[1] / "snn_nnunet"),
        "nnUNet_compile": "false",
        "nnUNet_n_proc_DA": "1",
        "OMP_NUM_THREADS": "1",
    })
    subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-m", "integration", "-q", "-s"],
        cwd=Path(__file__).resolve().parents[1], env=environment, check=True,
    )


def _run_native_lifecycle(root: Path) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    from snn_nnunet import cli, network_adapter
    from snn_nnunet.trainer import nnUNetTrainerSNNFPTT

    assert cli.main([
        "prepare", "--dataset-root", str(root / "raw" / DATASET_NAME),
        "--dataset-id", "987", "--dataset-name", "SyntheticSNN",
        "--fingerprint-processes", "1", "--preprocess-processes", "1",
    ]) == 0
    preprocessed = root / "preprocessed" / DATASET_NAME
    plans_path = preprocessed / "SNNPlans.json"
    assert plans_path.is_file()
    assert (preprocessed / "dataset_fingerprint.json").is_file()
    assert (preprocessed / "nnUNetPlans.json").is_file()
    assert (preprocessed / "nnUNetPlans_3d_fullres").is_dir()
    print("NATIVE_PREPARATION_COMPLETE", flush=True)

    production_plans = json.loads(plans_path.read_text())
    assert production_plans["configurations"]["3d_fullres"]["patch_size"] == [128, 128, 128]
    assert production_plans["configurations"]["3d_fullres"]["batch_size"] == 4
    smoke_plans = json.loads(plans_path.read_text())
    smoke_plans["plans_name"] = "SmokeSNNPlans"
    smoke_plans["configurations"]["3d_fullres"]["patch_size"] = [8, 8, 8]
    smoke_plans["configurations"]["3d_fullres"]["batch_size"] = 1
    (preprocessed / "SmokeSNNPlans.json").write_text(json.dumps(smoke_plans))
    dataset_json = json.loads((root / "raw" / DATASET_NAME / "dataset.json").read_text())

    # This is the only network substitution. All nnU-Net services stay native.
    patch = pytest.MonkeyPatch()
    patch.setattr(network_adapter, "build_core", lambda config: TinySequentialCore())
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NATIVE_SMOKE_DEVICE={device}", flush=True)
        trainer = nnUNetTrainerSNNFPTT(
            {**smoke_plans, "continue_training": False}, "3d_fullres", 0,
            dataset_json, device,
        )
        trainer.initialize()
        assert type(trainer.network.core) is TinySequentialCore
        assert type(trainer.loss) is DC_and_BCE_loss
        assert type(trainer).validation_step is nnUNetTrainer.validation_step
        train_keys, val_keys = trainer.do_split()
        assert set(train_keys).isdisjoint(val_keys)
        splits = json.loads((preprocessed / "splits_final.json").read_text())
        assert len(splits) == 5
        assert {name for split in splits for name in split["val"]} == set(CASE_NAMES)
        assert all(set(split["train"]).isdisjoint(split["val"]) for split in splits)

        train_loader, val_loader = trainer.get_dataloaders()
        try:
            trainer.network.train()
            train_output = trainer.train_step(next(train_loader))
            assert np.isfinite(train_output["loss"])
            assert train_output["optimizer_updates"] == 1
            trainer.on_validation_epoch_start()
            validation_output = trainer.validation_step(next(val_loader))
            assert np.isfinite(validation_output["loss"])
        finally:
            train_loader._finish()
            val_loader._finish()
        print("NATIVE_TRAIN_AND_VALIDATION_COMPLETE", flush=True)

        result_folder = Path(trainer.output_folder_base)
        (result_folder / "plans.json").write_text(json.dumps(smoke_plans))
        (result_folder / "dataset.json").write_text(json.dumps(dataset_json))
        checkpoint_path = Path(trainer.output_folder) / "checkpoint_final.pth"
        trainer.save_checkpoint(str(checkpoint_path))
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert saved["trainer_name"] == "nnUNetTrainerSNNFPTT"
        assert "fptt_state" in saved
        resumed = nnUNetTrainerSNNFPTT(
            {**smoke_plans, "continue_training": True}, "3d_fullres", 0,
            dataset_json, device,
        )
        resumed.load_checkpoint(str(checkpoint_path))
        assert resumed.current_epoch == saved["current_epoch"]
        for name, parameter in trainer.network.named_parameters():
            torch.testing.assert_close(parameter, dict(resumed.network.named_parameters())[name])
        torch.testing.assert_close(resumed.network.avg_weights, trainer.network.avg_weights)
        torch.testing.assert_close(resumed.network.lambdas, trainer.network.lambdas)
        assert resumed.optimizer.state_dict()["state"]
        print("NATIVE_CHECKPOINT_RESUME_COMPLETE", flush=True)

        predictor = nnUNetPredictor(
            device=device, perform_everything_on_device=device.type == "cuda",
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(result_folder), use_folds=(0,), checkpoint_name="checkpoint_final.pth"
        )
        assert type(predictor) is nnUNetPredictor
        assert type(predictor.network.core) is TinySequentialCore
        assert predictor.configuration_manager.patch_size == [8, 8, 8]
        output_folder = root / "predictions"
        predictor.predict_from_files(
            str(root / "raw" / DATASET_NAME / "imagesTs"), str(output_folder),
            num_processes_preprocessing=1, num_processes_segmentation_export=1,
        )
        prediction = output_folder / "CaseTest.nii.gz"
        assert prediction.is_file()
        predicted = np.asanyarray(nib.load(str(prediction)).dataobj)
        assert predicted.shape == (16, 16, 16)
        assert set(np.unique(predicted)).issubset({0, 1, 2, 3})
        print("NATIVE_PREDICTION_EXPORT_COMPLETE", flush=True)
    finally:
        patch.undo()
