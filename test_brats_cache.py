import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
import pytest
import torch
import yaml

import snn_fptt
from build_brats_cache import build_cache
from snn_fptt import BratsVolumeDataset, build_subject_cache_file


def _write_synthetic_subject(
    root: Path,
    fold: int = 1,
    subject_id: str = "BraTS-GLI-TEST-000",
) -> Path:
    subject_dir = root / str(fold) / subject_id
    view_dir = subject_dir / "sagittal"
    view_dir.mkdir(parents=True)

    for slice_index in range(2):
        for modality_index, modality in enumerate(snn_fptt.MOD_ORDER):
            pixels = np.arange(12, dtype=np.uint8).reshape(3, 4)
            pixels = pixels + 20 * modality_index + 3 * slice_index
            Image.fromarray(pixels).save(
                view_dir / f"{subject_id}_{modality}_{slice_index:03d}.png"
            )

    segmentation = np.array(
        [
            [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]],
            [[3, 2, 1, 0], [2, 1, 0, 3], [1, 0, 3, 2]],
        ],
        dtype=np.uint8,
    )
    nib.save(
        nib.Nifti1Image(segmentation, affine=np.eye(4)),
        subject_dir / f"{subject_id}_seg.nii.gz",
    )
    return subject_dir


def test_cached_dataset_matches_original_dataset_bit_for_bit(tmp_path, monkeypatch):
    monkeypatch.setattr(snn_fptt, "TARGET_SHAPE", (2, 3, 4))
    subject_dir = _write_synthetic_subject(tmp_path / "dataset")

    original_dataset = BratsVolumeDataset(
        root=str(tmp_path / "dataset"),
        val_fold=1,
        view="sagittal",
    )
    expected_images, expected_labels, expected_meta = original_dataset[0]

    cache_root = tmp_path / "cache"
    cache_path = build_subject_cache_file(
        subject_dir=subject_dir,
        fold=1,
        view="sagittal",
        cache_root=cache_root,
    )
    cached_dataset = BratsVolumeDataset(
        root=str(tmp_path / "dataset"),
        val_fold=1,
        view="sagittal",
        cache_root=str(cache_root),
        cache_required=True,
    )
    actual_images, actual_labels, actual_meta = cached_dataset[0]

    pixel_grid = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected_images_literal = torch.stack(
        [
            torch.stack([pixel_grid + offset for offset in (0, 20, 40, 60)]),
            torch.stack([pixel_grid + offset for offset in (3, 23, 43, 63)]),
        ]
    ) / 255.0
    segmentation_literal = torch.tensor(
        [
            [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]],
            [[3, 2, 1, 0], [2, 1, 0, 3], [1, 0, 3, 2]],
        ]
    )
    expected_labels_literal = torch.stack(
        [
            segmentation_literal == 3,
            (segmentation_literal == 1) | (segmentation_literal == 3),
            (segmentation_literal == 1)
            | (segmentation_literal == 2)
            | (segmentation_literal == 3),
        ]
    ).permute(1, 0, 2, 3).float()

    assert torch.equal(expected_images, expected_images_literal)
    assert torch.equal(expected_labels, expected_labels_literal)
    assert expected_meta == {"sid": "BraTS-GLI-TEST-000", "xyz": (2, 3, 4)}
    assert torch.equal(actual_images, expected_images)
    assert torch.equal(actual_labels, expected_labels)
    assert actual_meta == expected_meta

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert payload["images"].dtype == torch.uint8
    assert payload["segmentation"].dtype == torch.uint8
    assert tuple(payload["xyz"]) == (2, 3, 4)


def test_cache_builder_writes_manifest_and_skips_valid_subjects(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(snn_fptt, "TARGET_SHAPE", (2, 3, 4))
    data_root = tmp_path / "dataset"
    _write_synthetic_subject(data_root)
    cache_root = tmp_path / "cache"

    first_summary = build_cache(
        data_root=data_root,
        cache_root=cache_root,
        view="sagittal",
        folds=[1],
        workers=1,
    )
    second_summary = build_cache(
        data_root=data_root,
        cache_root=cache_root,
        view="sagittal",
        folds=[1],
        workers=1,
    )

    assert first_summary["created"] == 1
    assert first_summary["skipped"] == 0
    assert second_summary["created"] == 0
    assert second_summary["skipped"] == 1

    manifest_path = cache_root / "sagittal" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert manifest["view"] == "sagittal"
    assert manifest["num_subjects"] == 1
    assert manifest["subjects"] == {
        "1/BraTS-GLI-TEST-000": {
            "cache_file": "1/BraTS-GLI-TEST-000.pt",
            "images_shape": [2, 4, 3, 4],
            "segmentation_shape": [2, 3, 4],
        }
    }


def test_experiment_config_rejects_invalid_cache_loader_settings(tmp_path):
    config = yaml.safe_load(
        (Path(__file__).parent / "experiments_snn_fptt.yaml").read_text(
            encoding="utf-8"
        )
    )
    config.update(
        {
            "cache_root": str(tmp_path / "cache"),
            "cache_required": True,
            "loader_workers": 0,
            "loader_prefetch_factor": 1,
        }
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="loader_workers"):
        snn_fptt.load_experiment_from_yaml(str(config_path))


def test_cache_builder_rebuilds_stale_cache_format(tmp_path, monkeypatch):
    monkeypatch.setattr(snn_fptt, "TARGET_SHAPE", (2, 3, 4))
    data_root = tmp_path / "dataset"
    subject_dir = _write_synthetic_subject(data_root)
    cache_root = tmp_path / "cache"
    cache_path = build_subject_cache_file(
        subject_dir=subject_dir,
        fold=1,
        view="sagittal",
        cache_root=cache_root,
    )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    payload["format_version"] = 0
    torch.save(payload, cache_path)

    summary = build_cache(
        data_root=data_root,
        cache_root=cache_root,
        view="sagittal",
        folds=[1],
        workers=1,
    )
    rebuilt = torch.load(cache_path, map_location="cpu", weights_only=False)

    assert summary["created"] == 1
    assert summary["skipped"] == 0
    assert rebuilt["format_version"] == snn_fptt.CACHE_FORMAT_VERSION


def test_cache_validation_rejects_wrong_segmentation_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(snn_fptt, "TARGET_SHAPE", (2, 3, 4))
    subject_dir = _write_synthetic_subject(tmp_path / "dataset")
    cache_path = build_subject_cache_file(
        subject_dir=subject_dir,
        fold=1,
        view="sagittal",
        cache_root=tmp_path / "cache",
    )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    payload["segmentation"] = payload["segmentation"][:1]
    payload["xyz"] = (1, 3, 4)
    torch.save(payload, cache_path)

    with pytest.raises(ValueError, match="TARGET_SHAPE"):
        snn_fptt.load_subject_cache_file(cache_path)


def test_optional_cache_falls_back_to_source_when_cache_is_invalid(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(snn_fptt, "TARGET_SHAPE", (2, 3, 4))
    data_root = tmp_path / "dataset"
    subject_dir = _write_synthetic_subject(data_root)
    cache_root = tmp_path / "cache"
    cache_path = build_subject_cache_file(
        subject_dir=subject_dir,
        fold=1,
        view="sagittal",
        cache_root=cache_root,
    )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    payload["format_version"] = 0
    torch.save(payload, cache_path)

    original = BratsVolumeDataset(
        root=str(data_root),
        val_fold=1,
        view="sagittal",
    )[0]
    with pytest.warns(RuntimeWarning, match="Falling back to source files"):
        fallback = BratsVolumeDataset(
            root=str(data_root),
            val_fold=1,
            view="sagittal",
            cache_root=str(cache_root),
            cache_required=False,
        )[0]

    assert torch.equal(fallback[0], original[0])
    assert torch.equal(fallback[1], original[1])
    assert fallback[2] == original[2]


def test_manifest_retains_valid_subjects_from_previous_fold_builds(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(snn_fptt, "TARGET_SHAPE", (2, 3, 4))
    data_root = tmp_path / "dataset"
    _write_synthetic_subject(data_root, fold=1, subject_id="BraTS-TEST-A")
    _write_synthetic_subject(data_root, fold=2, subject_id="BraTS-TEST-B")
    cache_root = tmp_path / "cache"

    build_cache(data_root, cache_root, "sagittal", folds=[1], workers=1)
    summary = build_cache(
        data_root,
        cache_root,
        "sagittal",
        folds=[2],
        workers=1,
    )

    assert summary["num_subjects"] == 2
    assert set(summary["subjects"]) == {
        "1/BraTS-TEST-A",
        "2/BraTS-TEST-B",
    }


def test_cache_builder_deduplicates_fold_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(snn_fptt, "TARGET_SHAPE", (2, 3, 4))
    data_root = tmp_path / "dataset"
    _write_synthetic_subject(data_root)

    summary = build_cache(
        data_root,
        tmp_path / "cache",
        "sagittal",
        folds=[1, 1],
        workers=2,
    )

    assert summary["created"] == 1
    assert summary["num_subjects"] == 1


def test_required_cache_is_checked_before_first_dataset_item(tmp_path):
    data_root = tmp_path / "dataset"
    _write_synthetic_subject(data_root)

    with pytest.raises(FileNotFoundError, match="Required subject cache"):
        BratsVolumeDataset(
            root=str(data_root),
            val_fold=1,
            view="sagittal",
            cache_root=str(tmp_path / "empty-cache"),
            cache_required=True,
        )
