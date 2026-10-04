"""Behavioral tests for extracted BraTS24 to nnU-Net raw conversion."""

import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pytest

from snn_nnunet.dataset_conversion import (
    convert_or_register_dataset,
    discover_subjects,
    is_nnunet_raw_dataset,
    remap_brats24_segmentation,
)


AFFINE = np.array(
    [[1.2, 0, 0, 8], [0, 1.4, 0, -3], [0, 0, 2.5, 12], [0, 0, 0, 1]],
    dtype=float,
)
TOKENS = ("t1n", "t1c", "t2w", "t2f")


def _write_nifti(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(values, AFFINE)
    image.header.set_xyzt_units("mm", "sec")
    image.header.set_qform(AFFINE, code=2)
    image.header.set_sform(AFFINE, code=4)
    nib.save(image, path)


def _subject(root: Path, name: str, *, labeled: bool = True) -> Path:
    folder = root / name
    for index, token in enumerate(TOKENS):
        _write_nifti(folder / f"{name}-{token}.nii.gz", np.full((2, 2, 2), index + 5, np.float32))
    if labeled:
        _write_nifti(
            folder / f"{name}-seg.nii.gz",
            np.array([0, 1, 2, 3, 4, 4, 1, 2], dtype=np.uint8).reshape(2, 2, 2),
        )
    return folder


def test_conversion_routes_cases_in_exact_channel_order_and_preserves_geometry(tmp_path):
    source = tmp_path / "extracted"
    _subject(source, "BraTS-GLI-0002", labeled=False)
    labeled = _subject(source, "BraTS-GLI-0001")

    destination = convert_or_register_dataset(source, 12, "GLI", tmp_path / "raw")

    assert destination == tmp_path / "raw" / "Dataset012_GLI"
    assert sorted(p.name for p in (destination / "imagesTr").iterdir()) == [
        f"BraTS-GLI-0001_{i:04d}.nii.gz" for i in range(4)
    ]
    assert sorted(p.name for p in (destination / "imagesTs").iterdir()) == [
        f"BraTS-GLI-0002_{i:04d}.nii.gz" for i in range(4)
    ]
    for index, token in enumerate(TOKENS):
        source_image = labeled / f"BraTS-GLI-0001-{token}.nii.gz"
        output_image = destination / "imagesTr" / f"BraTS-GLI-0001_{index:04d}.nii.gz"
        assert output_image.read_bytes() == source_image.read_bytes()
    assert sorted(p.name for p in (destination / "labelsTr").iterdir()) == [
        "BraTS-GLI-0001.nii.gz"
    ]
    source_seg = nib.load(str(labeled / "BraTS-GLI-0001-seg.nii.gz"))
    output_seg = nib.load(str(destination / "labelsTr" / "BraTS-GLI-0001.nii.gz"))
    np.testing.assert_array_equal(
        np.asanyarray(output_seg.dataobj),
        np.array([0, 1, 2, 3, 0, 0, 1, 2], dtype=np.uint8).reshape(2, 2, 2),
    )
    np.testing.assert_array_equal(output_seg.affine, source_seg.affine)
    np.testing.assert_array_equal(output_seg.header.get_zooms(), source_seg.header.get_zooms())
    assert output_seg.header.get_xyzt_units() == source_seg.header.get_xyzt_units()
    assert output_seg.header.get_qform(coded=True)[1] == 2
    assert output_seg.header.get_sform(coded=True)[1] == 4
    metadata = json.loads((destination / "dataset.json").read_text())
    assert metadata["channel_names"] == {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"}
    assert metadata["labels"] == {
        "background": 0,
        "whole_tumor": [1, 2, 3],
        "tumor_core": [1, 3],
        "enhancing_tumor": 3,
    }
    assert metadata["regions_class_order"] == [2, 1, 3]
    assert metadata["numTraining"] == 1
    assert metadata["file_ending"] == ".nii.gz"
    assert metadata["name"] == "GLI"


def test_discovery_is_deterministic_and_sanitizes_subject_identifiers(tmp_path):
    _subject(tmp_path / "nested", "Case B")
    _subject(tmp_path, "Case-A", labeled=False)

    subjects = discover_subjects(tmp_path)

    assert [subject.subject_id for subject in subjects] == ["Case-A", "Case_B"]
    assert [subject.segmentation is None for subject in subjects] == [True, False]


def test_remap_changes_only_label_four_and_preserves_nifti_geometry(tmp_path):
    source = tmp_path / "source.nii.gz"
    destination = tmp_path / "target.nii.gz"
    _write_nifti(source, np.array([0, 1, 2, 3, 4, 3, 2, 1], dtype=np.uint8).reshape(2, 2, 2))

    remap_brats24_segmentation(source, destination)

    before, after = nib.load(str(source)), nib.load(str(destination))
    np.testing.assert_array_equal(np.asanyarray(after.dataobj).ravel(), [0, 1, 2, 3, 0, 3, 2, 1])
    np.testing.assert_array_equal(after.affine, before.affine)
    np.testing.assert_array_equal(after.header.get_zooms(), before.header.get_zooms())
    assert after.header.get_xyzt_units() == before.header.get_xyzt_units()


def test_zip_input_is_rejected_before_destination_creation(tmp_path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"not an extracted directory")
    raw = tmp_path / "raw"

    with pytest.raises(ValueError, match="ZIP|extracted"):
        convert_or_register_dataset(source, 12, "GLI", raw)

    assert not raw.exists()


@pytest.mark.parametrize("invalid", ["duplicate", "missing", "ambiguous_segmentation"])
def test_invalid_late_subject_prevents_all_output(tmp_path, invalid):
    source = tmp_path / "source"
    _subject(source, "Case-A")
    bad = _subject(source, "Case-Z")
    if invalid == "duplicate":
        _write_nifti(bad / "Case-Z-extra-t1n.nii.gz", np.zeros((2, 2, 2), np.float32))
    elif invalid == "missing":
        (bad / "Case-Z-t2f.nii.gz").unlink()
    else:
        _write_nifti(bad / "Case-Z-extra-seg.nii.gz", np.zeros((2, 2, 2), np.uint8))
    raw = tmp_path / "raw"

    with pytest.raises(ValueError, match="duplicate|missing|segmentation|modality"):
        convert_or_register_dataset(source, 12, "GLI", raw)

    assert not raw.exists()


def test_sanitized_identifier_collision_prevents_all_output(tmp_path):
    source = tmp_path / "source"
    _subject(source, "Case A")
    _subject(source, "Case_A")
    raw = tmp_path / "raw"

    with pytest.raises(ValueError, match="collision|identifier"):
        convert_or_register_dataset(source, 12, "GLI", raw)

    assert not raw.exists()


def test_formatted_dataset_registers_byte_for_byte_and_rejects_collision(tmp_path):
    extracted = tmp_path / "extracted"
    _subject(extracted, "Case-A")
    source = convert_or_register_dataset(extracted, 12, "GLI", tmp_path / "first_raw")
    assert is_nnunet_raw_dataset(source)
    target_raw = tmp_path / "second_raw"

    destination = convert_or_register_dataset(source, 12, "GLI", target_raw)

    assert destination == target_raw / "Dataset012_GLI"
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == {
        p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()
    }
    assert convert_or_register_dataset(source, 12, "GLI", target_raw) == destination
    (destination / "dataset.json").write_text("{}")
    with pytest.raises(ValueError, match="collision|different|identical"):
        convert_or_register_dataset(source, 12, "GLI", target_raw)


def test_formatted_dataset_id_and_name_must_match_request(tmp_path):
    extracted = tmp_path / "extracted"
    _subject(extracted, "Case-A")
    source = convert_or_register_dataset(extracted, 12, "GLI", tmp_path / "first_raw")
    raw = tmp_path / "other_raw"

    with pytest.raises(ValueError, match="ID|name|match"):
        convert_or_register_dataset(source, 13, "GLI", raw)

    assert not raw.exists()


def test_interrupted_formatted_copy_leaves_no_partial_destination(tmp_path, monkeypatch):
    extracted = tmp_path / "extracted"
    _subject(extracted, "Case-A")
    source = convert_or_register_dataset(extracted, 12, "GLI", tmp_path / "first_raw")
    raw = tmp_path / "second_raw"

    def interrupted_copytree(_source, target):
        target.mkdir()
        (target / "partial.txt").write_text("partial")
        raise OSError("simulated copy failure")

    monkeypatch.setattr(shutil, "copytree", interrupted_copytree)
    with pytest.raises(OSError, match="simulated copy failure"):
        convert_or_register_dataset(source, 12, "GLI", raw)

    assert not (raw / "Dataset012_GLI").exists()
