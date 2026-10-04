"""Convert extracted BraTS24 cases to nnU-Net raw data without preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import filecmp
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import nibabel as nib
import numpy as np
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


MODALITIES = ("t1n", "t1c", "t2w", "t2f")
CHANNEL_NAMES = {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"}
REGION_LABELS = {
    "background": 0,
    "whole_tumor": (1, 2, 3),
    "tumor_core": (1, 3),
    "enhancing_tumor": 3,
}
DATASET_DIR = re.compile(r"^Dataset(?P<id>\d{3})_(?P<name>[A-Za-z0-9][A-Za-z0-9_-]*)$")


@dataclass(frozen=True)
class SubjectFiles:
    """One extracted subject, with modalities ordered for nnU-Net channels."""

    subject_id: str
    modalities: tuple[Path, Path, Path, Path]
    segmentation: Path | None
    source_dir: Path


def _subject_id(name: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    if not identifier:
        raise ValueError(f"Subject identifier is empty after sanitizing {name!r}")
    return identifier


def discover_subjects(root: Path) -> list[SubjectFiles]:
    """Validate and list every case in an extracted tree in stable order."""

    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".zip":
        raise ValueError("ZIP archives are unsupported; supply an extracted directory")
    if not root.is_dir():
        raise ValueError(f"Expected an extracted directory: {root}")
    if any(path.suffix.lower() == ".zip" for path in root.rglob("*")):
        raise ValueError("ZIP archives are unsupported; supply an extracted directory")

    grouped: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower().endswith(".nii.gz"):
            grouped.setdefault(path.parent, []).append(path)
    if not grouped:
        raise ValueError(f"No BraTS NIfTI subjects found in {root}")

    subjects: list[SubjectFiles] = []
    seen: dict[str, Path] = {}
    for directory, files in sorted(grouped.items()):
        matches: dict[str, list[Path]] = {token: [] for token in (*MODALITIES, "seg")}
        for path in sorted(files):
            tokens = [token for token in matches if token in path.name.lower()]
            if len(tokens) != 1:
                raise ValueError(f"Ambiguous or unrecognized modality in {path}")
            matches[tokens[0]].append(path)
        for token in MODALITIES:
            if len(matches[token]) != 1:
                reason = "missing" if not matches[token] else "duplicate"
                raise ValueError(f"{directory}: {reason} modality {token}")
        if len(matches["seg"]) > 1:
            raise ValueError(f"{directory}: ambiguous segmentation (duplicate seg files)")
        subject_id = _subject_id(directory.name)
        if subject_id in seen:
            raise ValueError(
                f"Sanitized subject identifier collision: {seen[subject_id]} and {directory}"
            )
        seen[subject_id] = directory
        subjects.append(
            SubjectFiles(
                subject_id,
                tuple(matches[token][0] for token in MODALITIES),
                matches["seg"][0] if matches["seg"] else None,
                directory,
            )
        )
    return sorted(subjects, key=lambda subject: subject.subject_id)


def is_nnunet_raw_dataset(root: Path) -> bool:
    """Recognize a native raw dataset with its required training structure."""

    root = Path(root)
    if not root.is_dir() or DATASET_DIR.fullmatch(root.name) is None:
        return False
    if not (root / "imagesTr").is_dir() or not (root / "labelsTr").is_dir():
        return False
    try:
        metadata = json.loads((root / "dataset.json").read_text())
    except (OSError, ValueError):
        return False
    return all(
        key in metadata for key in ("channel_names", "labels", "numTraining", "file_ending")
    )


def _load_segmentation(source: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(source))
    labels = np.asanyarray(image.dataobj)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected a 3D integer segmentation: {source}")
    unexpected = np.setdiff1d(np.unique(labels), [0, 1, 2, 3, 4])
    if unexpected.size:
        raise ValueError(f"Unexpected BraTS24 labels in {source}: {unexpected.tolist()}")
    return image, labels


def remap_brats24_segmentation(source: Path, destination: Path) -> None:
    """Write a geometry-preserving NIfTI with only label 4 changed to 0."""

    image, labels = _load_segmentation(Path(source))
    remapped = np.array(labels, copy=True)
    remapped[remapped == 4] = 0
    output = nib.Nifti1Image(remapped, image.affine, header=image.header.copy())
    nib.save(output, str(destination))


def _same_files(source: Path, destination: Path) -> bool:
    source_files = {path.relative_to(source) for path in source.rglob("*") if path.is_file()}
    destination_files = {path.relative_to(destination) for path in destination.rglob("*") if path.is_file()}
    if source_files != destination_files:
        return False
    return all(
        filecmp.cmp(source / relative, destination / relative, shallow=False)
        for relative in source_files
    )


def _check_formatted_source(source: Path, dataset_id: int, dataset_name: str) -> None:
    if not is_nnunet_raw_dataset(source):
        raise ValueError(f"Invalid nnU-Net raw dataset: {source}")
    match = DATASET_DIR.fullmatch(source.name)
    assert match is not None
    metadata = json.loads((source / "dataset.json").read_text())
    if int(match["id"]) != dataset_id or match["name"] != dataset_name:
        raise ValueError("Formatted dataset ID/name does not match the request")
    if "name" in metadata and metadata["name"] != dataset_name:
        raise ValueError("Formatted dataset.json name does not match the request")


def convert_or_register_dataset(
    dataset_root: Path, dataset_id: int, dataset_name: str, nnunet_raw: Path
) -> Path:
    """Convert extracted cases or register an already formatted raw dataset."""

    source = Path(dataset_root)
    nnunet_raw = Path(nnunet_raw)
    if not isinstance(dataset_id, int) or not 0 <= dataset_id <= 999:
        raise ValueError("Dataset ID must be between 0 and 999")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset_name):
        raise ValueError("Invalid dataset name")
    destination = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}"

    if source.is_file() and source.suffix.lower() == ".zip":
        raise ValueError("ZIP archives are unsupported; supply an extracted directory")
    if (source / "dataset.json").exists() or DATASET_DIR.fullmatch(source.name):
        _check_formatted_source(source, dataset_id, dataset_name)
        if source.resolve() == destination.resolve():
            return destination
        if destination.exists():
            if _same_files(source, destination):
                return destination
            raise ValueError(f"Non-identical raw destination collision: {destination}")
        nnunet_raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=nnunet_raw) as staging:
            staged_dataset = Path(staging) / "dataset"
            shutil.copytree(source, staged_dataset)
            os.replace(staged_dataset, destination)
        return destination

    subjects = discover_subjects(source)
    for subject in subjects:
        for image in subject.modalities:
            nib.load(str(image))
        if subject.segmentation is not None:
            _load_segmentation(subject.segmentation)
    if destination.exists():
        raise ValueError(f"Raw destination collision: {destination}")

    nnunet_raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=nnunet_raw) as staging:
        output = Path(staging)
        for folder in ("imagesTr", "labelsTr", "imagesTs"):
            (output / folder).mkdir()
        for subject in subjects:
            images = output / ("imagesTr" if subject.segmentation else "imagesTs")
            for channel, image in enumerate(subject.modalities):
                shutil.copy2(image, images / f"{subject.subject_id}_{channel:04d}.nii.gz")
            if subject.segmentation is not None:
                remap_brats24_segmentation(
                    subject.segmentation, output / "labelsTr" / f"{subject.subject_id}.nii.gz"
                )
        generate_dataset_json(
            str(output),
            CHANNEL_NAMES.copy(),
            REGION_LABELS.copy(),
            sum(subject.segmentation is not None for subject in subjects),
            ".nii.gz",
            regions_class_order=(2, 1, 3),
            dataset_name=dataset_name,
        )
        os.replace(output, destination)
    return destination
