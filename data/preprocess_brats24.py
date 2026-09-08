#!/usr/bin/env python3
"""Preprocess BraTS24 NIfTI volumes into the project's PNG/fold format."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:
    import imageio


TARGET_SHAPE = (160, 192, 152)
FOLD_NAMES = ("1", "2", "3", "4", "5")
MODALITIES = {
    "t1n": "t1",
    "t1c": "t1ce",
    "t2w": "t2",
    "t2f": "flair",
}
VALID_LABELS = {0, 1, 2, 3, 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/home/aurora/data/BRATS2024"))
    parser.add_argument("--output", type=Path, default=Path("/home/aurora/data/BRATS2024_preprocessed"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--exclude-additional",
        action="store_true",
        help="Process only training_data1_v2 instead of all training folders",
    )
    return parser.parse_args()


def center_crop3d(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {array.shape}")
    slices = []
    for source, target in zip(array.shape, TARGET_SHAPE):
        if target > source:
            raise ValueError(f"Target shape {TARGET_SHAPE} exceeds source shape {array.shape}")
        start = (source - target) // 2
        slices.append(slice(start, start + target))
    return array[tuple(slices)]


def normalize(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    if not finite.all():
        volume = np.nan_to_num(volume, copy=False)
    nonzero = volume[volume != 0]
    if nonzero.size:
        low, high = float(nonzero.min()), float(nonzero.max())
    else:
        low, high = float(volume.min()), float(volume.max())
    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    result = np.zeros_like(volume, dtype=np.float32)
    result[volume != 0] = (volume[volume != 0] - low) / (high - low)
    return np.clip(result, 0.0, 1.0)


def save_views(volume: np.ndarray, output_dir: Path, stem: str) -> None:
    views = {
        "sagittal": (0, TARGET_SHAPE[0]),
        "coronal": (1, TARGET_SHAPE[1]),
        "axial": (2, TARGET_SHAPE[2]),
    }
    for view, (axis, count) in views.items():
        view_dir = output_dir / view
        view_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = np.take(volume, index, axis=axis)
            imageio.imwrite(
                view_dir / f"Brats17_{stem}_{view}_{index:03d}.png",
                np.rint(image * 255).astype(np.uint8),
            )


def is_seg(path: Path) -> bool:
    return path.name.lower().endswith("-seg.nii.gz") or path.name.lower().endswith("_seg.nii.gz")


def modality_path(subject_dir: Path, token: str) -> Path | None:
    matches = sorted(subject_dir.glob(f"*-{token}.nii.gz"))
    return matches[0] if matches else None


def collect_subjects(group: Path) -> list[Path]:
    if not group.exists():
        raise FileNotFoundError(group)
    subjects = sorted(path for path in group.iterdir() if path.is_dir())
    usable = []
    for subject in subjects:
        if not is_seg(next(iter(subject.glob("*-seg.nii.gz")), Path())):
            continue
        if all(modality_path(subject, token) for token in MODALITIES):
            usable.append(subject)
        else:
            missing = [token for token in MODALITIES if modality_path(subject, token) is None]
            print(f"[Warn] Skipping {subject.name}; missing: {', '.join(missing)}")
    return usable


def training_groups(root: Path, include_additional: bool) -> list[Path]:
    groups = [root / "training_data1_v2"]
    if include_additional:
        groups.append(root / "training_data_additional")
    return groups


def validate_segmentation(seg_path: Path) -> None:
    values = set(np.unique(np.asanyarray(nib.load(str(seg_path)).dataobj)).astype(int).tolist())
    unexpected = values - VALID_LABELS
    if unexpected:
        raise ValueError(f"Unexpected labels {sorted(unexpected)} in {seg_path}")


def process_subject(subject: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_path = next(subject.glob("*-seg.nii.gz"))
    validate_segmentation(seg_path)
    seg_img = nib.load(str(seg_path))
    seg = center_crop3d(np.rint(seg_img.get_fdata(dtype=np.float32)).astype(np.int16))
    header = seg_img.header.copy()
    header.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(seg, seg_img.affine, header), output_dir / f"{subject.name}-seg.nii.gz")

    # BraTS24 GLI: 1=NETC, 2=SNFH, 3=ET, 4=RC.
    regions = {
        "netc": seg == 1,
        "snfh": seg == 2,
        "et": seg == 3,
        "rc": seg == 4,
    }
    regions["tc"] = regions["netc"] | regions["et"]
    regions["wt"] = regions["netc"] | regions["snfh"] | regions["et"]
    binary_header = seg_img.header.copy()
    binary_header.set_data_dtype(np.uint8)
    for name, region in regions.items():
        nib.save(
            nib.Nifti1Image(region.astype(np.uint8), seg_img.affine, binary_header),
            output_dir / f"{subject.name}-{name}.nii.gz",
        )

    for input_token, output_token in MODALITIES.items():
        volume = nib.load(str(modality_path(subject, input_token))).get_fdata(dtype=np.float32)
        save_views(normalize(center_crop3d(volume)), output_dir, f"{subject.name}_{output_token}")


def make_folds(subjects: list[Path], seed: int) -> dict[str, list[Path]]:
    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    folds = {name: [] for name in FOLD_NAMES}
    for index, subject in enumerate(shuffled):
        folds[FOLD_NAMES[index % len(FOLD_NAMES)]].append(subject)
    return folds


def main() -> None:
    args = parse_args()
    input_groups = training_groups(args.input, not args.exclude_additional)
    subjects = []
    source_by_subject = {}
    for input_group in input_groups:
        group_subjects = collect_subjects(input_group)
        print(f"Found {len(group_subjects)} usable subjects in {input_group.name}")
        for subject in group_subjects:
            if subject.name in source_by_subject:
                raise ValueError(f"Duplicate subject ID across training folders: {subject.name}")
            subjects.append(subject)
            source_by_subject[subject.name] = input_group.name

    if not subjects:
        raise RuntimeError(f"No usable BraTS24 subjects found under {args.input}")

    folds = make_folds(subjects, args.seed)
    output_train = args.output / "BraTS2024TrainingData"
    output_train.mkdir(parents=True, exist_ok=True)
    (output_train / "folds_manifest.json").write_text(
        json.dumps({fold: [subject.name for subject in items] for fold, items in folds.items()}, indent=2) + "\n"
    )
    (output_train / "source_manifest.json").write_text(json.dumps(source_by_subject, indent=2) + "\n")

    print(f"\nCombined dataset: {len(subjects)} subjects")
    for fold, fold_subjects in folds.items():
        print(f"Fold {fold}: {len(fold_subjects)} subjects")
        for index, subject in enumerate(fold_subjects, start=1):
            process_subject(subject, output_train / fold / subject.name)
            if index % 25 == 0 or index == len(fold_subjects):
                print(f"  {index}/{len(fold_subjects)}")
    print(f"Output: {output_train}")


if __name__ == "__main__":
    main()