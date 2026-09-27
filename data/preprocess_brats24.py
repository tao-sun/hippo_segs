#!/usr/bin/env python3
"""Preprocess BraTS24 directories or ZIPs into versioned float32/uint8 caches or legacy PNGs."""

from __future__ import annotations

import argparse
import json
import hashlib
import pickle
import random
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brats_preprocessing import discover_subjects, normalize_foreground_zscore

import nibabel as nib
import numpy as np

from PIL import Image


TARGET_SHAPE = (160, 192, 152)
FOLD_NAMES = ("1", "2", "3", "4", "5")
MODALITIES = {
    "t1n": "t1",
    "t1c": "t1ce",
    "t2w": "t2",
    "t2f": "flair",
}
VALID_LABELS = {0, 1, 2, 3, 4}


def parse_boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected 'true' or 'false'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/home/aurora/data/BRATS2024"))
    parser.add_argument("--output", type=Path, default=Path("/home/aurora/data/BRATS2024_preprocessed"))
    parser.add_argument("--input-mode", choices=("auto", "archives", "directories"), default="auto",
                        help="auto prefers ZIP archives over extracted copies in mixed input directories")
    parser.add_argument("--exclude-validation", nargs="?", const=True, default=False, type=parse_boolean,
                        help="Skip the official held-out validation set")
    parser.add_argument("--cache-root", type=Path, help="Write subject/view .pt caches directly")
    parser.add_argument("--preprocessing-normalization", choices=("minmax", "zscore"), default="minmax")
    parser.add_argument("--preserve-float32", nargs="?", const=True, default=False, type=parse_boolean,
                        help="Required automatically by zscore")
    parser.add_argument("--view", nargs="+", choices=("axial", "sagittal", "coronal"), default=["axial", "sagittal", "coronal"])
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--diagnostic-subjects", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=[1], default=1, help="One subject at a time bounds scratch usage")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--exclude-additional",
        nargs="?",
        const=True,
        default=False,
        type=parse_boolean,
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
            Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(
                view_dir / f"Brats17_{stem}_{view}_{index:03d}.png"
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
    raw = np.asanyarray(nib.load(str(seg_path)).dataobj)
    if not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all():
        raise ValueError(f"Non-finite or non-integer segmentation in {seg_path}")
    values = set(np.unique(raw).astype(int).tolist())
    unexpected = values - VALID_LABELS
    if unexpected:
        raise ValueError(f"Unexpected labels {sorted(unexpected)} in {seg_path}")


def process_subject(subject: Path, output_dir: Path, *, normalization="minmax",
                    cache_root=None, fold=1, views=("axial", "sagittal", "coronal"),
                    diagnostics=False, split="training"):
    """The same spatial/intensity/GT pipeline for extracted and ZIP subjects."""
    if normalization not in {"minmax", "zscore"}:
        raise ValueError("Unknown preprocessing normalization")
    if normalization == "zscore" and cache_root is None:
        raise ValueError("zscore requires float32 .pt cache output")
    seg_path = next(subject.glob("*-seg.nii.gz"), None)
    if seg_path is None and split != "validation":
        raise ValueError(f"Missing segmentation for training subject {subject.name}")
    seg = None
    volumes = None
    reference_img = nib.load(str(seg_path or modality_path(subject, "t1n")))
    if seg_path is not None:
        validate_segmentation(seg_path)
        seg = center_crop3d(reference_img.get_fdata(dtype=np.float32)).astype(np.uint8)
        regions = {"netc": seg == 1, "snfh": seg == 2, "et": seg == 3, "rc": seg == 4}
        regions["tc"] = regions["netc"] | regions["et"]
        regions["wt"] = regions["netc"] | regions["snfh"] | regions["et"]
        volumes = {f"{name}_voxels": int(regions[name].sum()) for name in ("et", "tc", "wt")}
        if cache_root is None:
            output_dir.mkdir(parents=True, exist_ok=True)
            header = reference_img.header.copy()
            header.set_data_dtype(np.int16)
            nib.save(nib.Nifti1Image(seg.astype(np.int16), reference_img.affine, header),
                     output_dir / f"{subject.name}-seg.nii.gz")
            for name, region in regions.items():
                header = reference_img.header.copy()
                header.set_data_dtype(np.uint8)
                nib.save(nib.Nifti1Image(region.astype(np.uint8), reference_img.affine, header),
                         output_dir / f"{subject.name}-{name}.nii.gz")
    modalities = []
    for input_token, output_token in MODALITIES.items():
        img = nib.load(str(modality_path(subject, input_token)))
        if img.shape != reference_img.shape or not np.allclose(img.affine, reference_img.affine):
            raise ValueError(f"Spatial mismatch for {subject.name}: {input_token}")
        raw = center_crop3d(img.get_fdata(dtype=np.float32))
        volume = normalize_foreground_zscore(raw) if normalization == "zscore" else normalize(raw)
        if diagnostics:
            mask = (raw != 0) & np.isfinite(raw)
            before, after = raw[mask], volume[mask]
            print(f"  {output_token}: foreground before mean/std="
                  f"{before.mean() if before.size else 0:.5g}/{before.std() if before.size else 0:.5g}; "
                  f"after={after.mean() if after.size else 0:.5g}/{after.std() if after.size else 0:.5g}; "
                  f"dtype={volume.dtype}; min/max={volume.min():.5g}/{volume.max():.5g}")
        if cache_root is None:
            save_views(volume, output_dir, f"{subject.name}_{output_token}")
        else:
            # Quantization belongs exclusively to the reproducible baseline.
            modalities.append(volume if normalization == "zscore" else np.rint(volume * 255).astype(np.uint8))
    if cache_root is not None:
        from snn_fptt import take_view, subject_cache_path, write_subject_cache_file
        image_volume = np.stack(modalities)
        for view in views:
            write_subject_cache_file(
                subject_cache_path(cache_root, view, fold, subject.name), subject.name,
                view, take_view(image_volume, view), seg,
                normalization=normalization, lesion_volumes=volumes, label_format="brats24",
                split=split, xyz=tuple(image_volume.shape[1:]))
    return volumes


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def preprocess(input_path, output, *, normalization="minmax", views=("axial",),
               temp_dir=None, seed=2025, include_additional=True, diagnostics=0,
               cache=True, include_validation=True, input_mode="auto"):
    """Restartable subject-at-a-time preprocessing, with no full raw extraction."""
    sources = discover_subjects(input_path, include_additional,
                                include_validation=include_validation, input_mode=input_mode)
    supervised = [source for source in sources if source.split == "training"]
    validation = [source for source in sources if source.split == "validation"]
    print(f"Training subjects: {len(supervised)}; official validation subjects: {len(validation)}")
    output = Path(output)
    if normalization == "zscore" and not cache:
        raise ValueError("zscore requires float32 cache output")
    config = {"preprocessing_version": 2, "normalization": normalization,
              "preserve_float32": normalization == "zscore", "shape": list(TARGET_SHAPE),
              "views": sorted(views) if cache else ["axial", "coronal", "sagittal"],
              "seed": seed, "cache": cache, "label_format": "brats24",
              "subjects": [source.subject_id for source in supervised]}
    if validation:
        config["validation_subjects"] = [source.subject_id for source in validation]
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = output / "preprocessing.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Incompatible preprocessing configuration in {output}; use a separate output directory")
    # Unversioned existing caches may not be silently mixed with the new run.
    if not config_path.exists() and output.exists() and any(output.iterdir()):
        raise ValueError(f"Existing output lacks preprocessing metadata: {output}; use a separate output directory")
    _atomic_json(config_path, config)
    shuffled = list(supervised)
    random.Random(seed).shuffle(shuffled)
    folds = {name: [] for name in FOLD_NAMES}
    for index, source in enumerate(shuffled):
        folds[FOLD_NAMES[index % 5]].append(source.subject_id)
    for root in ([output / view for view in views] if cache else [output]):
        _atomic_json(root / "folds_manifest.json", folds)
        _atomic_json(root / "validation_manifest.json", {"subjects": [source.subject_id for source in validation]})
    work = {**folds, "validation": [source.subject_id for source in validation]}
    by_id = {source.subject_id: source for source in sources}
    created = skipped = 0
    start = time.monotonic()
    for fold, ids in work.items():
        for sid in ids:
            marker = output / "completion" / fold / f"{sid}.json"
            complete = False
            if marker.exists():
                try:
                    record = json.loads(marker.read_text())
                    complete = record["config_id"] == config_id
                    if complete and cache:
                        from snn_fptt import load_subject_cache_file, subject_cache_path
                        for view in views:
                            load_subject_cache_file(subject_cache_path(output, view, fold, sid),
                                                    sid, view, expected_normalization=normalization,
                                                    allow_unlabeled=fold == "validation")
                    elif complete:
                        from PIL import Image
                        subject_out = output / fold / sid
                        for token in (("seg", "netc", "snfh", "et", "rc", "tc", "wt") if "seg" in by_id[sid].files else ()):
                            data = nib.load(str(subject_out / f"{sid}-{token}.nii.gz")).get_fdata()
                            if data.shape != TARGET_SHAPE:
                                raise ValueError("Incomplete segmentation output")
                        for view, count in zip(("sagittal", "coronal", "axial"), TARGET_SHAPE):
                            for modality in MODALITIES.values():
                                for index in range(count):
                                    with Image.open(subject_out / view / f"Brats17_{sid}_{modality}_{view}_{index:03d}.png") as image:
                                        image.verify()
                except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError):
                    complete = False
            if complete:
                skipped += 1
                print(f'[{created+skipped}/{len(sources)}] {sid}: skipped (verified)')
                continue
            marker.unlink(missing_ok=True)
            print(f'[{created+skipped+1}/{len(sources)}] {sid}: extracting/preprocessing/saving...')
            with by_id[sid].materialize(temp_dir) as subject:
                volumes = process_subject(subject, output / fold / sid, normalization=normalization,
                                          cache_root=output if cache else None, fold=fold, views=views,
                                          diagnostics=created < diagnostics, split=by_id[sid].split)
            if cache:
                from snn_fptt import load_subject_cache_file, subject_cache_path
                for view in views:
                    load_subject_cache_file(subject_cache_path(output, view, fold, sid),
                                            sid, view, expected_normalization=normalization,
                                                    allow_unlabeled=fold == "validation")
            completion = {"config_id": config_id, "subject_id": sid}
            if volumes is not None:
                completion["lesion_volumes"] = volumes
            _atomic_json(marker, completion)
            created += 1
            print(f'  done; elapsed {time.monotonic()-start:.1f}s')
    return {"created": created, "skipped": skipped, "subjects": len(sources)}


def make_folds(subjects: list[Path], seed: int) -> dict[str, list[Path]]:
    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    folds = {name: [] for name in FOLD_NAMES}
    for index, subject in enumerate(shuffled):
        folds[FOLD_NAMES[index % len(FOLD_NAMES)]].append(subject)
    return folds


def main() -> None:
    args = parse_args()
    if args.preserve_float32 and args.preprocessing_normalization != "zscore":
        raise ValueError("preserve_float32 requires zscore; minmax reproduces uint8 baseline")
    cache = args.cache_root is not None or args.preprocessing_normalization == "zscore"
    output = args.cache_root or (args.output if cache else args.output / "BraTS2024TrainingData")
    summary = preprocess(args.input, output, normalization=args.preprocessing_normalization,
                         views=tuple(args.view), temp_dir=args.temp_dir, seed=args.seed,
                         include_additional=not args.exclude_additional,
                         diagnostics=args.diagnostic_subjects, cache=cache,
                         include_validation=not args.exclude_validation, input_mode=args.input_mode)
    print(f"Output: {output}; {summary}")


if __name__ == "__main__":
    main()
