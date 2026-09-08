#!/usr/bin/env python3
"""Build one lossless uint8 cache file per BraTS subject and view."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from tqdm import tqdm

import snn_fptt


def _cache_one_subject(
    subject_dir: Path,
    fold: int,
    view: str,
    cache_root: Path,
    overwrite: bool,
    existing_record: Optional[Dict],
) -> Tuple[str, bool, Dict]:
    cache_path = snn_fptt.subject_cache_path(
        cache_root,
        view,
        fold,
        subject_dir.name,
    )
    key = f"{fold}/{subject_dir.name}"
    skipped = existing_record is not None and not overwrite
    if skipped:
        return key, True, existing_record

    if not skipped:
        snn_fptt.build_subject_cache_file(
            subject_dir=subject_dir,
            fold=fold,
            view=view,
            cache_root=cache_root,
            overwrite=True,
        )

    payload = snn_fptt.load_subject_cache_file(
        cache_path,
        expected_subject_id=subject_dir.name,
        expected_view=view,
    )
    record = {
        "cache_file": str(cache_path.relative_to(cache_root / view)),
        "images_shape": list(payload["images"].shape),
        "segmentation_shape": list(payload["segmentation"].shape),
    }
    return key, skipped, record


def _write_manifest_atomic(path: Path, manifest: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _scan_view_cache(cache_root: Path, view: str) -> Tuple[Dict, Dict]:
    """Return manifest records for every valid cache currently on disk."""
    subjects = {}
    invalid_files = {}
    view_root = cache_root / view
    for cache_path in sorted(view_root.glob("*/*.pt")):
        relative_path = cache_path.relative_to(view_root)
        try:
            fold = int(cache_path.parent.name)
            if str(fold) not in snn_fptt.FOLD_NAMES:
                raise ValueError(f"invalid fold directory {cache_path.parent.name}")
            payload = snn_fptt.load_subject_cache_file(
                cache_path,
                expected_subject_id=cache_path.stem,
                expected_view=view,
            )
        except Exception as exc:
            invalid_files[str(relative_path)] = str(exc)
            continue

        subjects[f"{fold}/{cache_path.stem}"] = {
            "cache_file": str(relative_path),
            "images_shape": list(payload["images"].shape),
            "segmentation_shape": list(payload["segmentation"].shape),
        }
    return subjects, invalid_files


def build_cache(
    data_root: Path,
    cache_root: Path,
    view: str,
    folds: Iterable[int] = range(1, 6),
    workers: int = 4,
    overwrite: bool = False,
) -> Dict:
    """Build a restartable cache and return its manifest summary."""
    data_root = snn_fptt.ensure_train_root(Path(data_root).expanduser().resolve())
    cache_root = Path(cache_root).expanduser().resolve()
    folds = sorted(set(int(fold) for fold in folds))
    workers = int(workers)
    if view not in snn_fptt.VALID_VIEWS:
        raise ValueError(f"view must be one of {snn_fptt.VALID_VIEWS}")
    if not folds or any(str(fold) not in snn_fptt.FOLD_NAMES for fold in folds):
        raise ValueError("folds must contain values from 1 to 5")
    if workers <= 0:
        raise ValueError("workers must be a positive integer")

    subjects, invalid_files = _scan_view_cache(cache_root, view)
    tasks = []
    for fold in folds:
        for subject_dir in snn_fptt.find_subject_dirs(
            data_root,
            [fold],
            verbose=False,
        ):
            key = f"{fold}/{subject_dir.name}"
            tasks.append(
                (
                    subject_dir,
                    fold,
                    view,
                    cache_root,
                    overwrite,
                    subjects.get(key),
                )
            )
    if not tasks:
        raise RuntimeError(f"No subjects found under {data_root} for folds={folds}")

    if workers == 1:
        results = [
            _cache_one_subject(*task)
            for task in tqdm(tasks, desc=f"cache {view}")
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(
                tqdm(
                    executor.map(lambda task: _cache_one_subject(*task), tasks),
                    total=len(tasks),
                    desc=f"cache {view}",
                )
            )

    for key, _skipped, record in results:
        subjects[key] = record
        invalid_files.pop(record["cache_file"], None)
    manifest = {
        "format_version": snn_fptt.CACHE_FORMAT_VERSION,
        "view": view,
        "requested_folds": sorted(set(folds)),
        "num_subjects": len(subjects),
        "created": sum(not skipped for _key, skipped, _record in results),
        "skipped": sum(skipped for _key, skipped, _record in results),
        "invalid_files": invalid_files,
        "subjects": subjects,
    }
    _write_manifest_atomic(cache_root / view / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one uint8 .pt cache file per BraTS subject and view."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--view", required=True, choices=sorted(snn_fptt.VALID_VIEWS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(1, 6)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = build_cache(
        data_root=args.data_root,
        cache_root=args.cache_root,
        view=args.view,
        folds=args.folds,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(
        f"Cache complete: {summary['created']} created, "
        f"{summary['skipped']} skipped, {summary['num_subjects']} total"
    )


if __name__ == "__main__":
    main()
