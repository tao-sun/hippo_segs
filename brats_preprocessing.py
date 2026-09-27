"""Intensity normalization and bounded, safe BraTS24 input materialization."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import warnings
import zipfile

import numpy as np

REQUIRED_TOKENS = ('t1n', 't1c', 't2w', 't2f', 'seg')
FILE_PATTERN = re.compile(r'^(?P<sid>.+)-(?P<token>t1n|t1c|t2w|t2f|seg)\.nii\.gz$')


def normalize_foreground_zscore(volume: np.ndarray) -> np.ndarray:
    """Normalize finite nonzero voxels; invalid/background/constant voxels become zero.

    The source is never mutated. Accumulate statistics in float64 to avoid
    overflow/cancellation for finite float32 MRI values; storage stays float32.
    """
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f'Expected a 3D modality, got {array.shape}')
    mask = (array != 0) & np.isfinite(array)
    result = np.zeros_like(array, dtype=np.float32)
    if mask.any():
        foreground = array[mask].astype(np.float64)
        mean, std = foreground.mean(), foreground.std()
        if std > 1e-8:
            result[mask] = ((foreground - mean) / std).astype(np.float32)
    return result


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe ZIP member: {name}')


@dataclass
class SubjectSource:
    subject_id: str
    files: dict
    archive: Path | None = None
    split: str = "training"

    @contextmanager
    def materialize(self, temp_dir=None):
        if self.archive is None:
            yield Path(next(iter(self.files.values()))).parent
            return
        if temp_dir is not None:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='brats-', dir=temp_dir) as workspace:
            subject = Path(workspace) / self.subject_id
            subject.mkdir()
            with zipfile.ZipFile(self.archive) as archive:
                for token, name in self.files.items():
                    _safe_member(name)
                    # Destination is constructed from a validated basename, never
                    # from the ZIP path. Copy one member at a time, bounded RAM.
                    target = subject / f'{self.subject_id}-{token}.nii.gz'
                    with archive.open(name) as source, target.open('wb') as output:
                        shutil.copyfileobj(source, output, length=1024*1024)
            yield subject


def discover_subjects(input_path, include_additional=True, include_validation=False,
                      input_mode="auto"):
    """Discover supervised and optionally official validation subjects separately.

    Auto prefers ZIPs when present, avoiding duplicate extracted copies. Use
    input_mode=directories to select extracted input explicitly in mixed roots.
    """
    if input_mode not in {"auto", "archives", "directories"}:
        raise ValueError("input_mode must be auto, archives or directories")
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(root)
    sources = []

    def selected(path):
        return include_additional or 'additional' not in str(path).lower()

    def collect(entries, archive=None):
        groups = {}
        for entry in entries:
            name = str(entry)
            if archive is not None:
                _safe_member(name)
            if not selected(name):
                continue
            match = FILE_PATTERN.match(PurePosixPath(name).name)
            if match is None:
                continue
            sid, token = match['sid'], match['token']
            if sid in {'.', '..'} or '/' in sid or '\\' in sid or ':' in sid:
                raise ValueError(f'Unsafe subject ID: {sid}')
            # Distinct parent folders with the same ID must not be merged.
            key = (sid, str(PurePosixPath(name).parent))
            files = groups.setdefault(key, {})
            if token in files:
                raise ValueError(f'Duplicate file for {sid}: {token}')
            files[token] = entry
        for (sid, _parent), files in sorted(groups.items()):
            # Only recognized dataset path components classify a split. Do not
            # match arbitrary ancestor names such as a user's "validation_run".
            names = [part.lower() for part in PurePosixPath(_parent).parts]
            if archive is not None:
                names.append(archive.name.lower())
            official_validation = any(
                part in {"validation", "validation_data", "validationdata"}
                or part.endswith("validationdata.zip") for part in names)
            known_training = any(
                part in {"training_data1_v2", "training_data_additional", "trainingdata"}
                or part.endswith("trainingdata.zip") for part in names)
            missing = sorted(set(REQUIRED_TOKENS) - set(files))
            split = "validation" if official_validation else "training"
            if missing == ["seg"]:
                if not include_validation:
                    warnings.warn(f'Skipping {sid}; missing segmentation (unsupervised input)')
                    continue
                if known_training and not official_validation:
                    raise ValueError(f'{sid}: missing required files: seg in training input')
                split = "validation"
            elif missing:
                raise ValueError(f'{sid}: missing required files: {", ".join(missing)}')
            if split == "validation" and not include_validation:
                continue
            sources.append(SubjectSource(sid, files, archive, split))

    if root.is_file():
        if root.suffix.lower() != '.zip':
            raise ValueError(f'Expected directory or ZIP: {root}')
        archives = [root]
    else:
        archives = []
        if input_mode != "directories":
            archives = sorted(p for p in root.glob('*') if p.is_file() and p.suffix.lower() == '.zip' and selected(p))
            if not archives:
                archives = sorted(p for p in root.rglob('*.zip') if selected(p))
        if input_mode == "directories" or (input_mode == "auto" and not archives):
            collect(sorted(root.rglob('*.nii.gz')))
        elif archives:
            print(f'Using {len(archives)} ZIP archive(s); extracted copies are ignored')
    for index, path in enumerate(archives, 1):
        if not selected(path):
            continue
        before = len(sources)
        with zipfile.ZipFile(path) as archive:
            collect(archive.namelist(), archive=path)
        print(f'Archive {index}/{len(archives)}: {path.name}; found {len(sources)-before} subjects')
    seen = set()
    for source in sources:
        if source.subject_id in seen:
            raise ValueError(f'Duplicate subject ID: {source.subject_id}')
        seen.add(source.subject_id)
    if not sources:
        raise ValueError(f'No eligible BraTS24 subjects found in {root}')
    # Keep the historical primary-then-additional fold assignment, independent
    # of archive names/root depth and directory enumeration order.
    return sorted(sources, key=lambda source: (
        source.split == 'validation',
        'additional' in (str(source.archive or '') + ' '.join(map(str, source.files.values()))).lower(),
        source.subject_id))
