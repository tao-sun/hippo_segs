"""Training-only patient lesion statistics and epoch-deterministic sampling."""
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset, WeightedRandomSampler

CLASSES = ('ET', 'TC', 'WT')


def validate_sampling_options(classes=('ET', 'TC'), gamma=0.5, max_weight=4.0):
    if not isinstance(classes, (list, tuple)) or not classes or len(set(classes)) != len(classes) or any(c not in CLASSES for c in classes):
        raise ValueError('lesion_sampling_classes must contain unique ET/TC/WT names')
    for value, name, minimum in [(gamma, 'gamma', 0), (max_weight, 'max_weight', 1)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
            raise ValueError(f'lesion_sampling_{name} must be finite and >= {minimum}')


def volume_sampling_weight(volume, reference_size, gamma=0.5, max_weight=4.0):
    validate_sampling_options(gamma=gamma, max_weight=max_weight)
    if not math.isfinite(volume) or not math.isfinite(reference_size):
        raise ValueError('Lesion volumes/reference sizes must be finite')
    if volume <= 0 or reference_size <= 0:
        return 1.0
    # Log space avoids overflow before the cap for extreme configurable gamma.
    return math.exp(min(math.log(max_weight), max(0.0, gamma * (math.log(reference_size) - math.log(volume)))))


def compute_patient_weights(records, classes=('ET', 'TC'), gamma=0.5, max_weight=4.0):
    """The caller supplies ONLY the actual training subset, never validation GT."""
    validate_sampling_options(classes, gamma, max_weight)
    if not records:
        raise ValueError('Cannot sample an empty training dataset')
    stats = {}
    for name in classes:
        values = [record[f'{name.lower()}_voxels'] for record in records]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Lesion voxel counts must be nonnegative and finite')
        positive = [v for v in values if v > 0]
        stats[name] = {'positive_patients': len(positive),
                       'reference_size': float(np.median(positive)) if positive else 0.0}
    weights = [max(volume_sampling_weight(record[f'{name.lower()}_voxels'], stats[name]['reference_size'], gamma, max_weight)
                   for name in classes) for record in records]
    return weights, stats


class PatientWeightedSampler(WeightedRandomSampler):
    """One global patient sequence; Accelerator alone shards its batches.

    Exactly N replacement draws per epoch. Epoch seeding also reproduces an
    epoch after checkpoint resume without depending on worker RNG consumption.
    """
    def __init__(self, weights, seed=2025):
        if not weights or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError('Patient weights must be positive and finite')
        super().__init__(weights, len(weights), replacement=True, generator=torch.Generator())
        self.seed, self.epoch = int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        return super().__iter__()


def collect_training_lesion_records(dataset):
    """Read each selected patient's GT once, persisting small validated sidecars.

    Cache/source mtime and size plus label convention invalidate old counts.
    Subsets are traversed before any GT is accessed, including overfit mode.
    """
    from snn_fptt import brats_to_multilabel, subject_cache_path, load_subject_cache_file
    import nibabel as nib

    def patient(ds, index):
        if isinstance(ds, ConcatDataset):
            child = int(np.searchsorted(ds.cumulative_sizes, index, side='right'))
            offset = 0 if child == 0 else ds.cumulative_sizes[child-1]
            return patient(ds.datasets[child], index-offset)
        if isinstance(ds, Subset) or (
            hasattr(ds, 'dataset') and hasattr(ds, 'indices')
            and not hasattr(ds, 'subjects')
        ):
            return patient(ds.dataset, ds.indices[index])
        required = (
            'subjects', 'cache_root', 'view', 'fold', 'cache_required',
            'label_format', 'preprocessing_normalization',
        )
        if not all(hasattr(ds, attribute) for attribute in required):
            raise TypeError('Expected a BraTS patient dataset')
        subject = ds.subjects[index]
        cache_path = subject_cache_path(ds.cache_root, ds.view, ds.fold, subject.name) if ds.cache_root is not None else None
        if cache_path is not None and cache_path.exists():
            path = cache_path
            sidecar = path.with_suffix('.lesion_volumes.json')
        else:
            if ds.cache_required:
                raise FileNotFoundError(f'Required subject cache not found: {cache_path}')
            candidates = sorted(subject.glob('*_seg.nii*')) + sorted(subject.glob('*-seg.nii.gz'))
            if not candidates:
                raise FileNotFoundError(f'Missing segmentation: {subject}')
            path = candidates[0]
            sidecar = subject / 'lesion_volumes.json'
        stat = path.stat()
        fingerprint = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                       'label_format': ds.label_format, 'normalization': ds.preprocessing_normalization,
                       'version': 1}
        if sidecar.exists():
            try:
                record = json.loads(sidecar.read_text())
                if record['source'] == fingerprint:
                    counts = record['volumes']
                    if all(type(counts[f'{name.lower()}_voxels']) is int and counts[f'{name.lower()}_voxels'] >= 0 for name in CLASSES):
                        return {'subject_id': subject.name, **counts}
            except (ValueError, KeyError, OSError):
                pass
        label_format = ds.label_format
        counts = None
        if path == cache_path:
            payload = load_subject_cache_file(path, subject.name, ds.view, ds.preprocessing_normalization)
            stored_format = payload.get('label_format')
            if stored_format and label_format not in {'auto', stored_format}:
                raise ValueError('Requested label_format does not match cache')
            if stored_format and label_format == 'auto':
                label_format = stored_format
            if stored_format == label_format:
                counts = payload.get('lesion_volumes')
            seg = payload['segmentation'].numpy()
        else:
            seg = np.rint(nib.load(str(path)).get_fdata(dtype=np.float32)).astype(np.int16)
            # Match the dataset's fixed center crop, independent of GT extent.
            from snn_fptt import TARGET_SHAPE
            if seg.shape != TARGET_SHAPE:
                slices = tuple(slice((n-t)//2, (n-t)//2+t) for n,t in zip(seg.shape, TARGET_SHAPE))
                seg = seg[slices]
        if counts is None:
            masks = brats_to_multilabel(seg, label_format)
            counts = {f'{name.lower()}_voxels': int(masks[i].sum()) for i,name in enumerate(CLASSES)}
        record = {'source': fingerprint, 'volumes': counts}
        temporary = sidecar.with_name(f'.{sidecar.name}.{uuid.uuid4().hex}.tmp')
        try:
            temporary.write_text(json.dumps(record) + '\n')
            os.replace(temporary, sidecar)
        except OSError:
            # Read-only data is supported: counts remain in memory for this run.
            pass
        finally:
            temporary.unlink(missing_ok=True)
        return {'subject_id': subject.name, **counts}

    return [patient(dataset, index) for index in range(len(dataset))]


def sampling_summary(weights, stats, max_weight=4.0):
    values = np.asarray(weights)
    lines = ['Lesion-volume-aware sampling', f'Training patients: {len(weights)}']
    for name, record in stats.items():
        lines.append(f"{name}: positive patients={record['positive_patients']}; median positive volume={record['reference_size']:g}")
    lines.append(f'Patient sampling weights: mean={values.mean():.4g}; min={values.min():.4g}; max={values.max():.4g}')
    for low in (1, 2, 3):
        lines.append(f'{low}.0 <= w < {low+1}.0: {int(((values >= low) & (values < low+1)).sum())}')
    lines.append(f'w == {max_weight:g} (cap): {int((values == max_weight).sum())}')
    if max_weight > 4:
        lines.append(f'4.0 <= w < {max_weight:g}: {int(((values >= 4) & (values < max_weight)).sum())}')
    return '\n'.join(lines)
