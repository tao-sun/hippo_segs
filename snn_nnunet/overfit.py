"""Deterministic, deliberately overlapping volume-level BraTS diagnostics.

Selection reads training-label provenance and case properties before any dataset
is instantiated. Existing shared preprocessing is reused without modification;
``stage_selected_data`` confines native unpacking to the selected cases.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import random
import tempfile

import numpy as np


SPLITS = ('train', 'validation', 'test')
REGIONS = ('WT', 'TC', 'ET')
_ARTIFACT_SUFFIXES = ('.npz', '.npy', '_seg.npy', '.pkl', '.b2nd', '_seg.b2nd')


def _write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each rank can safely persist the same deterministic selection concurrently.
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(document, stream, indent=2)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _subjects(identifiers):
    result = tuple(sorted(str(identifier) for identifier in identifiers))
    if not result or len(result) != len(set(result)):
        raise ValueError('Overfit subjects must be nonempty and unique')
    if any(identifier.startswith('.') or Path(identifier).name != identifier for identifier in result):
        raise ValueError('Overfit subject identifiers must be visible file basenames')
    return result


def _training_labels(trainer, available):
    """Find training-label provenance overlapping the available preprocessing.

    A broken labelsTr symlink still records a training label identifier. Its
    mask can be read from the existing preprocessing, without following the
    missing raw target or admitting arbitrary image/test identifiers.
    """
    ending = trainer.dataset_json.get('file_ending', '.nii.gz')
    raw = getattr(trainer, 'raw_dataset_folder', None)
    if raw is None:
        from nnunetv2.paths import nnUNet_raw
        dataset_name = getattr(trainer.plans_manager, 'dataset_name', None)
        raw_is_set = nnUNet_raw.is_set() if hasattr(nnUNet_raw, 'is_set') else bool(nnUNet_raw)
        if raw_is_set and dataset_name:
            raw_root = nnUNet_raw.require() if hasattr(nnUNet_raw, 'require') else nnUNet_raw
            raw = Path(raw_root) / dataset_name
    if raw is not None:
        labels = Path(raw) / 'labelsTr'
        if labels.is_dir():
            cases = {
                path.name[:-len(ending)]: path for path in sorted(labels.iterdir())
                if (path.is_file() or path.is_symlink())
                and not path.name.startswith('.') and path.name.endswith(ending)
            }
            if set(cases).intersection(available):
                return cases, f'labelsTr:{labels.resolve()}'
    mapping = trainer.dataset_json.get('dataset')
    if isinstance(mapping, dict):
        cases = {}
        for identifier, details in mapping.items():
            if (not str(identifier).startswith('.') and isinstance(details, dict)
                    and details.get('label')):
                label = Path(os.path.expandvars(str(details['label'])))
                if not label.is_absolute() and raw is not None:
                    label = Path(raw) / label
                cases[str(identifier)] = label
        if set(cases).intersection(available):
            return cases, 'dataset.json:dataset'
    # nnU-Net copies only its labeled training masks into gt_segmentations.
    base = Path(getattr(trainer, 'preprocessed_dataset_folder_base',
                        Path(trainer.preprocessed_dataset_folder).parent))
    labels = base / 'gt_segmentations'
    if labels.is_dir():
        cases = {path.name[:-len(ending)]: path for path in sorted(labels.iterdir())
                 if (path.is_file() or path.is_symlink())
                 and not path.name.startswith('.') and path.name.endswith(ending)}
        if set(cases).intersection(available):
            return cases, f'gt_segmentations:{labels.resolve()}'
    # Native preprocessing writes class_locations only when a training label
    # was supplied. Preserve that provenance when raw-mask retention was cleaned.
    folder = Path(trainer.preprocessed_dataset_folder)
    cases = {}
    for identifier in sorted(available):
        if identifier.startswith('.') or Path(identifier).name != identifier:
            continue
        properties = folder / f'{identifier}.pkl'
        if not properties.is_file():
            continue
        with properties.open('rb') as stream:
            locations = pickle.load(stream).get('class_locations')
        if not isinstance(locations, dict) or not locations:
            continue
        paired = any((folder / f'{identifier}{suffix}').is_file()
                     for suffix in ('_seg.npy', '_seg.b2nd'))
        archive = folder / f'{identifier}.npz'
        if not paired and archive.is_file():
            with np.load(archive) as contents:
                paired = 'seg' in contents.files
        if paired:
            # This sentinel deliberately is not a raw file; selection reads seg
            # from the actual native storage format before any dataset instance.
            cases[identifier] = folder / '.raw_labels_not_retained' / identifier
    if cases:
        return cases, f'preprocessed_labeled_training:{folder.resolve()}'
    raise ValueError('Cannot establish labeled training provenance: labelsTr, dataset.json dataset labels, '
                     'gt_segmentations, or native class_locations with segmentation artifacts are required')


def _read_label(trainer, path, label_reader):
    if label_reader is not None:
        result = label_reader(path)
    else:
        reader = trainer.plans_manager.image_reader_writer_class()
        result = reader.read_seg(str(path))
    # Native image readers return (array, properties); injected readers may return an array.
    array = np.asarray(result[0] if isinstance(result, tuple) else result)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or not np.isin(array, (0, 1, 2, 3, 4)).all():
        raise ValueError(f'Invalid BraTS training labelmap: {path}')
    return array


def _property_presence(folder, identifier):
    """nnU-Net foreground locations preserve emptiness, but are not voxel counts."""
    path = Path(folder) / f'{identifier}.pkl'
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        locations = pickle.load(stream).get('class_locations', {})
    et, tc = None, None
    for key, values in locations.items():
        if isinstance(key, (int, np.integer)):
            key = (int(key),)
        elif isinstance(key, (tuple, list)):
            key = tuple(int(value) for value in key)
        else:
            continue
        if set(key) == {3}:
            et = len(values) > 0
        if set(key) == {1, 3}:
            tc = len(values) > 0
    if et is not None and tc is not None:
        return et, tc
    return None


def _read_preprocessed_seg(folder, identifier):
    folder = Path(folder)
    if (folder / f'{identifier}_seg.npy').is_file():
        return np.load(folder / f'{identifier}_seg.npy', mmap_mode='r')
    if (folder / f'{identifier}.npz').is_file():
        with np.load(folder / f'{identifier}.npz') as data:
            return data['seg']
    if (folder / f'{identifier}_seg.b2nd').is_file():
        import blosc2
        return np.asarray(blosc2.open(str(folder / f'{identifier}_seg.b2nd'), mode='r'))
    raise ValueError(f'Missing preprocessed training segmentation for {identifier}')


def select_overfit_subjects(trainer, count, seed, *, label_reader=None):
    """Persist exactly ``count`` unique training IDs with seeded ET/TC preference.

    Case properties determine ET/TC presence without loading every image volume.
    Selected available raw masks are verified before callers construct datasets.
    Reusing a run with a different seed, count, pool, or selection is rejected.
    ``label_reader(path)`` is an optional injection returning a 3D/1x3D array.
    """
    if type(count) is not int or count <= 0:
        raise ValueError('Overfit count must be a positive integer')
    if type(seed) is not int or seed < 0:
        raise ValueError('Overfit seed must be a nonnegative integer')
    available = set(trainer.dataset_class.get_identifiers(str(trainer.preprocessed_dataset_folder)))
    labels, provenance = _training_labels(trainer, available)
    pool = sorted(identifier for identifier in labels if identifier in available and not identifier.startswith('.'))
    if len(pool) < count:
        raise ValueError(f'Overfit count {count} exceeds {len(pool)} available labeled training volumes')
    manifest_path = Path(trainer.output_folder) / 'overfit_subjects.json'
    saved = None
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (saved.get('count') != count or saved.get('seed') != seed
                or saved.get('training_pool') != pool or saved.get('provenance') != provenance):
            raise ValueError('Overfit manifest count, seed, or labeled training pool mismatch')
    preferred, fallback = [], []
    for identifier in pool:
        presence = _property_presence(trainer.preprocessed_dataset_folder, identifier)
        if presence is None:
            if labels[identifier].is_file():
                seg = _read_label(trainer, labels[identifier], label_reader)
            else:
                seg = _read_preprocessed_seg(trainer.preprocessed_dataset_folder, identifier)
            presence = (bool((seg == 3).any()), bool(np.isin(seg, (1, 3)).any()))
        (preferred if all(presence) else fallback).append(identifier)
    rng = random.Random(seed)
    rng.shuffle(preferred)
    rng.shuffle(fallback)
    selected = _subjects((preferred + fallback)[:count])
    if saved is not None:
        if (saved.get('version') != 1 or saved.get('subjects') != list(selected)
                or saved.get('splits') != {split: list(selected) for split in SPLITS}
                or saved.get('intentional_split_overlap') is not True):
            raise ValueError('Overfit manifest subjects or splits differ from the deterministic selection')
    for identifier in selected:
        if labels[identifier].is_file():
            seg = _read_label(trainer, labels[identifier], label_reader)
        else:
            seg = _read_preprocessed_seg(trainer.preprocessed_dataset_folder, identifier)
        array = np.asarray(seg)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if (array.ndim != 3 or not np.isin(array, (-1, 0, 1, 2, 3, 4)).all()
                or not np.isin(array, (0, 1, 2, 3, 4)).any()):
            raise ValueError(f'Invalid or unlabeled preprocessed training segmentation for {identifier}')
        if identifier in preferred and not ((seg == 3).any() and np.isin(seg, (1, 3)).any()):
            raise ValueError(f'Case properties and training ET/TC masks disagree for {identifier}')
    manifest = {
        'version': 1, 'count': count, 'seed': seed, 'subjects': list(selected),
        'training_pool': pool, 'provenance': provenance,
        'selection_preference': 'nonempty ET (label 3) and TC (labels 1/3)',
        'preferred_subjects': sorted(identifier for identifier in selected if identifier in preferred),
        'splits': {split: list(selected) for split in SPLITS},
        'intentional_split_overlap': True,
    }
    if saved is None:
        _write_json(manifest_path, manifest)
    return selected


def stage_selected_data(source, destination, subject_ids):
    """Symlink only selected preprocessing artifacts into a run-local folder.

    The source identity is recorded beside (not inside) the folder because
    nnU-Net infers its storage format from all file extensions inside it.
    Locally generated selected-case numpy caches are allowed on subsequent calls.
    """
    source, destination = Path(source).resolve(), Path(destination)
    identifiers = _subjects(subject_ids)
    if source == destination.resolve():
        raise ValueError('Overfit staging destination must differ from shared preprocessing source')
    artifacts = {}
    for identifier in identifiers:
        names = [identifier + suffix for suffix in _ARTIFACT_SUFFIXES]
        found = [name for name in names if (source / name).is_file()]
        has_images = any(name in found for name in (identifier + '.npz', identifier + '.b2nd', identifier + '.npy'))
        has_seg = any(name in found for name in (identifier + '.npz', identifier + '_seg.b2nd', identifier + '_seg.npy'))
        if not has_images or not has_seg or identifier + '.pkl' not in found:
            raise ValueError(f'Incomplete preprocessed source artifacts for {identifier}')
        for name in found:
            stat = (source / name).stat()
            artifacts[name] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                               'inode': stat.st_ino, 'device': stat.st_dev}
    metadata = {'source': str(source), 'subjects': list(identifiers), 'artifacts': artifacts}
    manifest_path = destination.parent / f'{destination.name}_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != metadata:
        raise ValueError('Overfit staging source or subject selection changed')
    destination.mkdir(parents=True, exist_ok=True)
    allowed = {identifier + suffix for identifier in identifiers for suffix in _ARTIFACT_SUFFIXES}
    for path in destination.iterdir():
        if path.name not in allowed:
            raise ValueError(f'Unexpected artifact in overfit staging folder: {path.name}')
        if path.name in artifacts:
            if not path.is_symlink() or path.resolve() != source / path.name:
                raise ValueError(f'Unexpected non-source artifact in overfit staging folder: {path.name}')
        elif not (path.name.endswith('.npy') and not path.is_symlink()
                  and any(path.name == identifier + suffix and identifier + '.npz' in artifacts
                          for identifier in identifiers for suffix in ('.npy', '_seg.npy'))):
            raise ValueError(f'Unexpected generated overfit artifact: {path.name}')
    for name in artifacts:
        path = destination / name
        try:
            path.symlink_to(source / name)
        except FileExistsError:
            if not path.is_symlink() or path.resolve() != source / name:
                raise ValueError(f'Overfit staging artifact changed: {path}')
    _write_json(manifest_path, metadata)
    return destination


def assert_overfit_splits(datasets, identifiers):
    """Assert deliberate train=validation=test identity and absence of extra cases."""
    expected = _subjects(identifiers)
    assert set(datasets) == set(SPLITS), 'Overfit requires train, validation, and test datasets'
    assert len({id(dataset) for dataset in datasets.values()}) == 3, 'Overfit requires distinct dataset instances'
    for split, dataset in datasets.items():
        actual = tuple(str(value) for value in dataset.identifiers)
        assert len(actual) == len(expected) and len(set(actual)) == len(expected), f'{split} must contain N unique subjects'
        assert tuple(sorted(actual)) == expected, f'{split} contains subjects outside the selected overfit set'


def make_overfit_datasets(trainer, identifiers):
    """Construct three independent read-only views of the same selected subjects."""
    identifiers = _subjects(identifiers)
    datasets = {split: trainer.dataset_class(
        str(trainer.preprocessed_dataset_folder), list(identifiers),
        folder_with_segs_from_previous_stage=trainer.folder_with_segs_from_previous_stage,
    ) for split in SPLITS}
    assert_overfit_splits(datasets, identifiers)
    return datasets


def _unwrap_network(network):
    import torch
    if isinstance(network, torch.nn.parallel.DistributedDataParallel):
        network = network.module
    return network


def evaluate_overfit_volumes(trainer, identifiers, *, epoch=None, output_dir=None, predictor_class=None):
    """Evaluate complete volumes once, reusing identical predictions in all splits.

    Uses the configured validation patch and native tile traversal in FP32, without
    mirroring, postprocessing, or region canonicalization. The network's mode
    and Python/NumPy/Torch RNG are restored even when inference fails. Recurrent
    states are reset before every case and on exit, without changing tile order.
    """
    import torch
    from snn_nnunet.custom_full_validation import (
        dice_metrics, labelmap_to_regions, temporary_patch_size, threshold_probabilities,
    )
    from snn_nnunet.empty_dice_evaluation import DICE_CONVENTION
    from snn_nnunet.paper_unet import _reset_neurons
    from snn_nnunet.resume_validation import _preserve_training_state

    identifiers = _subjects(identifiers)
    patch_size = tuple(getattr(getattr(trainer, 'snn_config', None),
                               'validation_patch_size', trainer.configuration_manager.patch_size))
    if predictor_class is None:
        from snn_nnunet.inference import FP32nnUNetPredictor
        predictor_class = FP32nnUNetPredictor
    epoch = trainer.current_epoch if epoch is None else epoch
    destination = Path(output_dir) if output_dir is not None else Path(trainer.output_folder) / 'overfit_validation' / f'epoch_{epoch:04d}'
    cases = []
    network = _unwrap_network(trainer.network)
    geometry = (temporary_patch_size(trainer.configuration_manager, patch_size)
                if hasattr(trainer.configuration_manager, 'configuration') else nullcontext())
    with _preserve_training_state(trainer), geometry, torch.no_grad():
        try:
            datasets = make_overfit_datasets(trainer, identifiers)
            network.eval()
            predictor = predictor_class(
                tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                perform_everything_on_device=True, device=trainer.device,
                verbose=False, verbose_preprocessing=False, allow_tqdm=False,
            )
            predictor.manual_initialization(
                network, trainer.plans_manager, trainer.configuration_manager, None,
                trainer.dataset_json, trainer.__class__.__name__, None,
            )
            for identifier in identifiers:
                _reset_neurons(network)
                data, seg, seg_prev, _ = datasets['train'].load_case(identifier)
                if seg is None:
                    raise ValueError(f'Missing reference segmentation for {identifier}')
                data = np.asarray(data, dtype=np.float32)
                if trainer.is_cascaded:
                    from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
                    data = np.vstack((data, convert_labelmap_to_one_hot(
                        np.asarray(seg_prev), trainer.label_manager.foreground_labels,
                        output_dtype=np.float32)))
                reference = np.asarray(seg)
                if reference.ndim == 4 and reference.shape[0] == 1:
                    reference = reference[0]
                reference = labelmap_to_regions(reference)
                logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(np.ascontiguousarray(data)))
                logits = logits.detach().float().cpu()
                if logits.ndim != 4 or logits.shape[0] != 3:
                    raise ValueError(f'Expected 3 WT/TC/ET logits for {identifier}; got {tuple(logits.shape)}')
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f'Non-finite overfit logits for {identifier}')
                prediction = threshold_probabilities(torch.sigmoid(logits).numpy(), 0.5)
                metrics = dice_metrics(reference, prediction)
                cases.append({'case': identifier, 'metrics': metrics})
        finally:
            _reset_neurons(network)
    mean = {region: float(np.mean([case['metrics'][region]['dice'] for case in cases])) for region in REGIONS}
    mean['mean'] = float(np.mean(list(mean.values())))
    split_report = {'subjects': list(identifiers), 'cases': cases, 'mean': mean}
    summary = {
        'epoch': int(epoch), 'subjects': list(identifiers), 'count': len(identifiers),
        'intentional_split_overlap': True, 'diagnostic_only': True, 'prediction_reuse': True,
        'prediction_reuse_reason': 'All three distinct dataset instances contain identical volumes with no transforms; one deterministic prediction per subject is reused.',
        'splits': {split: deepcopy(split_report) for split in SPLITS},
        'dice_convention': dict(DICE_CONVENTION),
        'settings': {
            'precision': 'float32', 'patch_size': [int(value) for value in patch_size],
            'tile_step_size': 0.5, 'gaussian_blending': True, 'mirroring': False,
            'probability_threshold': 0.5, 'postprocessing': False, 'region_canonicalization': False,
        },
    }
    _write_json(destination / 'summary.json', summary)
    log = trainer.print_to_log_file
    for split in SPLITS:
        log(f"Overfit {split} mean Dice: " + ' '.join(f'{region}={mean[region]:.6f}' for region in REGIONS))
        for case in cases:
            log(f"Overfit {split} {case['case']}: " + ' '.join(
                f"{region} GT={case['metrics'][region]['gt_voxels']} pred={case['metrics'][region]['pred_voxels']}"
                for region in ('ET', 'TC')))
    return summary
