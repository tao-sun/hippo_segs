"""Persist patient partitions and give each training run private nnU-Net metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from snn_nnunet.patient_partition import build_train_val_test_folds, validate_train_val_test_folds

PROTOCOL = 'patient_train_val_test_5fold_v1'


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _native_splits(manifest: dict) -> list[dict]:
    return [{key: fold[key] for key in ('train', 'val')} for fold in manifest['folds']]


def available_cases(folder: Path) -> list[str]:
    """Read the preprocessed inventory; require both image and segmentation arrays."""
    ids = sorted(path.stem for path in folder.glob('*.pkl'))
    if not ids:
        raise ValueError(f'No preprocessed cases in {folder}')
    for case in ids:
        paired = any((folder / f'{case}{image}').is_file()
                     and (folder / f'{case}{seg}').is_file()
                     for image, seg in (('.b2nd', '_seg.b2nd'), ('.npy', '_seg.npy')))
        if not paired and not (folder / f'{case}.npz').is_file():
            raise ValueError(f'Preprocessed case has missing/incomplete arrays: {folder / case}')
    return ids


def _complete_dataset_cases(source: Path, data_identifier: str) -> list[str]:
    cases = available_cases(source / data_identifier)
    dataset = json.loads((source / 'dataset.json').read_text(encoding='utf-8'))
    expected = dataset.get('numTraining')
    if type(expected) is not int or expected < 1:
        raise ValueError(f'dataset.json must declare a positive integer numTraining: {source}')
    if len(cases) != expected:
        raise ValueError(f'Incomplete dataset: expected numTraining={expected}, '
                         f'found {len(cases)} preprocessed cases in {source / data_identifier}')
    return cases


def _validate_manifest(manifest: dict) -> dict:
    if manifest.get('protocol') != PROTOCOL:
        raise ValueError('Unsupported patient partition protocol')
    name = manifest['dataset_name']
    if not isinstance(name, str) or Path(name).name != name or not name.startswith('Dataset'):
        raise ValueError('Invalid dataset name in patient partition')
    counts = validate_train_val_test_folds(manifest['folds'], manifest['cases'])
    if manifest['folds'] != build_train_val_test_folds(
            manifest['cases'], manifest['seed'], manifest['validation_fraction']):
        raise ValueError('Partition differs from its declared seed or validation fraction')
    if counts != manifest['counts']:
        raise ValueError('Partition counts do not match its case lists')
    return manifest


def load_partition(path: Path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / 'partition.json'
    return _validate_manifest(json.loads(path.read_text(encoding='utf-8')))


def write_partition(source: Path, destination: Path, *, seed: int = 42,
                    validation_fraction: float = 0.1,
                    data_identifier: str = 'nnUNetPlans_3d_fullres') -> dict:
    """Publish an explicit partition without replacing a legacy or existing split."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    cases = _complete_dataset_cases(source, data_identifier)
    folds = build_train_val_test_folds(cases, seed, validation_fraction)
    manifest = {'protocol': PROTOCOL, 'dataset_name': source.name,
                'grouping': 'BraTS-GLI-NNNNN (session suffix removed)',
                'seed': seed, 'validation_fraction': validation_fraction,
                'source_dataset': str(source), 'data_identifier': data_identifier,
                'cases': cases, 'folds': folds,
                'counts': validate_train_val_test_folds(folds, cases)}
    legacy = source / 'splits_final.json'
    if legacy.is_file():
        manifest['legacy_split_sha256'] = hashlib.sha256(legacy.read_bytes()).hexdigest()
    # Refuse replacement, including accidental publication over the shared dataset.
    if destination.exists():
        raise FileExistsError(f'Partition destination already exists: {destination}')
    destination.mkdir(parents=True)
    _write_json(destination / 'partition.json', manifest)
    _write_json(destination / 'splits_final.json', _native_splits(manifest))
    _write_json(destination / 'test_splits.json', [{'test': fold['test']} for fold in folds])
    return manifest


def stage_run_partition(source: Path, run_dir: Path, partition: Path,
                        data_identifier: str) -> Path:
    """Copy small metadata, symlink arrays, and save a self-contained split snapshot."""
    source, run_dir = Path(source).resolve(), Path(run_dir).resolve()
    manifest = load_partition(partition)
    if manifest['dataset_name'] != source.name:
        raise ValueError('Patient partition belongs to a different dataset')
    if _complete_dataset_cases(source, data_identifier) != manifest['cases']:
        raise ValueError('Preprocessed case inventory differs from the patient partition')
    root = run_dir / 'nnUNet_preprocessed'
    snapshot_path = run_dir / 'data_partition.json'
    if root.exists() or snapshot_path.exists():
        raise FileExistsError(f'Run already has a data partition: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.partition-', dir=run_dir) as temporary:
        staged_root = Path(temporary) / 'nnUNet_preprocessed'
        dataset = staged_root / source.name
        dataset.mkdir(parents=True)
        for item in source.iterdir():
            if item.is_dir():
                (dataset / item.name).symlink_to(item.resolve(), target_is_directory=True)
            elif item.suffix == '.json' and item.name not in (
                    'splits_final.json', 'test_splits.json', 'patient_partition.json'):
                shutil.copy2(item, dataset / item.name)
        _write_json(dataset / 'splits_final.json', _native_splits(manifest))
        _write_json(dataset / 'patient_partition.json', manifest)
        os.replace(staged_root, root)
    snapshot = {'preprocessed_root': str(root), 'source_dataset': str(source),
                'data_identifier': data_identifier, 'partition': manifest}
    _write_json(snapshot_path, snapshot)
    return root


def restore_run_partition(run_dir: Path) -> dict | None:
    """Restore new runs automatically; legacy runs retain their original environment."""
    path = Path(run_dir).resolve() / 'data_partition.json'
    if not path.is_file():
        return None
    snapshot = json.loads(path.read_text(encoding='utf-8'))
    manifest = _validate_manifest(snapshot['partition'])
    root = Path(snapshot['preprocessed_root'])
    dataset = root / manifest['dataset_name']
    if json.loads((dataset / 'splits_final.json').read_text()) != _native_splits(manifest):
        raise ValueError(f'The saved training/validation split was modified: {dataset}')
    if json.loads((dataset / 'patient_partition.json').read_text()) != manifest:
        raise ValueError(f'The saved patient partition was modified: {dataset}')
    if _complete_dataset_cases(dataset, snapshot['data_identifier']) != manifest['cases']:
        raise ValueError('The saved patient partition has missing or changed data')
    os.environ['nnUNet_preprocessed'] = str(root)
    return snapshot


def evaluation_case_ids(trainer, subset: str = 'val') -> list[str]:
    """Keep training-time validation unchanged; test requires an explicit request."""
    if subset not in ('val', 'test'):
        raise ValueError('Evaluation subset must be val or test')
    # Native do_split also initializes the dataset class needed by the evaluator.
    _, validation = trainer.do_split()
    if subset == 'val':
        return validation
    path = Path(trainer.preprocessed_dataset_folder_base) / 'patient_partition.json'
    if not path.is_file():
        raise ValueError('Held-out test requires a saved patient partition')
    manifest = load_partition(path)
    if type(trainer.fold) is not int or not 0 <= trainer.fold < 5:
        raise ValueError('Held-out test requires an outer fold between 0 and 4')
    return manifest['folds'][trainer.fold]['test']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Existing preprocessed Dataset directory')
    parser.add_argument('--output', required=True, type=Path, help='New partition directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--data-identifier', default='nnUNetPlans_3d_fullres')
    args = parser.parse_args()
    manifest = write_partition(args.source, args.output, seed=args.seed,
                               validation_fraction=args.validation_fraction,
                               data_identifier=args.data_identifier)
    print(json.dumps(manifest['counts'], indent=2))


if __name__ == '__main__':
    main()
