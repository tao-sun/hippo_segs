"""Reproducible BraTS cross-validation with all sessions of a base ID together."""

from __future__ import annotations

from collections import Counter
import re

import numpy as np


_CASE_ID = re.compile(r'(BraTS-GLI-\d{5})-\d{3}')


def patient_id(case_id: str) -> str:
    """Group by the base identifier; do not infer groups from an ambiguous name."""
    match = _CASE_ID.fullmatch(case_id) if isinstance(case_id, str) else None
    if match is None:
        raise ValueError(f'Expected a BraTS-GLI-NNNNN-SSS case identifier: {case_id!r}')
    return match.group(1)


def _checked_cases(case_ids: list[str]) -> list[str]:
    if not case_ids:
        raise ValueError('No BraTS cases supplied')
    for case_id in case_ids:
        patient_id(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError('Case identifiers contain duplicates')
    return sorted(case_ids)


def build_patient_folds(case_ids: list[str], seed: int = 42) -> list[dict]:
    """Shuffle sorted base IDs, divide into five balanced folds, expand to cases.

    ``val`` is nnU-Net's held-out CV fold, including its usual online validation
    and checkpoint-selection role; it is not an independent final test set.
    """
    cases = _checked_cases(case_ids)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('seed must be an integer between 0 and 2**32 - 1')
    patients = np.array(sorted({patient_id(case) for case in cases}))
    if len(patients) < 5:
        raise ValueError('At least five patients are required for five folds')
    np.random.RandomState(seed).shuffle(patients)
    folds = []
    for held_out in np.array_split(patients, 5):
        validation_patients = set(held_out)
        folds.append({
            'train': [case for case in cases if patient_id(case) not in validation_patients],
            'val': [case for case in cases if patient_id(case) in validation_patients],
        })
    validate_patient_folds(folds, cases)
    return folds


def validate_patient_folds(splits: list[dict], case_ids: list[str]) -> list[dict]:
    """Fail closed on leakage, incomplete folds or repeated held-out cases."""
    cases = set(_checked_cases(case_ids))
    if not isinstance(splits, list) or len(splits) != 5:
        raise ValueError('Expected exactly five patient folds')
    counts, held_out = [], []
    for index, split in enumerate(splits):
        if not isinstance(split, dict) or set(split) != {'train', 'val'}:
            raise ValueError(f'Fold {index}: expected train and val lists')
        for subset in ('train', 'val'):
            values = split[subset]
            if not isinstance(values, list) or not values:
                raise ValueError(f'Fold {index}: {subset} must be a nonempty list')
            _checked_cases(values)
        training, validation = set(split['train']), set(split['val'])
        if training & validation:
            raise ValueError(f'Fold {index}: case overlap between training and validation')
        if training | validation != cases:
            raise ValueError(f'Fold {index}: case coverage differs from the supplied dataset')
        training_patients = {patient_id(case) for case in training}
        validation_patients = {patient_id(case) for case in validation}
        if training_patients & validation_patients:
            raise ValueError(f'Fold {index}: patient overlap between training and validation')
        counts.append({'fold': index, 'train_patients': len(training_patients),
                       'val_patients': len(validation_patients),
                       'train_cases': len(training), 'val_cases': len(validation)})
        held_out.extend(split['val'])
    if Counter(held_out) != Counter({case: 1 for case in cases}):
        raise ValueError('Each case must be held out exactly once across all five folds')
    return counts
