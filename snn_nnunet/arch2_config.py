"""Strict, JSON-serializable options for the three-stage spatial Arch2."""
from collections.abc import Mapping
from copy import deepcopy
import math


DEFAULT_ARCH2 = {
    'patch_size': 2,
    'encoder': {
        'channels': [32, 64, 128],
        'patch_embedding_spiking': True,
        'dwconv2d_spiking': True,
        'vss_output_spiking': True,
    },
    'decoder': {'channels': [128, 128, 128], 'spiking': True},
    'deep_supervision': {'enabled': False, 'weights': [1.0, 0.5, 0.25], 'resolution': 'native'},
    'ssm': {
        'd_state': 16, 'expand': 2, 'dt_rank': 'auto',
        'dwconv2d_kernel_size': 3,
        'linear_projection': True, 'residual_connections': True,
    },
    'neurons': {
        'init_tau': 2.0, 'surrogate_alpha': 2.0,
        'v_threshold': 1.0, 'v_reset': 0.0, 'detach_reset': True,
    },
    'dropout': 0.1,
}


def arch2_options(values=None):
    """Fill defaults without accepting inactive legacy architecture controls."""
    if values is None:
        values = {}

    def merge(defaults, supplied, path):
        if not isinstance(supplied, Mapping):
            raise ValueError(f'{path} must be a mapping')
        unknown = supplied.keys() - defaults.keys()
        if unknown:
            raise ValueError(f'{path}: unknown options {sorted(unknown)}')
        return {key: (merge(default, supplied.get(key, {}), f'{path}.{key}')
                      if isinstance(default, dict) else deepcopy(supplied.get(key, default)))
                for key, default in defaults.items()}

    result = merge(DEFAULT_ARCH2, values, 'arch2')

    def positive_int(value, name):
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')

    def finite(value, name):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{name} must be finite')

    positive_int(result['patch_size'], 'arch2.patch_size')
    for section in ('encoder', 'decoder'):
        widths = result[section]['channels']
        if not isinstance(widths, (list, tuple)) or len(widths) != 3:
            raise ValueError(f'arch2.{section}.channels must contain three positive integers')
        for width in widths:
            positive_int(width, f'arch2.{section}.channels')
        result[section]['channels'] = list(widths)
    for section, keys in (
            ('encoder', ('patch_embedding_spiking', 'dwconv2d_spiking', 'vss_output_spiking')),
            ('decoder', ('spiking',)),
            ('deep_supervision', ('enabled',)),
            ('ssm', ('linear_projection', 'residual_connections')),
            ('neurons', ('detach_reset',))):
        for key in keys:
            if type(result[section][key]) is not bool:
                raise ValueError(f'arch2.{section}.{key} must be boolean')
    supervision = result['deep_supervision']
    if supervision['resolution'] not in ('native', 'full'):
        raise ValueError('arch2.deep_supervision.resolution must be native or full')
    weights = supervision['weights']
    if not isinstance(weights, (list, tuple)) or len(weights) != 3:
        raise ValueError('arch2.deep_supervision.weights must contain three positive weights')
    for value in weights:
        finite(value, 'arch2.deep_supervision.weights')
        if value <= 0:
            raise ValueError('arch2.deep_supervision.weights must be positive')
    if not math.isfinite(sum(weights)):
        raise ValueError('arch2.deep_supervision.weights must have a finite sum')
    supervision['weights'] = [float(value) for value in weights]
    for key in ('d_state', 'expand', 'dwconv2d_kernel_size'):
        positive_int(result['ssm'][key], f'arch2.ssm.{key}')
    if result['ssm']['dwconv2d_kernel_size'] % 2 == 0:
        raise ValueError('arch2.ssm.dwconv2d_kernel_size must be odd')
    if result['ssm']['dt_rank'] != 'auto':
        positive_int(result['ssm']['dt_rank'], 'arch2.ssm.dt_rank')
    neurons = result['neurons']
    for key in ('init_tau', 'surrogate_alpha', 'v_threshold'):
        finite(neurons[key], f'arch2.neurons.{key}')
        if neurons[key] <= (1. if key == 'init_tau' else 0.):
            raise ValueError(f'arch2.neurons.{key} is outside its valid range')
    if neurons['v_reset'] is not None:
        finite(neurons['v_reset'], 'arch2.neurons.v_reset')
        if neurons['v_reset'] >= neurons['v_threshold']:
            raise ValueError('arch2.neurons.v_reset must be below v_threshold')
    finite(result['dropout'], 'arch2.dropout')
    if not 0 <= result['dropout'] < 1:
        raise ValueError('arch2.dropout must be in [0, 1)')
    return result
