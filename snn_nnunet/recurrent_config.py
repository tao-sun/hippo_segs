"""Validated optional VSS output neuron settings (also serialized in plans)."""
from collections.abc import Mapping
from copy import deepcopy
import math


RECURRENT_DEFAULTS = {
    'model': 'selective_multitimescale_lif', 'compartments': 3,
    'lambda_init': [.50, .80, .95],
    'spatial_mixing': {
        'enabled': True, 'kernel_size': 3, 'padding': 'zeros', 'eta_init': .1,
        'kernel_init': 'near_identity', 'kernel_center_init': .95,
        'kernel_surround_init': .001, 'max_kernel_l1_norm': 1.,
    },
    'selective_readout': {
        'enabled': True, 'gate': 'sigmoid', 'gate_kernel_size': 1,
        'gate_init': 'zeros', 'static_gate_init': .5, 'fusion': 'weighted_sum',
    },
    'memory_projection': {'kernel_size': 1, 'bias': False, 'init': 'xavier_uniform'},
    'memory_dtype': 'float32', 'direction': 'forward',
    'diagnostics': {'every_n_windows': 100, 'near_lambda_gap': .001,
                    'growth_ratio_threshold': 100.},
}


def initial_temporal_raw(coefficients):
    """Stable inverse softplus, rejecting only FP32-infeasible ordered decays."""
    import torch
    from torch.nn import functional as F
    lambdas = torch.tensor(coefficients, dtype=torch.float64)
    tau = -1. / torch.log(lambdas)
    increments = torch.cat((tau[:1] - 1e-6, tau[1:] - tau[:-1]))
    raw = (increments + torch.log(-torch.expm1(-increments))).float()
    effective = torch.exp(-(F.softplus(raw).cumsum(0) + 1e-6).reciprocal())
    if not ((effective > 0).all() and (effective < 1).all()
            and (effective[1:] > effective[:-1]).all()):
        raise ValueError('lambda_init cannot preserve strict ordering and bounds in FP32')
    return raw


def initial_bounded_logit(value, label):
    """Match the effective FP32 sigmoid used by learned mixing/readout weights."""
    import torch
    logit = math.log(value) - math.log1p(-value)
    effective = torch.tensor(logit, dtype=torch.float32).sigmoid()
    if not 0 < effective < 1:
        raise ValueError(f'{label} cannot preserve strict bounds in FP32')
    return logit


def recurrent_options(value):
    """None preserves the baseline. Reject unsupported equations/unknown options."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('recurrent_neuron must be a mapping')
    result = deepcopy(RECURRENT_DEFAULTS)
    if value.keys() - result.keys():
        raise ValueError('recurrent_neuron contains unknown options')
    for key, item in value.items():
        if isinstance(result[key], dict):
            if not isinstance(item, Mapping) or item.keys() - result[key].keys():
                raise ValueError(f'recurrent_neuron.{key} contains unsupported options')
            result[key].update(item)
        else:
            result[key] = deepcopy(item)

    def fixed(group, key, expected):
        if group[key] != expected or type(group[key]) is not type(expected):
            raise ValueError(f'recurrent_neuron.{key} must be {expected!r}')

    def number(group, key, low, high, inclusive_high=False):
        v = group[key]
        if (type(v) not in (int, float) or not math.isfinite(v) or v <= low
                or (v > high if inclusive_high else v >= high)):
            raise ValueError(f'recurrent_neuron.{key} is outside its supported range')

    for key in ('model', 'compartments', 'memory_dtype', 'direction'):
        fixed(result, key, RECURRENT_DEFAULTS[key])
    init = result['lambda_init']
    if (not isinstance(init, (list, tuple)) or len(init) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 < x < 1 for x in init)
            or not init[0] < init[1] < init[2]):
        raise ValueError('lambda_init must contain three strictly increasing values in (0,1)')
    # The epsilon in tau_f is numerical protection, not a minimum decay gap.
    if -1 / math.log(init[0]) <= 1e-6:
        raise ValueError('lambda_init is too small for the supported tau precision')
    initial_temporal_raw(init)
    result['lambda_init'] = list(init)
    for section in ('spatial_mixing', 'selective_readout', 'memory_projection'):
        for key, default in RECURRENT_DEFAULTS[section].items():
            if key in ('enabled',):
                if type(result[section][key]) is not bool:
                    raise ValueError(f'{section}.{key} must be a boolean')
            elif key not in ('eta_init', 'kernel_center_init', 'kernel_surround_init',
                             'max_kernel_l1_norm', 'static_gate_init'):
                fixed(result[section], key, default)
    spatial = result['spatial_mixing']
    number(spatial, 'eta_init', 0, 1)
    initial_bounded_logit(spatial['eta_init'], 'eta_init')
    number(spatial, 'max_kernel_l1_norm', 0, 1, True)
    number(spatial, 'kernel_center_init', 0, 1)
    number(spatial, 'kernel_surround_init', 0, 1)
    if spatial['kernel_center_init'] + 8*spatial['kernel_surround_init'] > spatial['max_kernel_l1_norm']:
        raise ValueError('near_identity initialization exceeds max_kernel_l1_norm')
    number(result['selective_readout'], 'static_gate_init', 0, 1)
    initial_bounded_logit(result['selective_readout']['static_gate_init'], 'static_gate_init')
    diagnostics = result['diagnostics']
    if type(diagnostics['every_n_windows']) is not int or diagnostics['every_n_windows'] < 0:
        raise ValueError('diagnostics.every_n_windows must be a nonnegative integer (0 disables)')
    number(diagnostics, 'near_lambda_gap', 0, 1)
    number(diagnostics, 'growth_ratio_threshold', 0, float('inf'))
    return result
