"""Strict, serializable parameters for the fixed main:model.py SNNBraTS graph."""
from collections.abc import Mapping
from copy import deepcopy
import math

from snn_nnunet.recurrent_config import RECURRENT_DEFAULTS, ATTENTION_MODEL, normalize_recurrent_model, recurrent_options

BLOCK_NAMES = ('conv_block1', 'conv_block2', 'conv_block3',
               'deconv_block1', 'deconv1_conv', 'concat1_conv',
               'deconv_block2', 'deconv2_conv', 'concat2_conv',
               'deconv_block3', 'deconv3_conv')
ENCODER_BLOCK_NAMES = BLOCK_NAMES[:3]
NEURON_MODELS = ('plif', 'selective_multitimescale_lif', ATTENTION_MODEL)
BIDIRECTIONAL_FUSIONS = ('conv_plif', 'or', 'Shared_SOMA')
PLIF_DEFAULTS = {'init_tau': 2.0, 'surrogate_alpha': 2.0, 'detach_reset': True}
PAPER_DEFAULTS = {
    'encoder_channels': [32, 64, 128], 'decoder_channels': [128, 128, 128],
    'encoder_spiking': True, 'decoder_spiking': True,
    'full_resolution_skip': False,
    'temporal_mamba': False,
    'previous_probability_feedback': False,
    'previous_probability_feedback_detach': True,
    'previous_probability_feedback_init': 'zeros',
    'temporal_mamba_options': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'dt_rank': 'auto'},
    'deep_supervision': {'enabled': False, 'weights': [1.0, 0.3, 0.1], 'resolution': 'full'},
    'conv': {'kernel_size': 3, 'padding': 1, 'stride': 1, 'bias': False, 'dropout': .1},
    'deconv': {'kernel_size': 2, 'padding': 0, 'stride': 2, 'output_padding': 0,
               'bias': False, 'dropout': .1},
    'pool': {'kernel_size': 2, 'stride': 2, 'padding': 0},
    'normalization': {'type': 'groupnorm', 'num_groups': 1, 'eps': 1e-5, 'affine': True},
    'neurons': {'model': 'plif', 'overrides': {}, 'plif': PLIF_DEFAULTS,
                'multi_memory': {key: value for key, value in RECURRENT_DEFAULTS.items()
                                 if key != 'selective_readout'},
                'selective_readout': RECURRENT_DEFAULTS['selective_readout']},
    'head': {'kernel_size': 3, 'padding': 1, 'stride': 1, 'bias': False,
             'dropout': 0., 'normalization': False, 'spiking': False,
             'plif': {**PLIF_DEFAULTS, 'v_threshold': 1., 'v_reset': 0.}},
}


def normalize_bidirectional_fusion(value, *, enabled):
    """Ignore inactive fusion settings and default legacy plans to Conv+PLIF."""
    if not enabled:
        return 'conv_plif'
    aliases = {'or': 'or', 'conv_plif': 'conv_plif', 'shared_soma': 'Shared_SOMA'}
    if isinstance(value, str):
        value = aliases.get(value.strip().lower().replace(' ', '_'), value)
    if not isinstance(value, str) or value not in BIDIRECTIONAL_FUSIONS:
        raise ValueError(f'Unsupported fusion: {value!r}; expected one of {BIDIRECTIONAL_FUSIONS}')
    return value


def neuron_model(neurons, name):
    """Explicit block choices take priority over the old global readout selector."""
    if name in neurons['overrides']:
        return neurons['overrides'][name]
    selected = neurons['model']
    if (selected == 'selective_multitimescale_lif' and neurons['multi_memory'] is not None
            and neurons['multi_memory']['model'] == ATTENTION_MODEL):
        return ATTENTION_MODEL
    return selected


def bidirectional_blocks(options, *, enabled):
    """Only plain SiLIF encoder stages have a backward branch."""
    return tuple(name for name in ENCODER_BLOCK_NAMES
                 if enabled and options['encoder_spiking']
                 and neuron_model(options['neurons'], name) == 'selective_multitimescale_lif')


def _merge(defaults, values, path):
    if not isinstance(values, Mapping):
        raise ValueError(f'{path} must be a mapping')
    unknown = values.keys() - defaults.keys()
    if unknown:
        raise ValueError(f'{path}: unknown options {sorted(unknown)}')
    result = deepcopy(defaults)
    for key, value in values.items():
        # Named overrides and active selective readouts are validated separately.
        result[key] = (_merge(defaults[key], value, f'{path}.{key}')
                       if isinstance(defaults[key], dict) and key not in ('overrides', 'selective_readout')
                       else deepcopy(value))
    return result


def _int(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')


def _number(value, label, minimum=0., strict=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= minimum if strict else value < minimum)):
        raise ValueError(f'{label} has an invalid numeric value')


def plif_options(values, *, head=False):
    defaults = PAPER_DEFAULTS['head']['plif'] if head else PLIF_DEFAULTS
    result = _merge(defaults, values, 'plif')
    _number(result['init_tau'], 'init_tau', 1., True)
    _number(result['surrogate_alpha'], 'surrogate_alpha', 0., True)
    if type(result['detach_reset']) is not bool:
        raise ValueError('detach_reset must be boolean')
    if head:
        _number(result['v_threshold'], 'v_threshold', 0., True)
        if result['v_reset'] is not None:
            _number(abs(result['v_reset']) if type(result['v_reset']) in (int, float) else result['v_reset'], 'v_reset')
    return result


def paper_options(values):
    # Migrate old saved schemas before strict merging, without mutating callers.
    values = deepcopy(values)
    if isinstance(values, Mapping) and isinstance(values.get('neurons'), Mapping):
        neurons = values['neurons']
        if 'selective' in neurons:
            if 'multi_memory' in neurons:
                raise ValueError('neurons.selective and neurons.multi_memory cannot both be supplied')
            common = neurons.pop('selective')
            if not isinstance(common, Mapping):
                raise ValueError('neurons.selective must be a mapping')
            common = dict(common)
            if 'selective_readout' in common:
                if 'selective_readout' in neurons:
                    raise ValueError('Supply selective_readout only once, under neurons')
                neurons['selective_readout'] = common.pop('selective_readout')
            neurons['multi_memory'] = common
        if 'model' in neurons:
            neurons['model'] = normalize_recurrent_model(neurons['model'])
        if isinstance(neurons.get('overrides'), Mapping):
            neurons['overrides'] = {name: normalize_recurrent_model(model)
                                    for name, model in neurons['overrides'].items()}
    result = _merge(PAPER_DEFAULTS, values, 'paper_unet')
    for key in ('encoder_spiking', 'decoder_spiking', 'full_resolution_skip', 'temporal_mamba',
                'previous_probability_feedback', 'previous_probability_feedback_detach'):
        if type(result[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if result['previous_probability_feedback_init'] != 'zeros':
        raise ValueError('previous_probability_feedback_init must be zeros')
    if result['previous_probability_feedback'] and result['temporal_mamba']:
        raise ValueError('previous_probability_feedback requires temporal_mamba=false')
    temporal = result['temporal_mamba_options']
    for key in ('d_state', 'd_conv', 'expand'):
        _int(temporal[key], f'temporal_mamba_options.{key}')
    if temporal['dt_rank'] != 'auto':
        _int(temporal['dt_rank'], 'temporal_mamba_options.dt_rank')
    if result['temporal_mamba'] and not result['encoder_spiking']:
        raise ValueError('temporal_mamba binary bottleneck requires encoder_spiking=true')
    supervision = result['deep_supervision']
    if type(supervision['enabled']) is not bool:
        raise ValueError('deep_supervision.enabled must be boolean')
    if supervision['resolution'] not in ('full', 'native'):
        raise ValueError('deep_supervision.resolution must be full or native')
    weights = supervision['weights']
    if not isinstance(weights, (list, tuple)) or len(weights) != 3:
        raise ValueError('deep_supervision.weights must contain main, half and quarter weights')
    for value in weights:
        _number(value, 'deep_supervision.weights', 0., True)
    if not math.isfinite(sum(weights)):
        raise ValueError('deep_supervision.weights must have a finite sum')
    supervision['weights'] = [float(value) for value in weights]
    for key in ('encoder_channels', 'decoder_channels'):
        channels = result[key]
        if not isinstance(channels, (list, tuple)) or len(channels) != 3:
            raise ValueError(f'{key} must contain three channel counts')
        for channel in channels:
            _int(channel, key)
        result[key] = list(channels)
    for section in ('conv', 'deconv', 'pool', 'head'):
        opts = result[section]
        for key in ('kernel_size', 'stride'):
            _int(opts[key], f'{section}.{key}')
        _int(opts['padding'], f'{section}.padding', 0)
        if 'dropout' in opts:
            _number(opts['dropout'], f'{section}.dropout')
            if opts['dropout'] >= 1:
                raise ValueError(f'{section}.dropout must be < 1')
        if 'bias' in opts and type(opts['bias']) is not bool:
            raise ValueError(f'{section}.bias must be boolean')
    for section in ('conv', 'head'):
        opts = result[section]
        if opts['stride'] != 1 or 2 * opts['padding'] != opts['kernel_size'] - 1:
            raise ValueError(f'{section} must preserve spatial resolution (stride=1, 2*padding=kernel_size-1)')
    pool, up = result['pool'], result['deconv']
    if pool['kernel_size'] != pool['stride'] or pool['padding'] != 0 or pool['stride'] < 2:
        raise ValueError('pool must use kernel_size=stride>=2 and padding=0')
    _int(up['output_padding'], 'deconv.output_padding', 0)
    if (up['stride'] != pool['stride'] or up['output_padding'] >= up['stride']
            or up['kernel_size'] + up['output_padding'] - 2*up['padding'] != up['stride']):
        raise ValueError('deconv must exactly invert the configured pooling scale')
    norm = result['normalization']
    if norm['type'] != 'groupnorm':
        raise ValueError('normalization.type must be groupnorm')
    _int(norm['num_groups'], 'normalization.num_groups')
    if any(c % norm['num_groups'] for c in result['encoder_channels'] + result['decoder_channels']):
        raise ValueError('normalization.num_groups must divide every block channel count')
    _number(norm['eps'], 'normalization.eps', 0., True)
    if type(norm['affine']) is not bool:
        raise ValueError('normalization.affine must be boolean')
    head = result['head']
    if head['normalization'] is not False or head['spiking'] is not False:
        raise ValueError('head must remain non-spiking without normalization')
    head['plif'] = plif_options(head['plif'], head=True)
    neurons = result['neurons']
    if neurons['model'] not in NEURON_MODELS:
        raise ValueError(f'neurons.model must be one of {NEURON_MODELS}')
    overrides = neurons['overrides']
    if (not isinstance(overrides, Mapping) or overrides.keys() - set(BLOCK_NAMES)
            or any(value not in NEURON_MODELS for value in overrides.values())):
        raise ValueError('neurons.overrides must map original spiking block names to supported neuron models')
    neurons['overrides'] = dict(overrides)
    neurons['plif'] = plif_options(neurons['plif'])
    common = recurrent_options(neurons['multi_memory'])
    common.pop('selective_readout', None)
    neurons['multi_memory'] = common
    if any(neuron_model(neurons, name) == 'selective_multitimescale_lif' for name in BLOCK_NAMES):
        # These gates apply only to selected plain selective cells.
        config = recurrent_options(dict(common, model='selective_multitimescale_lif',
                                        selective_readout=neurons['selective_readout']))
        neurons['selective_readout'] = config['selective_readout']
    return result


def validate_shared_soma_options(options, *, enabled, fusion):
    """Check Shared_SOMA only where an active plain SiLIF encoder uses it."""
    if fusion != 'Shared_SOMA' or not bidirectional_blocks(options, enabled=enabled):
        return
    if not options['neurons']['selective_readout']['enabled']:
        raise ValueError('Shared_SOMA requires selective_readout.enabled=true')
