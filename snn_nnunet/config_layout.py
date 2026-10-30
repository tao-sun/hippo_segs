"""Translate the user-facing SNN hierarchy into the saved flat model settings."""

from copy import deepcopy


# Leaves name their existing destination in the launcher's normalized settings.
_LAYOUT = {
    'neurons': {
        'model': 'spatial_mamba.neuron_model',
        'silif': 'spatial_mamba.silif',
    },
    'stem': {
        'depth': 'spatial_mamba.stem_depth',
        'channels': 'spatial_mamba.stem_channels',
        'spiking': 'spatial_mamba.stem_spiking',
        'skip': 'spatial_mamba.full_resolution_skip',
        'skip_fusion': 'spatial_mamba.final_skip_fusion',
    },
    'encoder': {
        'hierarchical': 'spatial_mamba.hierarchical_encoder',
        'channels': 'spatial_mamba.encoder_channels',
        'stage': {
            'depths': 'spatial_mamba.vss_depths',
            'spiking': 'spatial_mamba.encoder_output_spiking',
            'recurrent_neuron': 'spatial_mamba.recurrent_neuron',
            'downsampling': {
                'kernel_size': 'spatial_mamba.downsample_kernel_size',
                'strides': 'spatial_mamba.downsample_strides',
                'spiking': 'spatial_mamba.downsampling_spiking',
            },
            'spatial_mamba': {
                'dwconv2d_kernel_size': 'spatial_mamba.dwconv2d_kernel_size',
                'patch_embedding_spiking': 'spatial_mamba.patch_embedding_spiking',
                'dwconv2d_spiking': 'spatial_mamba.dwconv2d_spiking',
            },
            'temporal_mamba': 'temporal_mamba',
        },
    },
    'decoder': {
        'full_resolution_skip': 'spatial_mamba.full_resolution_skip',
        'final_skip_fusion': 'spatial_mamba.final_skip_fusion',
        'stage': {
            'block': 'spatial_mamba.decoder_block',
            'conv_block_spiking': 'spatial_mamba.decoder_conv_spiking',
            'resdecoder_spiking1': 'spatial_mamba.resdecoder_spiking1',
            'resdecoder_spiking2': 'spatial_mamba.resdecoder_spiking2',
        },
        'skips': {
            'fue_skip1': 'spatial_mamba.fue_skip1',
            'fue_skip2': 'spatial_mamba.fue_skip2',
        },
    },
}


def normalize_snn_layout(values):
    """Accept grouped or legacy YAML, rejecting duplicate and unknown grouped keys."""
    result = deepcopy(values)

    def visit(group, schema, source):
        if not isinstance(group, dict):
            raise ValueError(f'{source} must be a YAML mapping')
        for key, value in group.items():
            location = f'{source}.{key}'
            if key not in schema:
                raise ValueError(f'Unknown option {location}')
            destination = schema[key]
            if isinstance(destination, dict):
                visit(value, destination, location)
                continue
            parts = destination.split('.')
            target = result
            for part in parts[:-1]:
                target = target.setdefault(part, {})
                if not isinstance(target, dict):
                    raise ValueError(f'snn.{part} must be a YAML mapping')
            name = parts[-1]
            if name in target:
                raise ValueError(f'Specify {location} only once; snn.{destination} is also set')
            target[name] = value

    for key, schema in _LAYOUT.items():
        if key in result:
            visit(result.pop(key), schema, f'snn.{key}')
    return result
