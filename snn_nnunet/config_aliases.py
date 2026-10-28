"""Lightweight compatibility helpers shared by YAML, CLI and model loaders."""

SPIKING_OPTION_ALIASES = {
    "stem_plif": "stem_spiking",
    "downsampling_plif": "downsampling_spiking",
    "resdecoder_plif1": "resdecoder_spiking1",
    "resdecoder_plif2": "resdecoder_spiking2",
}


def normalize_spiking_options(values):
    """Return canonical switch names without changing the caller's mapping."""
    result = dict(values)
    for legacy, canonical in SPIKING_OPTION_ALIASES.items():
        if legacy not in result:
            continue
        if canonical in result:
            raise ValueError(f"Specify only {canonical}; {legacy} is its legacy alias")
        result[canonical] = result.pop(legacy)
    return result
