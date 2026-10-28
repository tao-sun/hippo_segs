"""Validated, serializable initialization and runtime options for SiLIF."""

from collections.abc import Mapping
import math


SILIF_DEFAULTS = {
    "v_threshold": 1.0,
    "dt_init": 0.004,
    "rate_u_range": [10.0, 250.0],
    "rate_w_range": [2.5, 10.0],
    "a_range": [0.0, 1.0],
    "b_range": [0.0, 2.0],
    "surrogate_alpha": 2.0,
    "detach_reset": True,
}


def silif_options(value):
    """Fill omitted defaults and reject invalid or silently misspelled options."""
    if not isinstance(value, Mapping):
        raise ValueError("silif must be a mapping")
    unknown = value.keys() - SILIF_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"silif contains unknown options: {sorted(unknown)}")
    result = {**SILIF_DEFAULTS, **value}
    for key in ("v_threshold", "dt_init", "surrogate_alpha"):
        number = result[key]
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"silif.{key} must be positive and finite")
        result[key] = float(number)
    if type(result["detach_reset"]) is not bool:
        raise ValueError("silif.detach_reset must be a boolean")
    for key in ("rate_u_range", "rate_w_range", "a_range", "b_range"):
        interval = result[key]
        if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in interval)):
            raise ValueError(f"silif.{key} must contain two finite numbers")
        low, high = interval
        if low > high:
            raise ValueError(f"silif.{key} requires min <= max")
        if key.startswith("rate_") and low <= 0:
            raise ValueError(f"silif.{key} rates must be positive")
        if key in ("a_range", "b_range"):
            upper = 1 if key == "a_range" else 2
            if low < 0 or high > upper:
                raise ValueError(f"silif.{key} must lie within [0, {upper}]")
        result[key] = [float(low), float(high)]
    return result
