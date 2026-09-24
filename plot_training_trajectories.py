#!/usr/bin/env python3
"""Plot configured training trajectories from CSV logs."""

from __future__ import annotations

import csv
import itertools
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

CurveSpec = Tuple[str, str, str]

CURVES: list[CurveSpec] = [
    # ("experiments/brats24_sagittal_fold1_VSS_tbptt_20260914_112710/epoch_metrics.csv", "dice_mean", "Spiking Mamba - DICE - testing"),
    ("experiments/brats24_sagittal_fold1_VSS_tbptt_20260914_112710/training_dice.csv", "dice_mean", "Spiking Mamba - DICE - training"),
    # ("experiments/brats24_sagittal_fold1_VSS_tbptt_20260915_133521/epoch_metrics.csv", "dice_mean", "Mamba - DICE - testing"),
    ("experiments/brats24_sagittal_fold1_VSS_tbptt_20260915_133521/training_dice.csv", "dice_mean", "Mamba - DICE - training"),
]
OUTPUT_PATH = "dice_trajectories_training.png"
PLOT_TITLE = "DICE TESTING"

# CURVES: list[CurveSpec] = [
#     ("experiments/brats24_sagittal_fold1_VSS_tbptt_20260914_112710/loss_components.csv", "total_loss", "Spiking Mamba"),
#     ("experiments/brats24_sagittal_fold1_VSS_tbptt_20260915_133521/loss_components.csv", "total_loss", "Mamba"),
# ]
# OUTPUT_PATH = "loss_trajectories.png"
# PLOT_TITLE = "LOSS FPTT"


@dataclass(frozen=True)
class CurveData:
    epochs: list[float]
    values: list[float]
    label: str


def load_curve(spec: CurveSpec) -> CurveData:
    csv_name, metric_column, label = spec
    path = Path(csv_name)
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")
    epochs, values = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = {"epoch", metric_column} - columns
        if missing:
            raise ValueError(
                f"CSV {path} is missing column(s): {', '.join(sorted(missing))}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                epoch = float(row["epoch"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CSV {path} has an invalid epoch at row {row_number}"
                ) from exc
            if not math.isfinite(epoch):
                raise ValueError(
                    f"CSV {path} has an invalid epoch at row {row_number}"
                )
            raw_value = row[metric_column]
            if raw_value is None or not raw_value.strip():
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            epochs.append(epoch)
            values.append(value)
    if not epochs:
        raise ValueError(f"CSV {path} has no valid values for {metric_column!r}")
    return CurveData(epochs=epochs, values=values, label=label)

COLORS = (
    plt.get_cmap("tab10").colors[0],  # blue
    "red",
)

LINESTYLES = ("-", "--", "-.", ":")


def curve_styles(count: int) -> list[tuple]:
    max_curves = len(COLORS) * len(LINESTYLES)

    if count > max_curves:
        raise ValueError(
            f"Requested {count} curves but only {max_curves} distinct styles exist"
        )

    styles = []

    for i in range(count):
        color = COLORS[i % 2]
        linestyle = LINESTYLES[(i // 2) % len(LINESTYLES)]

        styles.append((color, linestyle))

    return styles

def plot_training_trajectories(
    curves: Sequence[CurveSpec], output_path: str | Path, title: str,
) -> Path:
    validate_curve_specs(curves)
    data = [load_curve(spec) for spec in curves]
    styles = curve_styles(len(data))
    figure, axes = plt.subplots(figsize=(11, 7))
    for curve, (color, linestyle) in zip(data, styles):
        axes.plot(
            curve.epochs,
            curve.values,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            marker="o",
            label=curve.label,
        )
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Metric value")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def validate_curve_specs(curves: Sequence[CurveSpec]) -> None:
    if not curves:
        raise ValueError("CURVES must contain at least one training curve")
    for index, spec in enumerate(curves, start=1):
        if not isinstance(spec, tuple) or len(spec) != 3:
            raise ValueError(f"CURVES item {index} must be a tuple with three values")
        if not all(isinstance(value, str) and value.strip() for value in spec):
            raise ValueError(f"CURVES item {index} values must be non-empty strings")


def main() -> None:
    destination = plot_training_trajectories(CURVES, OUTPUT_PATH, PLOT_TITLE)
    print(f"Saved training trajectory plot to {destination.resolve()}")


if __name__ == "__main__":
    main()
