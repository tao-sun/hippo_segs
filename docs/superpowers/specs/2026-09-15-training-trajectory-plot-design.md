# Training Trajectory Plot Design

## Goal

Add a standalone Python script that plots one or more training trajectories
from explicitly configured CSV files. Each curve is defined directly in the
script by a triple containing the complete CSV path, the metric column, and the
human-readable training name.

## Interface

The script is named `plot_training_trajectories.py`. Its editable configuration
is declared near the top of the file:

```python
CURVES = [
    ("experiments/run_A/training_dice.csv", "dice_mean", "Baseline - training"),
    ("experiments/run_A/epoch_metrics.csv", "dice_mean", "Baseline - validation"),
]

OUTPUT_PATH = "training_trajectories.png"
PLOT_TITLE = "Training trajectories"
```

No command-line arguments are required. Running
`python plot_training_trajectories.py` reads this configuration and writes the
plot to `OUTPUT_PATH`.

## Data flow

For each configured triple, the script opens the exact CSV path and reads
`epoch` for the X axis and the selected metric column for the Y axis. Rows with
a missing or non-finite metric value are skipped, supporting CSV logs where
validation is recorded only every `eval_every` epochs. Epoch values must be
numeric and are retained in their recorded order.

The resulting curves share one axes object. Every triple receives a distinct
combination of color and line style, cycling through both palettes when needed.
The legend label is the configured training name. The plot includes an epoch
axis label, a metric-value axis label, title, legend, and grid, and is saved as
a PNG with a tight layout.

## Validation and errors

The script raises clear errors when:

- `CURVES` is empty;
- a configured item is not a three-value tuple;
- a CSV file does not exist;
- the `epoch` or requested metric column is missing;
- no valid numeric points remain for a curve;
- two configured triples would receive an indistinguishable style.

Different metric columns may be plotted together; the generic Y-axis label is
`Metric value` so it does not incorrectly describe mixed metrics.

## Boundaries

The plotting utility does not modify training, checkpoints, CSV logs, model
selection, or evaluation. It only reads existing CSV files and writes the one
configured image. In the current training pipeline, `training_dice.csv`
contains training-set Dice values while `epoch_metrics.csv` contains validation
Dice values.

## Testing

Tests will use temporary CSV files and the non-interactive Matplotlib backend.
They will verify correct epoch/metric extraction, skipping blank metric rows,
clear failures for missing paths or columns, distinct style assignment, legend
labels, and successful PNG generation. Production behavior will be implemented
test-first.
