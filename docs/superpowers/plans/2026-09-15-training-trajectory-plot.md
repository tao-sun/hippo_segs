# Training Trajectory Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a directly configured Python utility that plots metric trajectories against training epochs from multiple CSV logs.

**Architecture:** A standalone module owns an editable list of `(csv_path, metric_column, training_name)` triples. Pure helpers validate and load each curve and assign unique visual styles; a plotting function renders all curves through Matplotlib's non-interactive backend and the module entry point uses the constants declared at the top of the file.

**Tech Stack:** Python 3.9, standard-library `csv`, `pathlib`, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-training-trajectory-plot-design.md`

## Global Constraints

- Configuration is edited directly in `plot_training_trajectories.py`; no command-line arguments are required.
- Every curve uses the CSV's `epoch` column for the X axis.
- Every configured triple receives a distinct `(color, linestyle)` combination.
- Missing or non-finite metric values are skipped; invalid epochs are rejected.
- The utility only reads CSV files and writes the configured PNG.
- Existing training, logging, checkpoint, and evaluation code must not change.

---

### Task 1: CSV curve loading and validation

**Files:**
- Create: `plot_training_trajectories.py`
- Create: `test_plot_training_trajectories.py`

**Interfaces:**
- Consumes: a `CurveSpec = tuple[str, str, str]` and CSV files containing `epoch` plus metric columns.
- Produces: `CurveData` with `epochs: list[float]`, `values: list[float]`, and `label: str`; `load_curve(spec: CurveSpec) -> CurveData`; `validate_curve_specs(curves: Sequence[CurveSpec]) -> None`.

- [ ] **Step 1: Write failing tests for valid data and blank/non-finite metric rows**

```python
import csv

import pytest

from plot_training_trajectories import load_curve


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "dice_mean"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_curve_uses_epoch_and_skips_missing_or_non_finite_metrics(tmp_path):
    path = tmp_path / "training_dice.csv"
    write_csv(path, [
        {"epoch": "1", "dice_mean": "0.25"},
        {"epoch": "2", "dice_mean": ""},
        {"epoch": "3", "dice_mean": "nan"},
        {"epoch": "4", "dice_mean": "0.75"},
    ])

    curve = load_curve((str(path), "dice_mean", "Baseline training"))

    assert curve.epochs == [1.0, 4.0]
    assert curve.values == [0.25, 0.75]
    assert curve.label == "Baseline training"
```

- [ ] **Step 2: Run the data-loading test and verify RED**

Run: `.venv/bin/python -m pytest test_plot_training_trajectories.py::test_load_curve_uses_epoch_and_skips_missing_or_non_finite_metrics -v`

Expected: FAIL during import because `plot_training_trajectories` does not exist.

- [ ] **Step 3: Add the minimal loader implementation**

```python
#!/usr/bin/env python3
"""Plot configured training trajectories from CSV logs."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

CurveSpec = Tuple[str, str, str]


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
```

- [ ] **Step 4: Run the loader test and verify GREEN**

Run: `.venv/bin/python -m pytest test_plot_training_trajectories.py::test_load_curve_uses_epoch_and_skips_missing_or_non_finite_metrics -v`

Expected: PASS.

- [ ] **Step 5: Write failing validation tests**

```python
from plot_training_trajectories import validate_curve_specs


def test_validate_curve_specs_rejects_empty_configuration():
    with pytest.raises(ValueError, match="at least one"):
        validate_curve_specs([])


def test_validate_curve_specs_rejects_non_triples():
    with pytest.raises(ValueError, match="three values"):
        validate_curve_specs([("file.csv", "dice_mean")])


def test_load_curve_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Training CSV"):
        load_curve((str(tmp_path / "missing.csv"), "dice_mean", "Missing"))


def test_load_curve_rejects_missing_metric_column(tmp_path):
    path = tmp_path / "metrics.csv"
    write_csv(path, [{"epoch": "1", "dice_mean": "0.5"}])
    with pytest.raises(ValueError, match="dice_ET"):
        load_curve((str(path), "dice_ET", "ET"))


def test_load_curve_rejects_invalid_epoch(tmp_path):
    path = tmp_path / "metrics.csv"
    write_csv(path, [{"epoch": "first", "dice_mean": "0.5"}])
    with pytest.raises(ValueError, match="invalid epoch"):
        load_curve((str(path), "dice_mean", "Training"))


def test_load_curve_rejects_curve_without_numeric_values(tmp_path):
    path = tmp_path / "metrics.csv"
    write_csv(path, [{"epoch": "1", "dice_mean": ""}])
    with pytest.raises(ValueError, match="no valid values"):
        load_curve((str(path), "dice_mean", "Training"))
```

- [ ] **Step 6: Run validation tests and verify RED**

Run: `.venv/bin/python -m pytest test_plot_training_trajectories.py -v`

Expected: FAIL because `validate_curve_specs` is absent.

- [ ] **Step 7: Implement curve-list validation**

```python
def validate_curve_specs(curves: Sequence[CurveSpec]) -> None:
    if not curves:
        raise ValueError("CURVES must contain at least one training curve")
    for index, spec in enumerate(curves, start=1):
        if not isinstance(spec, tuple) or len(spec) != 3:
            raise ValueError(f"CURVES item {index} must be a tuple with three values")
        if not all(isinstance(value, str) and value.strip() for value in spec):
            raise ValueError(f"CURVES item {index} values must be non-empty strings")
```

- [ ] **Step 8: Run Task 1 tests and verify GREEN**

Run: `.venv/bin/python -m pytest test_plot_training_trajectories.py -v`

Expected: all Task 1 tests PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add plot_training_trajectories.py test_plot_training_trajectories.py
git commit -m "feat: load configured training trajectories"
```

### Task 2: Distinct styles and PNG rendering

**Files:**
- Modify: `plot_training_trajectories.py`
- Modify: `test_plot_training_trajectories.py`

**Interfaces:**
- Consumes: `CurveData`, `CurveSpec`, an output `Path`, and a plot title.
- Produces: `curve_styles(count: int) -> list[tuple[str, str]]`; `plot_training_trajectories(curves: Sequence[CurveSpec], output_path: str | Path, title: str) -> Path`; `main() -> None`.

- [ ] **Step 1: Write failing tests for unique styles and rendered plot metadata**

```python
import matplotlib.pyplot as plt

from plot_training_trajectories import curve_styles, plot_training_trajectories


def test_curve_styles_are_distinct_for_every_requested_curve():
    styles = curve_styles(20)
    assert len(styles) == 20
    assert len(set(styles)) == 20


def test_plot_writes_png_with_epoch_axis_and_training_legends(tmp_path, monkeypatch):
    first = tmp_path / "training.csv"
    second = tmp_path / "validation.csv"
    write_csv(first, [{"epoch": "1", "dice_mean": "0.2"}])
    write_csv(second, [{"epoch": "1", "dice_mean": "0.3"}])
    captured = {}
    original_close = plt.close

    def capture_close(figure):
        axes = figure.axes[0]
        captured["xlabel"] = axes.get_xlabel()
        captured["ylabel"] = axes.get_ylabel()
        captured["labels"] = [line.get_label() for line in axes.lines]
        original_close(figure)

    monkeypatch.setattr(plt, "close", capture_close)
    output = tmp_path / "trajectory.png"
    result = plot_training_trajectories(
        [
            (str(first), "dice_mean", "Baseline training"),
            (str(second), "dice_mean", "Baseline validation"),
        ],
        output,
        "Dice trajectories",
    )

    assert result == output
    assert output.is_file() and output.stat().st_size > 0
    assert captured == {
        "xlabel": "Epoch",
        "ylabel": "Metric value",
        "labels": ["Baseline training", "Baseline validation"],
    }
```

- [ ] **Step 2: Run rendering tests and verify RED**

Run: `.venv/bin/python -m pytest test_plot_training_trajectories.py::test_curve_styles_are_distinct_for_every_requested_curve test_plot_training_trajectories.py::test_plot_writes_png_with_epoch_axis_and_training_legends -v`

Expected: FAIL because the style and plotting functions are absent.

- [ ] **Step 3: Implement style assignment and plotting**

```python
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = tuple(plt.get_cmap("tab10").colors)
LINESTYLES = ("-", "--", "-.", ":")


def curve_styles(count: int) -> list[tuple[str, str]]:
    available = list(itertools.product(COLORS, LINESTYLES))
    if count > len(available):
        raise ValueError(
            f"Requested {count} curves but only {len(available)} distinct styles exist"
        )
    return available[:count]


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
```

- [ ] **Step 4: Add direct file configuration and entry point**

```python
CURVES: list[CurveSpec] = [
    ("experiments/run_A/training_dice.csv", "dice_mean", "Baseline - training"),
    ("experiments/run_A/epoch_metrics.csv", "dice_mean", "Baseline - validation"),
]
OUTPUT_PATH = "training_trajectories.png"
PLOT_TITLE = "Training trajectories"


def main() -> None:
    destination = plot_training_trajectories(CURVES, OUTPUT_PATH, PLOT_TITLE)
    print(f"Saved training trajectory plot to {destination.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the plotter tests and verify GREEN**

Run: `.venv/bin/python -m pytest test_plot_training_trajectories.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Run regression and static verification**

Run: `.venv/bin/python -m py_compile plot_training_trajectories.py test_plot_training_trajectories.py`

Expected: exit code 0.

Run: `.venv/bin/python -m pytest -q test_plot_training_trajectories.py test_configurable_loss.py`

Expected: all tests PASS.

Run: `git diff --check`

Expected: exit code 0 with no output.

- [ ] **Step 7: Commit Task 2**

```bash
git add plot_training_trajectories.py test_plot_training_trajectories.py
git commit -m "feat: plot training metric trajectories"
```
