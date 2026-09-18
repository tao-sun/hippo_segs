#!/usr/bin/env python3
"""Plot subject-level BraTS label presence for every slice in one view."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from check_crop_label_loss import (
    LABEL_NAMES,
    LABELS,
    TARGET_SHAPE,
    center_crop_slices,
    find_segmentations,
    load_segmentation,
    select_segmentations,
)


VIEW_AXES = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}
LABEL_COLORS = {
    1: (230, 38, 26),
    2: (26, 191, 64),
    3: (26, 89, 242),
    4: (255, 191, 13),
}


def colorize_ground_truth(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D ground-truth slice, got {mask.shape}")
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        colored[mask == label] = color
    return colored


def normalize_modality_slice(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32, copy=False))
    nonzero = image[image != 0]
    if nonzero.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(nonzero, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    normalized = (image - float(low)) / float(high - low)
    normalized[image == 0] = 0
    return np.clip(normalized, 0.0, 1.0)


def overlay_ground_truth(
    modality: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    if modality.shape != mask.shape or modality.ndim != 2:
        raise ValueError(
            "Modality and ground truth must be matching 2D arrays, got "
            f"{modality.shape} and {mask.shape}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")

    grayscale = np.repeat(
        np.clip(modality, 0.0, 1.0)[..., None],
        3,
        axis=-1,
    )
    result = grayscale.copy()
    for label, color in LABEL_COLORS.items():
        selected = mask == label
        result[selected] = (
            (1.0 - alpha) * grayscale[selected]
            + alpha * np.asarray(color, dtype=np.float32) / 255.0
        )
    return result


def _add_label_legend(image: Image.Image) -> Image.Image:
    padding = 10
    legend_width = 135
    swatch_size = 14
    row_height = 25
    title_height = 25
    legend_height = 2 * padding + title_height + len(LABELS) * row_height
    canvas = Image.new(
        "RGB",
        (image.width + legend_width, max(image.height, legend_height)),
        (24, 24, 24),
    )
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    legend_x = image.width + padding
    draw.text(
        (legend_x, padding),
        "Ground truth",
        fill=(255, 255, 255),
        font=font,
    )
    for row, label in enumerate(LABELS):
        y = padding + title_height + row * row_height
        draw.rectangle(
            (legend_x, y, legend_x + swatch_size, y + swatch_size),
            fill=LABEL_COLORS[label],
        )
        draw.text(
            (legend_x + swatch_size + 8, y - 1),
            f"{label}: {LABEL_NAMES[label]}",
            fill=(255, 255, 255),
            font=font,
        )
    return canvas


def _save_slice_overlay(
    modality_slice: np.ndarray,
    mask: np.ndarray,
    output: Path,
) -> None:
    normalized = normalize_modality_slice(modality_slice)
    overlay = overlay_ground_truth(normalized, mask)
    image = Image.fromarray(np.rint(overlay * 255.0).astype(np.uint8))
    output.parent.mkdir(parents=True, exist_ok=True)
    _add_label_legend(image).save(output)


def find_modality_path(subject_dir: Path, modality: str) -> Path:
    matches = sorted(subject_dir.glob(f"*-{modality}.nii.gz"))
    if not matches:
        matches = sorted(subject_dir.glob(f"*_{modality}.nii.gz"))
    if not matches:
        raise FileNotFoundError(
            f"Missing modality {modality} in {subject_dir}"
        )
    return matches[0]


def _save_removed_ground_truth(
    modality_slice: np.ndarray,
    mask: np.ndarray,
    subject: str,
    view: str,
    modality: str,
    slice_index: int,
    output: Path,
) -> None:
    normalized = normalize_modality_slice(modality_slice)
    overlay = overlay_ground_truth(normalized, mask)
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("Cannot save a removed slice without non-zero labels")

    padding = 20
    row_start = max(0, int(rows.min()) - padding)
    row_stop = min(mask.shape[0], int(rows.max()) + padding + 1)
    column_start = max(0, int(columns.min()) - padding)
    column_stop = min(mask.shape[1], int(columns.max()) + padding + 1)

    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    for axis in axes:
        axis.imshow(overlay, interpolation="nearest")
        axis.axis("off")
    axes[0].set_title(f"Full {modality} with colored ground truth")
    axes[0].add_patch(
        Rectangle(
            (column_start - 0.5, row_start - 0.5),
            column_stop - column_start,
            row_stop - row_start,
            fill=False,
            edgecolor="white",
            linewidth=1.5,
        )
    )
    axes[1].set_title("Automatic zoom around non-zero labels")
    axes[1].set_xlim(column_start - 0.5, column_stop - 0.5)
    axes[1].set_ylim(row_stop - 0.5, row_start - 0.5)

    figure.suptitle(
        f"{subject} | {view} slice {slice_index} | removed by crop"
    )
    figure.legend(
        handles=[
            Patch(
                facecolor=np.asarray(LABEL_COLORS[label]) / 255.0,
                label=f"{label}: {LABEL_NAMES[label]}",
            )
            for label in LABELS
        ],
        loc="upper right",
        framealpha=0.9,
    )
    figure.tight_layout(rect=(0, 0, 0.9, 0.94))
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def count_subjects_by_slice(
    segmentations: Iterable[np.ndarray],
    view: str,
    labels: Sequence[int] = LABELS,
) -> dict[int, np.ndarray]:
    if view not in VIEW_AXES:
        raise ValueError(f"view must be one of {sorted(VIEW_AXES)}")
    axis = VIEW_AXES[view]
    counts = None
    slice_count = None

    for segmentation in segmentations:
        if segmentation.ndim != 3:
            raise ValueError(
                f"Expected a 3D segmentation, got shape {segmentation.shape}"
            )
        if slice_count is None:
            slice_count = int(segmentation.shape[axis])
            counts = {
                int(label): np.zeros(slice_count, dtype=np.int64)
                for label in labels
            }
        elif segmentation.shape[axis] != slice_count:
            raise ValueError(
                "All segmentations must have the same number of slices "
                f"for view {view}"
            )

        other_axes = tuple(index for index in range(3) if index != axis)
        for label in labels:
            present = np.any(segmentation == int(label), axis=other_axes)
            counts[int(label)] += present.astype(np.int64)

    if counts is None:
        raise ValueError("At least one segmentation is required")
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/aurora/data/BRATS2024"),
        help="Root containing the original BraTS segmentation files",
    )
    parser.add_argument(
        "--view",
        choices=sorted(VIEW_AXES),
        required=True,
        help="Anatomical view whose slice axis will be analyzed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: ROOT/analysis/label_presence_VIEW)",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        default=20,
        help="Number of randomly selected subjects; use 0 for all subjects",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--modality",
        choices=("t1n", "t1c", "t2w", "t2f"),
        default="t2f",
        help="MRI modality used as background for removed-slice images",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=TARGET_SHAPE,
        help="Center-crop target shape, used to mark removed slice ranges",
    )
    parser.add_argument(
        "--save-all-overlays",
        action="store_true",
        help=(
            "Save every MRI/GT slice only for subjects whose crop removes "
            "at least one non-zero label voxel"
        ),
    )
    return parser


def _save_histogram(
    counts: np.ndarray,
    label: int,
    view: str,
    subject_count: int,
    crop_start: int,
    crop_stop: int,
    output: Path,
) -> None:
    slice_indices = np.arange(counts.size)
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.bar(slice_indices, counts, width=1.0, color="#1565c0")

    if crop_start > 0:
        axis.axvspan(
            -0.5,
            crop_start - 0.5,
            color="#c62828",
            alpha=0.18,
            label="Removed by crop",
        )
    if crop_stop < counts.size:
        axis.axvspan(
            crop_stop - 0.5,
            counts.size - 0.5,
            color="#c62828",
            alpha=0.18,
            label="Removed by crop",
        )
    axis.axvline(crop_start - 0.5, color="#c62828", linestyle="--", linewidth=1)
    axis.axvline(crop_stop - 0.5, color="#c62828", linestyle="--", linewidth=1)
    axis.set_xlim(-0.5, counts.size - 0.5)
    axis.set_ylim(0, max(1, subject_count))
    axis.set_xlabel(f"{view.capitalize()} slice index in original volume")
    axis.set_ylabel(f"Subjects containing label {label}")
    axis.set_title(
        f"Label {label} ({LABEL_NAMES[label]}) by {view} slice "
        f"-- {subject_count} subjects"
    )
    axis.grid(axis="y", alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys())
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    paths = select_segmentations(
        find_segmentations(args.root),
        args.subjects,
        args.seed,
    )
    output = (
        args.output
        if args.output is not None
        else args.root / "analysis" / f"label_presence_{args.view}"
    )
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    shapes = []
    removed_dir = output / "removed_slices"
    removed_dir.mkdir(parents=True, exist_ok=True)
    removed_images = []
    all_slice_images = []
    all_slices_dir = output / "all_slices"
    axis_index = VIEW_AXES[args.view]

    def selected_segmentations():
        for path in paths:
            segmentation = load_segmentation(path)
            crop_axis = center_crop_slices(
                segmentation.shape,
                args.target_shape,
            )[axis_index]
            shapes.append(tuple(int(size) for size in segmentation.shape))

            other_axes = tuple(index for index in range(3) if index != axis_index)
            positive_slices = np.any(segmentation != 0, axis=other_axes)
            inside_crop = np.zeros(positive_slices.size, dtype=bool)
            inside_crop[crop_axis] = True
            removed_indices = np.flatnonzero(positive_slices & ~inside_crop)
            modality_volume = None
            if removed_indices.size:
                modality_path = find_modality_path(path.parent, args.modality)
                modality_volume = nib.load(str(modality_path)).get_fdata(
                    dtype=np.float32
                )
                if modality_volume.shape != segmentation.shape:
                    raise ValueError(
                        f"Modality shape {modality_volume.shape} does not match "
                        f"segmentation shape {segmentation.shape}: {modality_path}"
                    )

            if args.save_all_overlays and removed_indices.size:
                subject = path.parent.name
                subject_dir = all_slices_dir / subject
                for slice_index in range(segmentation.shape[axis_index]):
                    mask = np.take(
                        segmentation, slice_index, axis=axis_index
                    ).T
                    modality_slice = np.take(
                        modality_volume, slice_index, axis=axis_index
                    ).T
                    image_path = subject_dir / (
                        f"{subject}_{args.view}_{slice_index:03d}_"
                        f"{args.modality}.png"
                    )
                    _save_slice_overlay(modality_slice, mask, image_path)
                    all_slice_images.append(image_path)

            for slice_index in removed_indices:
                slice_index = int(slice_index)
                mask = np.take(
                    segmentation,
                    slice_index,
                    axis=axis_index,
                ).T
                modality_slice = np.take(
                    modality_volume,
                    slice_index,
                    axis=axis_index,
                ).T
                image_path = (
                    removed_dir
                    / (
                        f"{path.parent.name}_{args.view}_{slice_index:03d}_"
                        f"{args.modality}.png"
                    )
                )
                _save_removed_ground_truth(
                    modality_slice,
                    mask,
                    path.parent.name,
                    args.view,
                    args.modality,
                    slice_index,
                    image_path,
                )
                removed_images.append(image_path)
                print(f"Saved removed MRI/GT slice: {image_path}")

            yield segmentation

    counts = count_subjects_by_slice(selected_segmentations(), args.view)
    source_shape = shapes[0]
    if any(shape != source_shape for shape in shapes):
        raise ValueError(
            "All selected segmentations must have the same 3D shape to "
            "compare identical slice indices"
        )
    crop_axis = center_crop_slices(source_shape, args.target_shape)[axis_index]
    crop_start = int(crop_axis.start)
    crop_stop = int(crop_axis.stop)

    csv_path = output / f"{args.view}_subject_counts_by_slice.csv"
    fieldnames = [
        "slice_index",
        "inside_crop",
        *[f"label_{label}_subjects" for label in LABELS],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for slice_index in range(counts[LABELS[0]].size):
            row = {
                "slice_index": slice_index,
                "inside_crop": crop_start <= slice_index < crop_stop,
            }
            row.update(
                {
                    f"label_{label}_subjects": int(counts[label][slice_index])
                    for label in LABELS
                }
            )
            writer.writerow(row)

    for label in LABELS:
        plot_path = output / f"{args.view}_label_{label}.png"
        _save_histogram(
            counts[label],
            label,
            args.view,
            len(paths),
            crop_start,
            crop_stop,
            plot_path,
        )

    (output / "selected_subjects.txt").write_text(
        "\n".join(str(path) for path in paths) + "\n",
        encoding="utf-8",
    )
    print(f"View: {args.view}")
    print(f"Subjects analyzed: {len(paths)}")
    print(f"Source shape: {source_shape}")
    print(
        f"Crop keeps {args.view} slices [{crop_start}, {crop_stop}) "
        f"and removes {crop_start + source_shape[axis_index] - crop_stop}"
    )
    for label in LABELS:
        slices_with_label = int(np.count_nonzero(counts[label]))
        outside = np.concatenate(
            [counts[label][:crop_start], counts[label][crop_stop:]]
        )
        outside_slices = int(np.count_nonzero(outside))
        print(
            f"Label {label} ({LABEL_NAMES[label]}): "
            f"present in {slices_with_label} slice indices; "
            f"{outside_slices} removed slice indices contain the label"
        )
    print(f"Colored GT slices removed by crop: {len(removed_images)}")
    if removed_images:
        print(f"Removed GT directory: {removed_dir}")
    if args.save_all_overlays:
        print(f"All MRI/GT overlay slices: {len(all_slice_images)}")
        print(f"All slices directory: {all_slices_dir}")
    print(f"CSV: {csv_path}")
    print(f"Histograms: {output}")


if __name__ == "__main__":
    main()
