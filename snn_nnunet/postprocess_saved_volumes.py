"""Apply the project's BraTS morphology to saved nnU-Net segmentations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from snn_nnunet.custom_full_validation import (
    REGION_ORDER,
    dice_metrics,
    labelmap_to_regions,
    postprocess_regions,
    print_dice_summary,
    summarize_cases,
)


def _regions_to_labelmap(regions: np.ndarray, label_manager) -> np.ndarray:
    import torch

    masks = torch.as_tensor(regions, dtype=torch.bool)
    logits = torch.where(masks, 20.0, -20.0)
    segmentation = label_manager.convert_logits_to_segmentation(logits)
    if hasattr(segmentation, "cpu"):
        segmentation = segmentation.cpu().numpy()
    return np.asarray(segmentation)


def postprocess_saved_volumes(
    input_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    plans_json: Path,
    dataset_json_path: Path,
    min_component_sizes=(161, 166, 50),
    closing_radius=1,
) -> dict:
    """Postprocess saved BraTS segmentations using the custom validation rules."""
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    with dataset_json_path.open(encoding="utf-8") as stream:
        dataset_json = json.load(stream)
    plans_manager = PlansManager(str(plans_json))
    reader_writer = plans_manager.image_reader_writer_class()
    label_manager = plans_manager.get_label_manager(dataset_json)
    file_ending = dataset_json["file_ending"]

    prediction_files = sorted(input_dir.glob(f"*{file_ending}"))
    if not prediction_files:
        raise FileNotFoundError(f"No {file_ending} segmentations found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    case_results = []
    for index, prediction_file in enumerate(prediction_files, start=1):
        case_id = prediction_file.name[:-len(file_ending)]
        reference_file = reference_dir / prediction_file.name
        if not reference_file.is_file():
            raise FileNotFoundError(f"Reference segmentation not found: {reference_file}")

        prediction_data, properties = reader_writer.read_seg(str(prediction_file))
        reference_data, _ = reader_writer.read_seg(str(reference_file))
        if prediction_data.shape[0] != 1 or reference_data.shape[0] != 1:
            raise ValueError(f"Expected one segmentation channel for {case_id}")

        raw_labelmap = prediction_data[0]
        reference_labelmap = reference_data[0]
        if raw_labelmap.shape != reference_labelmap.shape:
            raise ValueError(f"Prediction/reference shape mismatch for {case_id}")

        raw_regions = labelmap_to_regions(raw_labelmap)
        processed_regions = postprocess_regions(
            raw_regions,
            min_component_sizes_et_tc_wt=min_component_sizes,
            closing_radius=closing_radius,
        )
        processed_labelmap = _regions_to_labelmap(processed_regions, label_manager)
        canonical_regions = labelmap_to_regions(processed_labelmap)

        case_results.append({
            "case": case_id,
            "raw": dice_metrics(labelmap_to_regions(reference_labelmap), raw_regions),
            "postprocessed": dice_metrics(
                labelmap_to_regions(reference_labelmap), canonical_regions
            ),
        })
        output_file = output_dir / prediction_file.name
        reader_writer.write_seg(processed_labelmap, str(output_file), properties)
        print(f"[{index}/{len(prediction_files)}] {case_id}", flush=True)

    summary = summarize_cases(case_results)
    summary["settings"] = {
        "postprocessing": "custom BraTS 3D component filtering and morphological closing",
        "min_component_sizes_et_tc_wt": [int(value) for value in min_component_sizes],
        "closing_radius": int(closing_radius),
        "regions": list(REGION_ORDER),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "stage", "region", "dice", "gt_voxels", "pred_voxels"))
        for case in case_results:
            for stage in ("raw", "postprocessed"):
                for region in REGION_ORDER:
                    metrics = case[stage][region]
                    writer.writerow((
                        case["case"], stage, region, metrics["dice"],
                        metrics["gt_voxels"], metrics["pred_voxels"],
                    ))
    print_dice_summary(summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plans-json", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument(
        "--min-component-sizes", nargs=3, type=int, default=(161, 166, 50),
        metavar=("ET", "TC", "WT"),
    )
    parser.add_argument("--closing-radius", type=int, default=1)
    args = parser.parse_args(argv)
    postprocess_saved_volumes(
        args.input_dir,
        args.reference_dir,
        args.output_dir,
        args.plans_json,
        args.dataset_json,
        args.min_component_sizes,
        args.closing_radius,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
