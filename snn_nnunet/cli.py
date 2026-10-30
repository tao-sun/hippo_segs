"""Thin command-line workflow over native nnU-Net v2.8.1 APIs.

Runtime variables are configured before importing nnU-Net's path and discovery
modules. Checkpoints and result plans remain the source of truth at inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Sequence
from snn_nnunet.config_aliases import SPIKING_OPTION_ALIASES


TRAINER = "nnUNetTrainerSNNFPTT"
PLANS = "SNNPlans"
CONFIGURATION = "3d_fullres"
FIVE_FOLDS = (0, 1, 2, 3, 4)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _dt_rank(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return number


def _fold(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fold must be an integer from 0 to 4") from exc
    if number not in FIVE_FOLDS:
        raise argparse.ArgumentTypeError("fold must be an integer from 0 to 4")
    return number


def _dataset_id(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dataset ID must be an integer from 0 to 999") from exc
    if not 0 <= number <= 999:
        raise argparse.ArgumentTypeError("dataset ID must be an integer from 0 to 999")
    return number


def _add_dataset_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-id", type=_dataset_id, required=True)


def _add_training_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("orig", "shallow", "medium", "deep"), default="orig")
    parser.add_argument("--temporal-axis", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--k", type=_positive_int, default=16)
    parser.add_argument("--epochs", type=_positive_int, help="total SNN training epochs; defaults to saved plans")
    parser.add_argument("--lr", type=_positive_float, help="initial learning rate; defaults to saved plans or 0.01")
    parser.add_argument("--training-patch-size", nargs=3, type=_positive_int,
                        help="training patch in preprocessed axis order; defaults to saved plans")
    parser.add_argument("--validation-every", type=_nonnegative_int,
                        help="full-fold validation with postprocessing every N completed epochs; 0 disables")
    parser.add_argument("--validation-patch-size", nargs=3, type=_positive_int,
                        help="spatial patch for periodic validation in preprocessed axis order")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fptt", dest="use_fptt", action="store_true", default=True)
    mode.add_argument("--no-fptt", dest="use_fptt", action="store_false")
    temporal_mode = parser.add_mutually_exclusive_group()
    temporal_mode.add_argument("--temporal-mamba", dest="temporal_mamba_enabled", action="store_true")
    temporal_mode.add_argument("--no-temporal-mamba", dest="temporal_mamba_enabled", action="store_false")
    parser.set_defaults(temporal_mamba_enabled=None)
    parser.add_argument("--temporal-mamba-kernel-size", type=_positive_int)
    parser.add_argument("--temporal-mamba-d-state", type=_positive_int)
    parser.add_argument("--temporal-mamba-dt-rank", type=_dt_rank)
    debug_mode = parser.add_mutually_exclusive_group()
    debug_mode.add_argument("--temporal-mamba-debug-shapes", dest="temporal_mamba_debug_shapes", action="store_true")
    debug_mode.add_argument("--no-temporal-mamba-debug-shapes", dest="temporal_mamba_debug_shapes", action="store_false")
    parser.set_defaults(temporal_mamba_debug_shapes=None)
    temporal_plif = parser.add_mutually_exclusive_group()
    temporal_plif.add_argument("--temporal-mamba-plif", dest="temporal_mamba_plif_enabled", action="store_true")
    temporal_plif.add_argument("--no-temporal-mamba-plif", dest="temporal_mamba_plif_enabled", action="store_false")
    parser.set_defaults(temporal_mamba_plif_enabled=None)
    for flag, destination in (
        ("temporal-mamba-norm-spiking", "temporal_mamba_norm_spiking"),
        ("temporal-mamba-dwconv1d-spiking", "temporal_mamba_dwconv1D_spiking"),
    ):
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{flag}", dest=destination, action="store_true")
        group.add_argument(f"--no-{flag}", dest=destination, action="store_false")
        parser.set_defaults(**{destination: None})
    for flag, destination in (
        ("spatial-patch-embedding-spiking", "spatial_patch_embedding_spiking"),
        ("spatial-dwconv2d-spiking", "spatial_dwconv2d_spiking"),
        ("spatial-output-spiking", "spatial_output_spiking"),
    ):
        help_text = (
            "Legacy option; SS2D output always uses GN without an activation."
            if destination == "spatial_output_spiking" else None
        )
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{flag}", dest=destination, action="store_true", help=help_text)
        group.add_argument(f"--no-{flag}", dest=destination, action="store_false", help=help_text)
        parser.set_defaults(**{destination: None})
    for key in ("stem_depth", "stem_channels", "downsample_kernel_size", "dwconv2d_kernel_size"):
        parser.add_argument("--" + key.replace("_", "-"), dest=key, type=_positive_int)
    for key in ("encoder_channels", "vss_depths", "downsample_strides"):
        parser.add_argument("--" + key.replace("_", "-"), dest=key, type=_positive_int, nargs=3)
    parser.add_argument("--final-skip-fusion", choices=("concat",))
    parser.add_argument("--decoder-block", choices=("standard", "resdecoder"))
    parser.add_argument("--neuron-model", choices=("plif", "silif"))
    parser.add_argument("--recurrent-neuron", type=json.loads, default=argparse.SUPPRESS, help="Optional VSS recurrent neuron JSON mapping; null selects baseline")
    parser.add_argument("--silif", type=json.loads, help="SiLIF options as a JSON mapping; stored in plans")
    for key in ("hierarchical_encoder", "full_resolution_skip", "stem_spiking", "downsampling_spiking",
                "resdecoder_spiking1", "resdecoder_spiking2", "fue_skip1", "fue_skip2",
                "encoder_output_spiking", "decoder_conv_spiking"):
        flag = key.replace("_", "-")
        legacy_flags = [legacy.replace("_", "-")
                        for legacy, canonical in SPIKING_OPTION_ALIASES.items() if canonical == key]
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--" + flag, *("--" + alias for alias in legacy_flags),
                           dest=key, action="store_true")
        group.add_argument("--no-" + flag, *("--no-" + alias for alias in legacy_flags),
                           dest=key, action="store_false")
        parser.set_defaults(**{key: None})
    final_plif = parser.add_mutually_exclusive_group()
    final_plif.add_argument("--output-spiking", "--vss-output-spiking",
                            dest="output_spiking", action="store_true")
    final_plif.add_argument("--no-output-spiking", "--no-vss-output-spiking",
                            dest="output_spiking", action="store_false")
    parser.set_defaults(output_spiking=None)


def _add_native_training_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpus", type=_positive_int, default=1)
    parser.add_argument("--save-probabilities", action="store_true")


def _add_run_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-dir", type=Path,
        help="isolated nnU-Net results directory; required with --continue, validate, and predict",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the five-command CLI with explicit, validated user inputs."""

    parser = argparse.ArgumentParser(prog="python -m snn_nnunet.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="convert/register data and run native preparation")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    _add_dataset_id(prepare)
    prepare.add_argument("--dataset-name", required=True)
    prepare.add_argument("--fingerprint-processes", type=_positive_int, default=8)
    prepare.add_argument("--preprocess-processes", type=_positive_int, default=8)

    train = commands.add_parser("train", help="train one native cross-validation fold")
    _add_dataset_id(train)
    train.add_argument("--fold", type=_fold, required=True)
    _add_training_config(train)
    _add_native_training_flags(train)
    _add_run_dir(train)
    train.add_argument("--continue", dest="continue_training", action="store_true")

    validate = commands.add_parser("validate", help="run native full-volume validation")
    _add_dataset_id(validate)
    validate.add_argument("--fold", type=_fold, required=True)
    _add_native_training_flags(validate)
    _add_run_dir(validate)
    validate.add_argument("--best", action="store_true", help="validate checkpoint_best.pth")

    predict = commands.add_parser("predict", help="predict with native sliding-window inference")
    _add_dataset_id(predict)
    predict.add_argument("--input", required=True, help="folder of raw nnU-Net channel files")
    predict.add_argument("--output", required=True)
    predict.add_argument("--folds", type=_fold, nargs="+", default=FIVE_FOLDS)
    predict.add_argument("--checkpoint", choices=("final", "best", "checkpoint_final.pth",
                                                   "checkpoint_best.pth"), default="checkpoint_final.pth")
    _add_run_dir(predict)
    predict.add_argument("--postprocess", action="store_true",
                         help="select native CV postprocessing and write OUTPUT_postprocessed")

    full_cv = commands.add_parser("full-cv", help="train folds 0–4 sequentially")
    _add_dataset_id(full_cv)
    _add_training_config(full_cv)
    _add_native_training_flags(full_cv)
    _add_run_dir(full_cv)
    full_cv.add_argument("--continue", dest="continue_training", action="store_true")
    full_cv.add_argument("--select-best", action="store_true",
                         help="run native best-configuration and postprocessing selection after CV")
    return parser


def configure_runtime() -> None:
    """Expose the installed trainer source before native import or child processes."""

    os.environ["nnUNet_extTrainer"] = str(Path(__file__).resolve().parent)
    os.environ.setdefault("nnUNet_compile", "false")


def _require_native_paths() -> None:
    missing = [name for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
               if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Required nnU-Net path variables are unset: " + ", ".join(missing))


def _configure_run_directory(args: argparse.Namespace) -> None:
    if args.command == "prepare":
        return

    requested = getattr(args, "run_dir", None)
    continuing = getattr(args, "continue_training", False)
    if continuing and requested is None:
        raise ValueError("--continue requires --run-dir pointing to the existing run")
    if requested is None and args.command in {"validate", "predict"}:
        raise ValueError(f"{args.command} requires --run-dir pointing to an existing run")

    if requested is None:
        root = Path(os.environ["nnUNet_results"]).expanduser().resolve()
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = root / stamp
        suffix = 1
        while run_dir.exists():
            run_dir = root / f"{stamp}_{suffix}"
            suffix += 1
    else:
        run_dir = Path(requested).expanduser().resolve()
        if continuing or args.command in {"validate", "predict"}:
            if not run_dir.is_dir():
                raise ValueError(f"Run directory does not exist: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    if not continuing and args.command in {"train", "full-cv"}:
        existing_results = list(run_dir.glob("Dataset*/"))
        if existing_results:
            raise ValueError(
                f"Run directory is not empty: {run_dir}; use --continue to reuse it"
            )
    args.run_dir = run_dir
    os.environ["nnUNet_results"] = str(run_dir)
    print(f"Run directory: {run_dir}")


def _dataset_name(dataset_id: int) -> str:
    from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

    return convert_id_to_dataset_name(dataset_id)


def _plans_path(dataset_id: int) -> Path:
    return Path(os.environ["nnUNet_preprocessed"]) / _dataset_name(dataset_id) / f"{PLANS}.json"


def _run_config_path(args: argparse.Namespace) -> Path:
    return Path(args.run_dir) / "run_config.json"


def _restore_run_plans(args: argparse.Namespace) -> None:
    source = _run_config_path(args)
    if not source.is_file():
        raise ValueError(f"Run configuration is missing: {source}")
    with source.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    plans = payload.get("plans")
    if not isinstance(plans, dict) or "snn_config" not in plans:
        raise ValueError(f"Run configuration has no valid plans snapshot: {source}")
    destination = _plans_path(args.dataset_id)
    temporary = destination.with_suffix(".json.run.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(plans, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, destination)


def _results_folder(dataset_id: int) -> Path:
    from nnunetv2.utilities.file_path_utilities import get_output_folder

    return Path(get_output_folder(dataset_id, TRAINER, PLANS, CONFIGURATION))


def _requested_config(args: argparse.Namespace, plans_path: Path) -> dict:
    from snn_nnunet.network_adapter import SNNConfig

    with plans_path.open(encoding="utf-8") as stream:
        config = SNNConfig.from_plans(json.load(stream)).to_dict()
    config.update(model_name=args.model, temporal_axis=args.temporal_axis,
                  k=args.k, use_fptt=args.use_fptt)
    epochs = getattr(args, "epochs", None)
    if epochs is not None:
        config["num_epochs"] = epochs
    learning_rate = getattr(args, "lr", None)
    if learning_rate is not None:
        config["initial_lr"] = learning_rate
    for key in ("validation_every", "validation_patch_size"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    temporal_names = {
        "enabled": "temporal_mamba_enabled",
        "kernel_size": "temporal_mamba_kernel_size",
        "d_state": "temporal_mamba_d_state",
        "dt_rank": "temporal_mamba_dt_rank",
        "debug_shapes": "temporal_mamba_debug_shapes",
        "norm_spiking": "temporal_mamba_norm_spiking",
        "dwconv1D_spiking": "temporal_mamba_dwconv1D_spiking",
    }
    legacy_plif = getattr(args, "temporal_mamba_plif_enabled", None)
    supplied_temporal = any(
        getattr(args, argument, None) is not None
        for argument in temporal_names.values()
    ) or legacy_plif is not None
    if supplied_temporal:
        temporal = dict(config["model_kwargs"].get("temporal_mamba", {
            "enabled": False, "kernel_size": 3, "d_state": 16,
            "dt_rank": "auto", "debug_shapes": False,
            "norm_spiking": True, "dwconv1D_spiking": True,
        }))
        if legacy_plif is not None:
            temporal["norm_spiking"] = legacy_plif
            temporal["dwconv1D_spiking"] = legacy_plif
        for key, argument in temporal_names.items():
            value = getattr(args, argument, None)
            if value is not None:
                temporal[key] = value
        config["model_kwargs"]["temporal_mamba"] = temporal
    final_spiking = getattr(args, "output_spiking", None)
    if final_spiking is not None:
        output_key = next(
            key for key in ("output_spiking", "vss_output_spiking")
            if key in config["model_kwargs"]
        )
        config["model_kwargs"][output_key] = final_spiking
    spatial_names = {
        "patch_embedding_spiking": "spatial_patch_embedding_spiking",
        "dwconv2d_spiking": "spatial_dwconv2d_spiking",
        "spatial_output_spiking": "spatial_output_spiking",
    }
    for name, argument in spatial_names.items():
        value = getattr(args, argument, None)
        if value is not None:
            config["model_kwargs"][name] = value
    from snn_nnunet.model import ARCHITECTURE_DEFAULTS, architecture_options
    for key in ARCHITECTURE_DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            config["model_kwargs"][key] = value
    if hasattr(args, "recurrent_neuron"):
        config["model_kwargs"]["recurrent_neuron"] = args.recurrent_neuron
    architecture_options(config["model_kwargs"])
    return config


def _check_saved_results(dataset_id: int) -> None:
    from snn_nnunet.prepare_plans import validate_results_compatibility

    with _plans_path(dataset_id).open(encoding="utf-8") as stream:
        plans = json.load(stream)
    validate_results_compatibility(_results_folder(dataset_id), plans)


def _persist_training_config(args: argparse.Namespace) -> None:
    from snn_nnunet.prepare_plans import update_snn_config

    if args.continue_training:
        _restore_run_plans(args)
    plans_path = _plans_path(args.dataset_id)
    update_snn_config(
        plans_path, _requested_config(args, plans_path), _results_folder(args.dataset_id),
        training_patch_size=getattr(args, "training_patch_size", None),
    )


def _save_run_config(args: argparse.Namespace) -> None:
    if not hasattr(args, "run_dir") or args.run_dir is None:
        return
    plans_path = _plans_path(args.dataset_id)
    with plans_path.open(encoding="utf-8") as stream:
        plans = json.load(stream)
    payload = {
        "command": args.command,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(args.run_dir),
        "dataset_id": args.dataset_id,
        "trainer": TRAINER,
        "plans_identifier": PLANS,
        "configuration": CONFIGURATION,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "snn_config": plans["snn_config"],
        "plans": plans,
    }
    destination = args.run_dir / "run_config.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, destination)


def _run_training(args: argparse.Namespace, fold: int, *, validation: bool) -> None:
    from nnunetv2.run.run_training import run_training
    from snn_nnunet.empty_dice_evaluation import finalize_fold_summary

    if validation:
        run_training(dataset_name_or_id=str(args.dataset_id), configuration=CONFIGURATION, fold=fold,
                     trainer_class_name=TRAINER, plans_identifier=PLANS, num_gpus=args.gpus,
                     continue_training=False, only_run_validation=True, val_with_best=args.best,
                     export_validation_probabilities=args.save_probabilities)
    else:
        command = [sys.executable, "-m", "accelerate.commands.launch",
                   "--num_processes", str(args.gpus), "--num_machines", "1",
                   "--mixed_precision", "no", "--dynamo_backend", "no"]
        if args.gpus > 1:
            command.extend(["--multi_gpu", "--main_process_port", "0"])
        command.extend(["--module", "snn_nnunet.distributed_training",
                        "--dataset-id", str(args.dataset_id), "--fold", str(fold)])
        if args.continue_training:
            command.append("--continue-training")
        if args.save_probabilities:
            command.append("--save-probabilities")
        print(f"Accelerate: {args.gpus} process(es), one per GPU, FP32", flush=True)
        subprocess.run(command, check=True)
        _run_custom_full_validation(args, fold)

    model_folder = _results_folder(args.dataset_id)
    summary_path = model_folder / f"fold_{fold}" / "validation" / "summary.json"
    if summary_path.is_file():
        metrics = finalize_fold_summary(model_folder, fold)
        print(
            "Mean Validation Dice (empty-empty=1): "
            f"{metrics['foreground_mean']['Dice']}",
            flush=True,
        )


def _run_custom_full_validation(args: argparse.Namespace, fold: int) -> None:
    """Run the custom FP32 128-cubed validation after a fold finishes training."""

    from snn_nnunet.custom_full_validation import main as run_custom_validation

    output_dir = (
        _results_folder(args.dataset_id)
        / f"fold_{fold}"
        / "validation_128patch_postprocessed"
    )
    run_custom_validation([
        "--dataset-id", str(args.dataset_id),
        "--fold", str(fold),
        "--results-root", str(args.run_dir),
        "--output-dir", str(output_dir),
        "--configuration", CONFIGURATION,
        "--trainer", TRAINER,
        "--plans", PLANS,
        "--checkpoint-name", "checkpoint_final.pth",
    ])


def _select_best(dataset_id: int) -> dict:
    from nnunetv2.evaluation.find_best_configuration import find_best_configuration

    return find_best_configuration(
        dataset_id,
        allowed_trained_models=({"plans": PLANS, "configuration": CONFIGURATION, "trainer": TRAINER},),
        allow_ensembling=False, folds=FIVE_FOLDS, strict=True,
    )


def _normalize_checkpoint(value: str) -> str:
    return {"final": "checkpoint_final.pth", "best": "checkpoint_best.pth"}.get(value, value)


def _postprocess(dataset_id: int, output: str) -> None:
    from batchgenerators.utilities.file_and_folder_operations import load_pickle
    from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder

    selected = _select_best(dataset_id)["best_model_or_ensemble"]
    expected = {"configuration": CONFIGURATION, "trainer": TRAINER, "plans_identifier": PLANS}
    if selected["selected_model_or_models"] != [expected]:
        raise ValueError("Native postprocessing selection did not resolve to the requested SNN model")
    functions, kwargs = load_pickle(selected["postprocessing_file"])
    apply_postprocessing_to_folder(output, output + "_postprocessed", functions, kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a user command and return a process-style success code."""

    args = build_parser().parse_args(argv)
    configure_runtime()
    _require_native_paths()
    _configure_run_directory(args)

    if args.command == "prepare":
        from snn_nnunet.dataset_conversion import convert_or_register_dataset
        from snn_nnunet.prepare_plans import run_native_prepare

        convert_or_register_dataset(args.dataset_root, args.dataset_id, args.dataset_name,
                                    Path(os.environ["nnUNet_raw"]))
        run_native_prepare(args.dataset_id, args.fingerprint_processes, args.preprocess_processes)
    elif args.command == "train":
        _persist_training_config(args)
        _save_run_config(args)
        _run_training(args, args.fold, validation=False)
    elif args.command == "validate":
        _restore_run_plans(args)
        _check_saved_results(args.dataset_id)
        _run_training(args, args.fold, validation=True)
    elif args.command == "full-cv":
        _persist_training_config(args)
        _save_run_config(args)
        for fold in FIVE_FOLDS:
            _run_training(args, fold, validation=False)
        if args.select_best:
            _select_best(args.dataset_id)
    elif args.command == "predict":
        from snn_nnunet.inference import FP32nnUNetPredictor as nnUNetPredictor

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True)
        predictor.initialize_from_trained_model_folder(
            str(_results_folder(args.dataset_id)), use_folds=tuple(args.folds),
            checkpoint_name=_normalize_checkpoint(args.checkpoint),
        )
        predictor.predict_from_files(args.input, args.output)
        if args.postprocess:
            _postprocess(args.dataset_id, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
