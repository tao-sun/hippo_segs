"""Run native nnU-Net validation directly from checkpoint_best.pth."""

import argparse
from pathlib import Path
from typing import Any, Sequence


def validate_best_checkpoint(trainer: Any, checkpoint: Path) -> None:
    """Load an existing best checkpoint and run native full-volume validation."""

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}")
    trainer.load_checkpoint(str(checkpoint))
    trainer.perform_actual_validation(save_probabilities=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an nnU-Net fold using checkpoint_best.pth without requiring checkpoint_final.pth"
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    from nnunetv2.run.run_training import get_trainer_from_args

    args = build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = get_trainer_from_args(
        args.dataset_id,
        args.configuration,
        args.fold,
        args.trainer,
        args.plans,
        False,
        device=device,
    )
    checkpoint = Path(trainer.output_folder) / "checkpoint_best.pth"
    print(f"Validating checkpoint: {checkpoint}", flush=True)
    validate_best_checkpoint(trainer, checkpoint)
    from snn_nnunet.empty_dice_evaluation import finalize_summary

    metrics = finalize_summary(Path(trainer.output_folder) / "validation" / "summary.json")
    print(
        "Mean Validation Dice (empty-empty=1): ",
        metrics["foreground_mean"]["Dice"],
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
