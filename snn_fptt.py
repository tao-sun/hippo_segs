#!/usr/bin/env python3
# dnn_3ch.py — SNN (TBPTT) 2D-train / 3D-eval for BraTS (ET/TC/WT multilabel)

import os

# Must be set before PyTorch initializes CUDA/CUBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import re
import random
import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import csv
import shutil
import warnings
import uuid

import numpy as np
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
import yaml

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

# ==== use your spiking UNet-like model ====
# SNNBraTS: forward(x_win[B,k,4,H,W], t0) -> (B, out_channels, k, H, W)
from model import SNNBraTS, SNNBraTSUNetShallow, SNNBraTSUNetMedium, SNNBraTSUNetDeep, print_model_info  # mirrors your SNN implementation with PLIF nodes

# ------------------ SEEDING ------------------
SEED = 2025

# Python random
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# PyTorch
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make sure deterministic algorithms are used
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

# Ensure reproducible hashing (affects dataloader shuffling, etc.)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ---- Mamba / selective_scan availability check ----
MAMBA_SELECTIVE_SCAN_AVAILABLE = False
MAMBA_SELECTIVE_SCAN_ERROR = None
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_SELECTIVE_SCAN_AVAILABLE = selective_scan_fn is not None
except Exception as exc:  # pragma: no cover - startup diagnostics only
    MAMBA_SELECTIVE_SCAN_AVAILABLE = False
    MAMBA_SELECTIVE_SCAN_ERROR = repr(exc)

if MAMBA_SELECTIVE_SCAN_AVAILABLE:
    print(f"[MAMBA] selective_scan_cuda is available: {selective_scan_fn}")
else:
    print("[MAMBA] selective_scan_cuda is NOT available.")
    if MAMBA_SELECTIVE_SCAN_ERROR:
        print(f"[MAMBA] Import error: {MAMBA_SELECTIVE_SCAN_ERROR}")

# For dataloaders with multiple workers
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# --------------------------------------------


# -----------------------------
# Constants & helpers
# -----------------------------
VALID_VIEWS = {"sagittal", "coronal", "axial"}
MOD_ORDER = ["t1", "t1ce", "t2", "flair"]   # 4 input channels
MOD_ORDER = ["t1", "t1ce", "t2", "flair"]  # keep canonical names
ALIASES = {
    "t1":   ["t1", "t1n"],
    "t1ce": ["t1ce", "t1c"],   # <- handle both spellings
    "t2":   ["t2", "t2w"],
    "flair":["flair", "t2f"],
}
OUT_TOKENS = ["et", "tc", "wt"]             # 3 output channels
TARGET_SHAPE = (160, 192, 152)              # (x,y,z) used by preprocessing
FOLD_NAMES = {"1", "2", "3", "4", "5"}

DEFAULT_EXPERIMENTS_YAML = "experiments_snn_fptt.yaml"
RUNS_ROOT = Path("experiments")
CACHE_FORMAT_VERSION = 1


def reduce_loss_totals(accelerator: Accelerator,
                       loss_sum: torch.Tensor,
                       update_count: torch.Tensor) -> float:
    global_loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    global_update_count = accelerator.reduce(update_count, reduction="sum")
    return (global_loss_sum / global_update_count.clamp_min(1)).item()


def reduce_dice_totals(accelerator: Accelerator,
                       dice_sum: torch.Tensor,
                       subject_count: torch.Tensor) -> Dict[str, float]:
    global_dice_sum = accelerator.reduce(dice_sum, reduction="sum")
    global_subject_count = accelerator.reduce(subject_count, reduction="sum")
    mean_per_class = global_dice_sum / global_subject_count.clamp_min(1)
    return {
        "dice_ET": mean_per_class[0].item(),
        "dice_TC": mean_per_class[1].item(),
        "dice_WT": mean_per_class[2].item(),
        "dice_mean": mean_per_class.mean().item(),
        "n_subjects": int(global_subject_count.item()),
    }


def get_tmux_session_name() -> Optional[str]:
    tmux = os.environ.get("TMUX")
    pane = os.environ.get("TMUX_PANE")
    if not tmux or not pane:
        return None
    try:
        result = subprocess.run(
            ["tmux", "-S", tmux.rsplit(",", 2)[0], "display-message",
             "-p", "-t", pane, "#{session_name}"],
            capture_output=True, text=True, check=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def validate_checkpoint_input_skip(checkpoint: Dict, input_skip: bool) -> None:
    saved = checkpoint.get("config", {}).get("input_skip", False)
    if saved != input_skip:
        raise ValueError(
            f"Checkpoint input_skip={saved} differs from requested input_skip={input_skip}; "
            "start a new run with resume_from: null when changing architecture."
        )


def create_training_scheduler(optimizer, name: str):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=92, eta_min=1e-8,
        )
    if name == "reduce_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min"
        )
    raise ValueError("scheduler must be 'cosine' or 'reduce_plateau'")


def validate_checkpoint_scheduler(checkpoint: Dict, requested: str) -> None:
    saved = checkpoint.get("config", {}).get("scheduler")
    if saved is None:
        # Older checkpoints did not record the scheduler name.
        state = checkpoint.get("scheduler", {})
        if "T_max" in state:
            saved = "cosine"
        elif "num_bad_epochs" in state:
            saved = "reduce_plateau"
    if saved != requested:
        raise ValueError(
            f"Checkpoint scheduler={saved!r} differs from requested {requested!r}; "
            "set resume_scheduler: false to start the selected scheduler from the checkpoint LR."
        )


def step_training_scheduler(scheduler, train_loss: float) -> None:
    # This scheduler advances once per epoch. Accelerate's optimizer-step
    # wrapper can repeat step() once per process, shortening epoch schedules.
    base_scheduler = getattr(scheduler, "scheduler", scheduler)
    if isinstance(
        base_scheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ):
        base_scheduler.step(train_loss)
    else:
        base_scheduler.step()


def save_training_checkpoint_if_main(accelerator: Accelerator,
                                     model: nn.Module,
                                     optimizer,
                                     scheduler,
                                     checkpoint_path: Path,
                                     epoch: int,
                                     best_dice: float,
                                     best_dice_epoch: int,
                                     config: Dict,
                                     train_generator: torch.Generator) -> bool:
    if not accelerator.is_main_process:
        return False

    base_model = accelerator.unwrap_model(model)
    checkpoint = {
        "format_version": 1,
        "model": base_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "best_dice_epoch": best_dice_epoch,
        "config": config,
        "fptt_avg_weights": {
            name: value.detach().cpu()
            for name, value in base_model.avg_weights.items()
        },
        "fptt_lambdas": {
            name: value.detach().cpu()
            for name, value in base_model.lambdas.items()
        },
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "train_generator_state": train_generator.get_state(),
    }

    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    accelerator.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    return True


def validate_checkpoint_training_mode(checkpoint: Dict, use_fptt: bool) -> None:
    saved_mode = checkpoint.get("config", {}).get("use_fptt", True)
    if saved_mode != use_fptt:
        raise ValueError(
            f"Checkpoint use_fptt={saved_mode} differs from requested use_fptt={use_fptt}; "
            "start a new run with resume_from: null."
        )


def load_training_checkpoint(accelerator: Accelerator,
                             model: nn.Module,
                             optimizer,
                             scheduler,
                             checkpoint_path: Path,
                             train_generator: torch.Generator,
                             resume_scheduler: bool = True,
                             use_fptt: bool = True,
                             resume_lr: Optional[float] = None) -> Dict:
    if resume_lr is not None and resume_scheduler:
        raise ValueError("resume_lr requires resume_scheduler=false")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    validate_checkpoint_training_mode(checkpoint, use_fptt)
    required = {
        "model", "optimizer", "epoch", "best_dice",
        "best_dice_epoch", "fptt_avg_weights", "fptt_lambdas",
    }
    if resume_scheduler:
        required.add("scheduler")
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is not resumable; missing: "
            f"{', '.join(missing)}"
        )

    if resume_scheduler:
        base_scheduler = getattr(scheduler, "scheduler", scheduler)
        if isinstance(base_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            validate_checkpoint_scheduler(checkpoint, "cosine")
        elif isinstance(base_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            validate_checkpoint_scheduler(checkpoint, "reduce_plateau")

    base_model = accelerator.unwrap_model(model)
    validate_checkpoint_input_skip(checkpoint, getattr(base_model, "input_skip", False))
    base_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if resume_scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        base_scheduler = getattr(scheduler, "scheduler", scheduler)
        if resume_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(resume_lr)
        restart_lrs = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        if hasattr(base_scheduler, "base_lrs"):
            base_scheduler.base_lrs = restart_lrs
        if hasattr(base_scheduler, "_last_lr"):
            base_scheduler._last_lr = restart_lrs
        for group, restart_lr in zip(optimizer.param_groups, restart_lrs):
            group["initial_lr"] = restart_lr

    parameters = dict(base_model.named_parameters())
    for state_name, checkpoint_key in (
        ("avg_weights", "fptt_avg_weights"),
        ("lambdas", "fptt_lambdas"),
    ):
        saved_state = checkpoint[checkpoint_key]
        missing_parameters = sorted(set(parameters) - set(saved_state))
        if missing_parameters:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has incomplete {state_name}: "
                f"missing {', '.join(missing_parameters)}"
            )
        setattr(base_model, state_name, {
            name: saved_state[name].to(device=param.device, dtype=param.dtype)
            for name, param in parameters.items()
        })

    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if "train_generator_state" in checkpoint:
        train_generator.set_state(checkpoint["train_generator_state"])

    return checkpoint


class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def ensure_train_root(p: Path) -> Path:
    """
    Accept either:
      - .../BRATS2017_preprocessed/Brats17TrainingData  (preferred), or
      - .../BRATS2017_preprocessed (we'll append Brats17TrainingData if present)
    """
    if (p / "Brats17TrainingData").exists():
        return p / "Brats17TrainingData"
    return p


def find_subject_dirs(train_root: Path, folds: List[int], verbose: bool = True) -> List[Path]:
    """
    Collect subject directories under given folds (e.g., .../Brats17TrainingData/1/<subject>).
    Logs what it finds.
    """
    subjects: List[Path] = []
    for f in folds:
        fdir = train_root / str(f)
        if not fdir.exists():
            if verbose:
                print(f"[WARN] Fold directory missing: {fdir}")
            continue
        fold_subjs = sorted(p for p in fdir.iterdir() if p.is_dir())
        if verbose:
            print(f"[INFO] Fold {f}: found {len(fold_subjs)} subject dirs under {fdir}")
        subjects.extend(fold_subjs)
    if verbose and not subjects:
        print(f"[ERROR] No subject dirs found in folds={folds} under {train_root}")
    return subjects


def limit_subject_dirs(subjects: List[Path], limit: Optional[int], label: str) -> List[Path]:
    """Return the first `limit` sorted subject directories, or all subjects when limit is None."""
    if limit is None:
        return subjects

    limit = int(limit)
    if limit <= 0:
        raise ValueError("subjects_per_fold must be a positive integer when set")
    if limit > len(subjects):
        raise ValueError(f"subjects_per_fold={limit} exceeds available subjects={len(subjects)} for {label}")
    return subjects[:limit]


def brats_to_multilabel(mask3d: np.ndarray, label_format: str = "auto") -> np.ndarray:
    """
    Convert BraTS integer labels to multilabel [ET, TC, WT].
    ``label_format`` should be explicit when the dataset is known. ``auto`` is
    retained for backwards compatibility with unambiguous masks.
    Returns (3, X, Y, Z) float32 in {0,1}.
    """
    m = mask3d.astype(np.int32)
    if label_format not in {"auto", "brats17", "brats23", "brats24"}:
        raise ValueError(f"Unknown label format: {label_format}")

    if label_format == "auto":
        unique_labels = np.unique(m)
        if 3 in unique_labels:
            label_format = "brats24"
        elif 4 in unique_labels:
            label_format = "brats17"
        else:
            raise ValueError(
                "Cannot infer BraTS label format from labels; "
                "pass label_format explicitly"
            )

    if label_format in {"brats23", "brats24"}:
        # BraTS23/24 GLI: 1=NETC, 2=SNFH, 3=ET, 4=RC.
        et = (m == 3)
        tc = (m == 1) | (m == 3)
        wt = (m == 1) | (m == 2) | (m == 3)
    else:
        # BraTS17
        et = (m == 4)
        tc = (m == 1) | (m == 4)
        wt = (m == 1) | (m == 2) | (m == 4)

    return np.stack([et, tc, wt], axis=0).astype(np.float32)


def match_modality(p: Path, m: str) -> bool:
    """Return True if file name contains modality m with flexible separators."""
    return re.search(rf"[\-_]{m}[\-_]", p.stem.lower()) is not None


def load_subject_nii_and_pngs(subj_dir: Path, view: str) -> Tuple[Dict[str, List[Path]], Path]:
    """
    For one subject folder, return:
      - dict modality -> sorted list of PNG slice paths for the chosen view
      - seg_path: path to cropped *_seg.nii(.gz)
    """
    view_dir = subj_dir / view
    if not view_dir.exists():
        raise RuntimeError(f"Missing view dir: {view_dir}")

    img_paths_by_mod: Dict[str, List[Path]] = {
        m: sorted([
            p for p in view_dir.glob("*.[Pp][Nn][Gg]")
            if p.stem.startswith(("Bra", "bra")) and     # only match filenames starting with "Bra"
            any(
                re.search(rf"(?<![A-Za-z0-9]){alias}(?![A-Za-z0-9])", p.stem.lower())
                for alias in ALIASES[m]
            )
        ])
        for m in MOD_ORDER
    }
    if not all(img_paths_by_mod[m] for m in MOD_ORDER):
        raise RuntimeError(f"Incomplete modalities in {view_dir}")

    seg_candidates = list(subj_dir.glob("*_seg.nii")) + list(subj_dir.glob("*_seg.nii.gz")) + list(subj_dir.glob("*-seg.nii.gz"))
    if not seg_candidates:
        raise RuntimeError(f"No *_seg.nii in {subj_dir}")
    seg_path = seg_candidates[0]

    counts = [len(v) for v in img_paths_by_mod.values()]
    if len(set(counts)) != 1:
        raise RuntimeError(f"Unequal slice counts across modalities in {view_dir}: {counts}")

    return img_paths_by_mod, seg_path


def load_subject_source_uint8(
    subj_dir: Path,
    view: str,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Load one source subject without expanding uint8 data to float32."""
    img_paths_by_mod, seg_path = load_subject_nii_and_pngs(subj_dir, view)

    seg_img = nib.load(str(seg_path))
    seg = seg_img.get_fdata(dtype=np.float32)
    seg = np.rint(seg).astype(np.int16)
    if seg.shape != TARGET_SHAPE:
        x, y, z = seg.shape
        tx, ty, tz = TARGET_SHAPE
        xs, ys, zs = ((x - tx) // 2, (y - ty) // 2, (z - tz) // 2)
        seg = seg[xs:xs + tx, ys:ys + ty, zs:zs + tz]
    if seg.size and (seg.min() < 0 or seg.max() > np.iinfo(np.uint8).max):
        raise ValueError(f"Segmentation labels do not fit uint8: {seg_path}")

    frames = []
    slice_count = len(next(iter(img_paths_by_mod.values())))
    for slice_index in range(slice_count):
        channels = []
        for modality in MOD_ORDER:
            path = img_paths_by_mod[modality][slice_index]
            with Image.open(path) as image:
                channels.append(np.array(image.convert("L"), dtype=np.uint8))
        frames.append(np.stack(channels, axis=0))

    images = np.stack(frames, axis=0)
    segmentation = seg.astype(np.uint8, copy=False)
    xyz = tuple(int(size) for size in segmentation.shape)
    return images, segmentation, xyz


def subject_cache_path(
    cache_root: Path,
    view: str,
    fold: int,
    subject_id: str,
) -> Path:
    return Path(cache_root) / view / str(int(fold)) / f"{subject_id}.pt"


def build_subject_cache_file(
    subject_dir: Path,
    fold: int,
    view: str,
    cache_root: Path,
    overwrite: bool = False,
) -> Path:
    """Materialize one subject as an atomic, lossless uint8 cache file."""
    subject_dir = Path(subject_dir)
    cache_path = subject_cache_path(cache_root, view, fold, subject_dir.name)
    if cache_path.is_file() and not overwrite:
        return cache_path

    images, segmentation, _xyz = load_subject_source_uint8(subject_dir, view)
    return write_subject_cache_file(cache_path, subject_dir.name, view, images, segmentation)


def write_subject_cache_file(
    cache_path: Path,
    subject_id: str,
    view: str,
    images: np.ndarray,
    segmentation: np.ndarray,
) -> Path:
    """Atomically save uint8 data from PNGs or directly preprocessed volumes."""
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "subject_id": subject_id,
        "view": view,
        "images": torch.from_numpy(np.ascontiguousarray(images)),
        "segmentation": torch.from_numpy(np.ascontiguousarray(segmentation)),
        "xyz": tuple(int(size) for size in segmentation.shape),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return cache_path


def load_subject_cache_file(
    cache_path: Path,
    expected_subject_id: Optional[str] = None,
    expected_view: Optional[str] = None,
) -> Dict:
    """Load and validate one materialized subject cache."""
    cache_path = Path(cache_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid cache payload in {cache_path}: expected mapping")

    required_keys = {
        "format_version", "subject_id", "view", "images",
        "segmentation", "xyz",
    }
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        raise ValueError(
            f"Invalid cache file {cache_path}; missing: {', '.join(missing_keys)}"
        )
    if payload["format_version"] != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported cache format in {cache_path}: "
            f"{payload['format_version']}"
        )
    if expected_subject_id is not None and payload["subject_id"] != expected_subject_id:
        raise ValueError(f"Subject mismatch in cache file: {cache_path}")
    if expected_view is not None and payload["view"] != expected_view:
        raise ValueError(f"View mismatch in cache file: {cache_path}")
    if payload["view"] not in VALID_VIEWS:
        raise ValueError(f"Invalid view in cache file: {cache_path}")
    if not isinstance(payload["images"], torch.Tensor):
        raise ValueError(f"Cached images must be a tensor: {cache_path}")
    if not isinstance(payload["segmentation"], torch.Tensor):
        raise ValueError(f"Cached segmentation must be a tensor: {cache_path}")
    if payload["images"].dtype != torch.uint8:
        raise ValueError(f"Cached images must be uint8: {cache_path}")
    if payload["segmentation"].dtype != torch.uint8:
        raise ValueError(f"Cached segmentation must be uint8: {cache_path}")

    xyz = tuple(int(size) for size in payload["xyz"])
    if tuple(payload["segmentation"].shape) != xyz:
        raise ValueError(
            f"Cached segmentation shape does not match xyz: {cache_path}"
        )
    if xyz != tuple(TARGET_SHAPE):
        raise ValueError(
            f"Cached segmentation shape {xyz} does not match "
            f"TARGET_SHAPE {TARGET_SHAPE}: {cache_path}"
        )
    expected_slices, expected_height, expected_width = expected_DHW_for_view(
        payload["view"]
    )
    expected_images_shape = (
        expected_slices,
        len(MOD_ORDER),
        expected_height,
        expected_width,
    )
    if tuple(payload["images"].shape) != expected_images_shape:
        raise ValueError(
            f"Cached images have shape {tuple(payload['images'].shape)}, "
            f"expected {expected_images_shape}: {cache_path}"
        )
    payload["xyz"] = xyz
    return payload


def expected_DHW_for_view(view: str) -> Tuple[int, int, int]:
    if view == "sagittal":
        return (TARGET_SHAPE[0], TARGET_SHAPE[1], TARGET_SHAPE[2])  # (160,192,152)
    if view == "coronal":
        return (TARGET_SHAPE[1], TARGET_SHAPE[0], TARGET_SHAPE[2])  # (192,160,152)
    return (TARGET_SHAPE[2], TARGET_SHAPE[0], TARGET_SHAPE[1])      # (152,160,192)


def take_view(arr: np.ndarray, view: str) -> np.ndarray:
    """
    Reorder dims to slice along the first spatial axis for the chosen view.
    Input arr: (C, X, Y, Z) or (K, X, Y, Z)
    Returns: (S, C, H, W)
    """
    assert view in VALID_VIEWS
    if view == "sagittal":
        arr = np.moveaxis(arr, 1, -1)   # (C, Y, Z, X)
        arr = arr.transpose(3, 0, 1, 2) # (X, C, Y, Z)
    elif view == "coronal":
        arr = np.moveaxis(arr, 2, -1)   # (C, X, Z, Y)
        arr = arr.transpose(3, 0, 1, 2) # (Y, C, X, Z)
    else:  # axial
        arr = np.moveaxis(arr, 3, -1)   # (C, X, Y, Z)
        arr = arr.transpose(3, 0, 1, 2) # (Z, C, X, Y)
    return arr


def stack_back(slices: np.ndarray, view: str, ref_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Inverse of take_view for predictions.
    slices: (S, K, H, W)  -> returns (K, X, Y, Z)
    """
    if view == "sagittal":
        vol = slices.transpose(1, 2, 3, 0)  # (K, Y, Z, X)
        vol = np.moveaxis(vol, -1, 1)      # (K, X, Y, Z)
    elif view == "coronal":
        vol = slices.transpose(1, 2, 3, 0)  # (K, X, Z, Y)
        vol = np.moveaxis(vol, -1, 2)      # (K, X, Y, Z)
    else:  # axial
        vol = slices.transpose(1, 2, 3, 0)  # (K, X, Y, Z)
    return vol[:, :ref_shape[0], :ref_shape[1], :ref_shape[2]]


class BratsVolumeDataset(Dataset):
    """
    Per-item = one subject:
      x_vol: (S, 4, H, W) float32
      y_vol: (S, 3, H, W) float32
      meta:  dict with 'sid', 'xyz', and raw ground-truth labels
    """
    def __init__(
        self,
        root: str,
        val_fold: int,
        view: str,
        subjects_per_fold: Optional[int] = None,
        cache_root: Optional[str] = None,
        cache_required: bool = False,
        label_format: str = "auto",
    ):
        rootp = ensure_train_root(Path(root))
        self.view = view
        self.fold = int(val_fold)
        self.cache_root = (
            Path(cache_root).expanduser().resolve()
            if cache_root is not None
            else None
        )
        self.cache_required = bool(cache_required)
        if label_format not in {"auto", "brats17", "brats23", "brats24"}:
            raise ValueError(f"Unknown label format: {label_format}")
        self.label_format = label_format
        if view not in VALID_VIEWS:
            raise ValueError(f"view must be one of {VALID_VIEWS}")
        if self.cache_required and self.cache_root is None:
            raise ValueError("cache_required=True requires cache_root")
        if self.cache_required and (
            rootp.resolve() == self.cache_root or not (rootp / str(self.fold)).is_dir()
        ):
            view_root = self.cache_root / view
            folds_manifest = view_root / "folds_manifest.json"
            if folds_manifest.is_file():
                fold_subjects = json.loads(folds_manifest.read_text()).get(str(self.fold), [])
            else:
                manifest = json.loads((view_root / "manifest.json").read_text())
                fold_subjects = [
                    key.split("/", 1)[1] for key in manifest["subjects"]
                    if key.split("/", 1)[0] == str(self.fold)
                ]
            subjects = [rootp / str(self.fold) / sid for sid in sorted(fold_subjects)]
        else:
            subjects = find_subject_dirs(rootp, [val_fold])
        self.subjects = limit_subject_dirs(subjects, subjects_per_fold, f"fold {val_fold}")
        if self.cache_required:
            if self.cache_root is None:
                raise ValueError("cache_required=True requires cache_root")
            missing_cache_paths = []
            for subject_dir in self.subjects:
                cache_path = subject_cache_path(
                    self.cache_root,
                    self.view,
                    self.fold,
                    subject_dir.name,
                )
                if not cache_path.is_file():
                    missing_cache_paths.append(cache_path)
            if missing_cache_paths:
                raise FileNotFoundError(
                    f"Required subject cache is incomplete: "
                    f"{len(missing_cache_paths)} file(s) missing; first: "
                    f"{missing_cache_paths[0]}"
                )

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx: int):
        subj_dir = self.subjects[idx]
        images = None
        seg = None
        xyz = None
        if self.cache_root is not None:
            cache_path = subject_cache_path(
                self.cache_root,
                self.view,
                self.fold,
                subj_dir.name,
            )
            if cache_path.is_file():
                try:
                    payload = load_subject_cache_file(
                        cache_path,
                        expected_subject_id=subj_dir.name,
                        expected_view=self.view,
                    )
                except Exception as exc:
                    if self.cache_required:
                        raise
                    warnings.warn(
                        f"Invalid cache {cache_path}: {exc}. "
                        "Falling back to source files.",
                        RuntimeWarning,
                    )
                else:
                    images = payload["images"].numpy()
                    seg = payload["segmentation"].numpy()
                    xyz = payload["xyz"]
            elif self.cache_required:
                raise FileNotFoundError(
                    f"Required subject cache not found: {cache_path}"
                )

        if images is None:
            images, seg, xyz = load_subject_source_uint8(subj_dir, self.view)

        # segmentation → multilabel → (S,3,H,W)
        ml = brats_to_multilabel(seg, self.label_format)    # (3,X,Y,Z)
        ys = take_view(ml, self.view)    # (S,3,H,W)

        xs = images.astype(np.float32) / 255.0          # (S,4,H,W)

        metadata = {
            "sid": subj_dir.name,
            "xyz": xyz,
            "raw_labels": tuple(int(label) for label in np.unique(seg)),
        }
        return torch.from_numpy(xs).float(), torch.from_numpy(ys).float(), metadata


def collate_training_subjects(batch):
    images, labels, metadata = zip(*batch)
    return torch.stack(images), torch.stack(labels), list(metadata)


class ViewSubset(Dataset):
    """Subset wrapper that preserves the view attribute required by 3D evaluation."""
    def __init__(self, dataset: Dataset, indices: List[int], view: str):
        if not indices:
            raise ValueError("ViewSubset requires at least one index")
        self.dataset = dataset
        self.indices = list(indices)
        self.view = view

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def make_overfit_datasets(train_ds: Dataset, view: str, n_subjects: int) -> Tuple[Dataset, Dataset]:
    """
    Build train/validation datasets from the same first N training subjects.

    This intentionally evaluates on the same examples used for training. It is
    meant only as a fast sanity check that the model, loss, labels, and optimizer
    can memorize a tiny dataset.
    """
    n_subjects = int(n_subjects)
    if n_subjects <= 0:
        raise ValueError("overfit_subjects must be a positive integer")
    if n_subjects > len(train_ds):
        raise ValueError(f"overfit_subjects={n_subjects} exceeds train subjects={len(train_ds)}")

    indices = list(range(n_subjects))
    return ViewSubset(train_ds, indices, view), ViewSubset(train_ds, indices, view)

# -----------------------------
# Firing-rate monitor
# -----------------------------
class FiringRateMonitor:
    """
    Counts spikes (>0) / total elements for each spiking layer.
    Prints global & per-layer rates when report() is called.
    """
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.reset()
        for name, m in model.named_modules():
            if getattr(m, "is_spiking_layer", False) or any(
                k in m.__class__.__name__.lower() for k in ("lif", "plif", "spike")
            ):
                self._register_hook(name, m)

    def _register_hook(self, name, module):
        def hook(_mod, _inp, out):
            if isinstance(out, torch.Tensor):
                spk = (out > 0).float()
                self.layer_spikes[name] = self.layer_spikes.get(name, 0) + spk.sum().item()
                self.layer_total[name]  = self.layer_total.get(name, 0) + spk.numel()
                self.global_spikes += spk.sum().item()
                self.global_total  += spk.numel()
        self.handles.append(module.register_forward_hook(hook))

    def reset(self):
        self.layer_spikes, self.layer_total = {}, {}
        self.global_spikes, self.global_total = 0.0, 0.0

    def report(self, tag=""):
        g_rate = self.global_spikes / self.global_total if self.global_total > 0 else 0.0
        print(f"[Spikes][{tag}] Global firing rate: {g_rate:.6f}")
        rates = {n: self.layer_spikes[n]/self.layer_total[n] for n in self.layer_spikes}
        for n, r in sorted(rates.items(), key=lambda kv: kv[1], reverse=True)[:8]:
            print(f"  [Top] {n:50s} rate={r:.6f}")
        self.reset()


# -----------------------------
# Losses & Metrics (multilabel)
# -----------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,K,*,H,W) arbitrary extra dims (e.g., k time steps)
        targets: same shape
        """
        probs = torch.sigmoid(logits)
        # sum over batch + spatial + time dims; keep channel dim
        reduce_dims = tuple(i for i in range(probs.ndim) if i != 1)
        intersect = (probs * targets).sum(dim=reduce_dims)
        denom = probs.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
        dice = (2 * intersect + self.eps) / (denom + self.eps)  # (K,)
        return 1 - dice.mean()


def size_aware_positive_weights(targets, reference_sizes, gamma=0.5,
                                max_weight=5.0, eps=1.0):
    """Return one positive-voxel weight per sample and class."""
    n_pos = targets.sum(dim=tuple(range(2, targets.ndim)))
    references = torch.as_tensor(
        reference_sizes, device=targets.device, dtype=targets.dtype
    ).reshape(1, -1)
    weights = ((references + eps) / (n_pos + eps)).pow(gamma)
    weights = weights.clamp(min=1.0, max=max_weight)
    return torch.where(n_pos > 0, weights, torch.ones_like(weights))


def size_aware_weighted_bce(logits, targets, reference_sizes, gamma=0.5,
                            max_weight=5.0, eps=1.0):
    class_weights = size_aware_positive_weights(
        targets, reference_sizes, gamma, max_weight, eps
    )
    raw_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive_weights = class_weights.reshape(
        *class_weights.shape, *((1,) * (targets.ndim - 2))
    )
    voxel_weights = torch.where(
        targets > 0, positive_weights, torch.ones_like(targets)
    )
    return (raw_bce * voxel_weights).mean(), class_weights


def foreground_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    reduce_dims = tuple(range(2, probs.ndim))
    intersection = (probs * targets).sum(dim=reduce_dims)
    pred_sum = probs.sum(dim=reduce_dims)
    target_sum = targets.sum(dim=reduce_dims)
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    present = target_sum > 0
    if not present.any():
        return logits.sum() * 0.0
    return 1.0 - dice[present].mean()


def compute_supervised_loss(logits, targets, loss_function, lambda_bce,
                            lambda_dice, reference_sizes=None,
                            size_aware_bce=None, foreground_dice=None):
    if loss_function == "bce_dice":
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = SoftDiceLoss()(logits, targets)
        return lambda_bce * bce + lambda_dice * dice, bce, dice, None
    if loss_function != "wbce_fdice":
        raise ValueError(f"Unknown loss_function: {loss_function}")
    if reference_sizes is None:
        raise ValueError("wbce_fdice requires reference lesion sizes")
    bce, weights = size_aware_weighted_bce(
        logits, targets, reference_sizes, **(size_aware_bce or {})
    )
    dice = foreground_dice_loss(logits, targets, **(foreground_dice or {}))
    return lambda_bce * bce + lambda_dice * dice, bce, dice, weights


@torch.no_grad()
def compute_reference_lesion_sizes(dataset, k):
    """Median positive ET/TC/WT voxel counts in training TBPTT windows."""
    positive_counts = [[] for _ in OUT_TOKENS]
    for index in range(len(dataset)):
        _, targets, _ = dataset[index]
        for t0 in range(0, targets.shape[0], k):
            counts = targets[t0:t0 + k].sum(dim=(0, 2, 3))
            for class_index, count in enumerate(counts.tolist()):
                if count > 0:
                    positive_counts[class_index].append(float(count))
    missing = [
        OUT_TOKENS[i].upper()
        for i, values in enumerate(positive_counts) if not values
    ]
    if missing:
        raise ValueError(
            "Cannot compute reference lesion size; no positive training windows for: "
            + ", ".join(missing)
        )
    return torch.tensor(
        [float(np.median(values)) for values in positive_counts],
        dtype=torch.float32,
    )



def combined_loss_window(logits: torch.Tensor, targets: torch.Tensor,
                         lambda_bce: float, lambda_dice: float) -> torch.Tensor:
    # logits/targets shape: (B,K,k,H,W) for a TBPTT window
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = SoftDiceLoss()(logits, targets)
    return lambda_bce * bce + lambda_dice * dice


@torch.no_grad()
def dice_per_channel(vol_pred_bin: np.ndarray, vol_gt_bin: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    vol_pred_bin, vol_gt_bin: (K, X, Y, Z) in {0,1}
    returns (K,) dice
    """
    K = vol_pred_bin.shape[0]
    d = []
    for k in range(K):
        p = vol_pred_bin[k].reshape(-1).astype(np.uint8)
        t = vol_gt_bin[k].reshape(-1).astype(np.uint8)
        inter = (p & t).sum()
        denom = p.sum() + t.sum()
        d.append(float((2 * inter + eps) / (denom + eps)))
    return np.array(d)


def load_experiment_from_yaml(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("YAML config must be a mapping at the top level")

    required_keys = {
        "name",
        "data_root",
        "val_fold",
        "view",
        "model",
        "patch_size",
        "linear_projection",
        "residual_connections",
        "dwconv2d_spiking",
        "patch_embedding_spiking",
        "batch_size_subjects",
        "lr",
        "epochs",
        "weight_decay",
        "grad_clip",
        "tbptt_k",
        "alpha_fptt",
        "beta_fptt",
        "rho_fptt",
        "lambda_fptt",
        "lambda_bce",
        "lambda_dice",
        "eval_every",
        "eval_batch_slices",
        "prob_threshold",
    }

    missing = sorted(required_keys - set(raw))
    if missing:
        raise ValueError(f"YAML config is missing keys: {', '.join(missing)}")
    use_fptt = raw.get("use_fptt", True)
    if not isinstance(use_fptt, bool):
        raise ValueError("use_fptt must be a boolean")
    raw["use_fptt"] = use_fptt
    eval_batch_subjects = raw.get("eval_batch_subjects", 1)
    if (
        isinstance(eval_batch_subjects, bool)
        or not isinstance(eval_batch_subjects, int)
        or eval_batch_subjects <= 0
    ):
        raise ValueError("eval_batch_subjects must be a positive integer")
    raw["eval_batch_subjects"] = eval_batch_subjects
    if use_fptt and float(raw["alpha_fptt"]) <= 0:
        raise ValueError("alpha_fptt must be positive when use_fptt=true")
    if isinstance(raw["tbptt_k"], bool) or not isinstance(raw["tbptt_k"], int) or raw["tbptt_k"] <= 0:
        raise ValueError("tbptt_k must be a positive integer")
    patch_size = raw["patch_size"]
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    if not isinstance(raw["linear_projection"], bool):
        raise ValueError("linear_projection must be a boolean")
    if not isinstance(raw["residual_connections"], bool):
        raise ValueError("residual_connections must be a boolean")
    if not isinstance(raw["dwconv2d_spiking"], bool):
        raise ValueError("dwconv2d_spiking must be a boolean")
    if not isinstance(raw["patch_embedding_spiking"], bool):
        raise ValueError("patch_embedding_spiking must be a boolean")

    input_skip = raw.get("input_skip", False)
    if not isinstance(input_skip, bool):
        raise ValueError("input_skip must be a boolean")
    if input_skip and raw["model"] != "orig":
        raise ValueError("input_skip=true is supported only for model: orig")
    raw["input_skip"] = input_skip

    resume_from = raw.get("resume_from")
    if resume_from is not None:
        if not isinstance(resume_from, str) or not resume_from.strip():
            raise ValueError("resume_from must be null or a non-empty run directory")
        raw["resume_from"] = resume_from.strip()
    else:
        raw["resume_from"] = None

    resume_scheduler = raw.get("resume_scheduler", True)
    if not isinstance(resume_scheduler, bool):
        raise ValueError("resume_scheduler must be a boolean")
    raw["resume_scheduler"] = resume_scheduler
    resume_lr = raw.get("resume_lr")
    if resume_lr is not None:
        if (
            isinstance(resume_lr, bool)
            or not isinstance(resume_lr, (int, float))
            or not np.isfinite(resume_lr)
            or resume_lr <= 0
        ):
            raise ValueError("resume_lr must be null or a finite positive number")
        if raw["resume_from"] is None or resume_scheduler:
            raise ValueError("resume_lr requires resume_from and resume_scheduler=false")
        resume_lr = float(resume_lr)
    raw["resume_lr"] = resume_lr
    scheduler_name = raw.get("scheduler", "cosine")
    if scheduler_name not in ("cosine", "reduce_plateau"):
        raise ValueError("scheduler must be 'cosine' or 'reduce_plateau'")
    raw["scheduler"] = scheduler_name

    loss_function = raw.get("loss_function", "bce_dice")
    if loss_function not in {"bce_dice", "wbce_fdice"}:
        raise ValueError("loss_function must be one of: 'bce_dice', 'wbce_fdice'")
    raw["loss_function"] = loss_function
    size_aware_bce = raw.get(
        "size_aware_bce", {"gamma": 0.5, "max_weight": 5.0, "eps": 1.0}
    )
    foreground_dice = raw.get("foreground_dice", {"eps": 1e-6})
    if not isinstance(size_aware_bce, dict):
        raise ValueError("size_aware_bce must be a mapping")
    if not isinstance(foreground_dice, dict):
        raise ValueError("foreground_dice must be a mapping")
    bce_defaults = {"gamma": 0.5, "max_weight": 5.0, "eps": 1.0}
    dice_defaults = {"eps": 1e-6}
    if set(size_aware_bce) - set(bce_defaults) or set(foreground_dice) - set(dice_defaults):
        raise ValueError("Unknown size_aware_bce or foreground_dice option")
    bce_defaults.update(size_aware_bce)
    dice_defaults.update(foreground_dice)
    for key, value in bce_defaults.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"size_aware_bce.{key} must be non-negative")
    if bce_defaults["max_weight"] < 1:
        raise ValueError("size_aware_bce.max_weight must be at least 1")
    dice_eps = dice_defaults["eps"]
    if isinstance(dice_eps, bool) or not isinstance(dice_eps, (int, float)) or dice_eps <= 0:
        raise ValueError("foreground_dice.eps must be positive")
    raw["size_aware_bce"] = bce_defaults
    raw["foreground_dice"] = dice_defaults

    cache_root = raw.get("cache_root")
    if cache_root is not None:
        if not isinstance(cache_root, str) or not cache_root.strip():
            raise ValueError("cache_root must be null or a non-empty directory")
        raw["cache_root"] = cache_root.strip()
    else:
        raw["cache_root"] = None

    cache_required = raw.get("cache_required", False)
    if not isinstance(cache_required, bool):
        raise ValueError("cache_required must be a boolean")
    if cache_required and raw["cache_root"] is None:
        raise ValueError("cache_required=true requires cache_root")
    raw["cache_required"] = cache_required

    data_root = raw["data_root"]
    if data_root is None:
        if not cache_required:
            raise ValueError("data_root may be null only when cache_required=true")
        raw["data_root"] = raw["cache_root"]
    elif not isinstance(data_root, str) or not data_root.strip():
        raise ValueError("data_root must be null or a non-empty directory")
    else:
        raw["data_root"] = data_root.strip()

    label_format = raw.get("label_format", "auto")
    if label_format not in {"auto", "brats17", "brats23", "brats24"}:
        raise ValueError("label_format must be one of auto, brats17, brats23, brats24")
    raw["label_format"] = label_format

    for key, default in (
        ("loader_workers", 2),
        ("loader_prefetch_factor", 1),
    ):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        raw[key] = value

    return raw

def build_model(model_name, out_channels=3, patch_size=4,
                linear_projection=True,
                residual_connections=True,
                dwconv2d_spiking=True,
                patch_embedding_spiking=False,
                input_skip=False):
    if input_skip and model_name != "orig":
        raise ValueError("input_skip=true is supported only for model: orig")
    if model_name == "orig":
        return SNNBraTS(
            out_channels=out_channels,
            patch_size=patch_size,
            linear_projection=linear_projection,
            residual_connections=residual_connections,
            dwconv2d_spiking=dwconv2d_spiking,
            patch_embedding_spiking=patch_embedding_spiking,
            input_skip=input_skip,
        )
    if model_name == "shallow":
        return SNNBraTSUNetShallow(out_channels=out_channels)
    if model_name == "medium":
        return SNNBraTSUNetMedium(out_channels=out_channels)
    if model_name == "deep":
        return SNNBraTSUNetDeep(out_channels=out_channels)
    raise ValueError(f"Unknown model: {model_name}")


def run_experiment(exp_cfg: Dict, config_path: Optional[str] = None):
    accelerator = Accelerator()
    use_fptt = exp_cfg.get("use_fptt", True)
    input_skip = exp_cfg.get("input_skip", False)
    exp_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", exp_cfg["name"])
    resume_from = exp_cfg.get("resume_from")
    resume_scheduler = exp_cfg.get("resume_scheduler", True)
    resume_lr = exp_cfg.get("resume_lr")
    scheduler_name = exp_cfg.get("scheduler", "cosine")
    loss_function = exp_cfg.get("loss_function", "bce_dice")
    size_aware_bce_cfg = dict(exp_cfg.get("size_aware_bce", {}))
    foreground_dice_cfg = dict(exp_cfg.get("foreground_dice", {}))
    resume_checkpoint_config = {}
    run_metadata = [
        datetime.now().strftime("%Y%m%d_%H%M%S")
        if accelerator.is_main_process else None,
        os.environ.get("SNN_FPTT_RUN_ID")
        if accelerator.is_main_process else None,
    ]
    broadcast_object_list(run_metadata, from_process=0)
    start_time, configured_run_id = run_metadata

    if resume_from is not None:
        run_dir = Path(resume_from).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
        # Reject mode changes before opening logs or overwriting run metadata.
        resume_checkpoint = torch.load(
            run_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False
        )
        validate_checkpoint_training_mode(resume_checkpoint, use_fptt)
        validate_checkpoint_input_skip(resume_checkpoint, input_skip)
        if resume_scheduler:
            validate_checkpoint_scheduler(resume_checkpoint, scheduler_name)
        resume_checkpoint_config = dict(resume_checkpoint.get("config", {}))
        saved_loss = resume_checkpoint_config.get("loss_function", "bce_dice")
        if saved_loss != loss_function:
            raise ValueError(
                f"Checkpoint loss_function={saved_loss!r} differs from requested "
                f"loss_function={loss_function!r}; start a new run."
            )
        del resume_checkpoint
        run_id = run_dir.name
    else:
        run_id = configured_run_id or f"{exp_name}_{start_time}"
        run_dir = Path(os.environ.get("SNN_FPTT_RUN_DIR", RUNS_ROOT / run_id))
        if accelerator.is_main_process:
            run_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()
    metrics_path = run_dir / "epoch_metrics.csv"
    hyperparams_path = run_dir / "hyperparameters.yaml"
    log_path = run_dir / "train.out"
    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if accelerator.is_main_process:
        log_file = open(log_path, "a", encoding="utf-8")
        sys.stdout = TeeWriter(original_stdout, log_file)
        sys.stderr = TeeWriter(original_stderr, log_file)

    print = accelerator.print

    print(f"\n=== Experiment: {exp_name} ===")
    print(f"Start time: {start_time}")
    tmux_session = get_tmux_session_name()
    print(f"Tmux session: {tmux_session or 'non disponibile'}")
    print(f"Run directory: {run_dir.resolve()}")
    print(json.dumps(exp_cfg, indent=2, default=str))

    data_root = exp_cfg["data_root"]
    val_fold = int(exp_cfg["val_fold"])
    view = exp_cfg["view"]
    model_name = exp_cfg["model"]
    patch_size = exp_cfg["patch_size"]
    linear_projection = exp_cfg["linear_projection"]
    residual_connections = exp_cfg["residual_connections"]
    dwconv2d_spiking = exp_cfg["dwconv2d_spiking"]
    patch_embedding_spiking = exp_cfg["patch_embedding_spiking"]
    epochs = int(exp_cfg["epochs"])
    batch_size_subjects = int(exp_cfg["batch_size_subjects"])
    lr = float(exp_cfg["lr"])
    weight_decay = float(exp_cfg["weight_decay"])
    grad_clip = float(exp_cfg["grad_clip"])
    k = int(exp_cfg["tbptt_k"])
    alpha_fptt = float(exp_cfg["alpha_fptt"])
    beta_fptt = float(exp_cfg["beta_fptt"])
    rho_fptt = float(exp_cfg["rho_fptt"])
    lambda_fptt = float(exp_cfg["lambda_fptt"])
    lambda_bce = float(exp_cfg["lambda_bce"])
    lambda_dice = float(exp_cfg["lambda_dice"])
    eval_every = int(exp_cfg["eval_every"])
    eval_batch_slices = int(exp_cfg["eval_batch_slices"])
    eval_batch_subjects = exp_cfg["eval_batch_subjects"]
    prob_threshold = float(exp_cfg["prob_threshold"])
    cache_root = exp_cfg.get("cache_root")
    cache_required = bool(exp_cfg.get("cache_required", False))
    loader_workers = int(exp_cfg.get("loader_workers", 2))
    loader_prefetch_factor = int(exp_cfg.get("loader_prefetch_factor", 1))
    subjects_per_fold = exp_cfg.get("subjects_per_fold")
    if subjects_per_fold is not None:
        subjects_per_fold = int(subjects_per_fold)
        if subjects_per_fold <= 0:
            raise ValueError("subjects_per_fold must be a positive integer when set")
    overfit_subjects = exp_cfg.get("overfit_subjects")
    if overfit_subjects is not None:
        overfit_subjects = int(overfit_subjects)
        if overfit_subjects <= 0:
            raise ValueError("overfit_subjects must be a positive integer when set")

    config = {
        "name": exp_name,
        "run_id": run_id,
        "tmux_session": tmux_session,
        "resume_from": str(run_dir) if resume_from is not None else None,
        "resume_scheduler": resume_scheduler,
        "resume_lr": resume_lr,
        "scheduler": scheduler_name,
        "data_root": data_root,
        "val_fold": val_fold,
        "view": view,
        "model": model_name,
        "patch_size": patch_size,
        "linear_projection": linear_projection,
        "residual_connections": residual_connections,
        "dwconv2d_spiking": dwconv2d_spiking,
        "patch_embedding_spiking": patch_embedding_spiking,
        "input_skip": input_skip,
        "epochs": epochs,
        "batch_size_subjects": batch_size_subjects,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "tbptt_k": k,
        "use_fptt": use_fptt,
        "alpha_fptt": alpha_fptt,
        "beta_fptt": beta_fptt,
        "rho_fptt": rho_fptt,
        "lambda_fptt": lambda_fptt,
        "lambda_bce": lambda_bce,
        "lambda_dice": lambda_dice,
        "loss_function": loss_function,
        "size_aware_bce": size_aware_bce_cfg,
        "foreground_dice": foreground_dice_cfg,
        "eval_every": eval_every,
        "eval_batch_slices": eval_batch_slices,
        "eval_batch_subjects": eval_batch_subjects,
        "prob_threshold": prob_threshold,
        "cache_root": cache_root,
        "cache_required": cache_required,
        "loader_workers": loader_workers,
        "loader_prefetch_factor": loader_prefetch_factor,
        "subjects_per_fold": subjects_per_fold,
        "overfit_subjects": overfit_subjects,
    }

    print("\n=== CONFIG (SNN) ===")
    print(json.dumps(config, indent=2))

    source_files = []
    job_file = Path(__file__).with_suffix(".job")
    if job_file.exists():
        source_files.append(job_file)
    if config_path is not None:
        config_file = Path(config_path)
        if config_file.exists():
            source_files.append(config_file)

    copied_sources = []
    if accelerator.is_main_process:
        for source_file in source_files:
            if source_file.exists():
                destination = run_dir / source_file.name
                if source_file.resolve() != destination.resolve():
                    shutil.copy2(source_file, destination)
                    copied_sources.append(destination.name)
    if copied_sources:
        print(f"Copied source files: {', '.join(copied_sources)}")

    print(f"Using device: {accelerator.device} | processes: {accelerator.num_processes}")

    all_folds = {1, 2, 3, 4, 5}
    if val_fold not in all_folds:
        raise ValueError("val_fold must be in {1,2,3,4,5}")
    train_folds = sorted(list(all_folds - {val_fold}))
    print(f"Train folds: {train_folds} | Val fold: {val_fold}")

    _train_root = ensure_train_root(Path(data_root))
    for f in train_folds + [val_fold]:
        fdir = _train_root / str(f)
        n_subj = len([p for p in fdir.iterdir() if p.is_dir()]) if fdir.exists() else 0
        tag = "VAL" if f == val_fold else "TRN"
        print(f"[SCAN] Fold {f} ({tag}): {n_subj} subject dirs under {fdir}")

    if subjects_per_fold is not None:
        print(f"[SUBSET] Using first {subjects_per_fold} subject(s) from each train/val fold")

    train_subjects = []
    for f in train_folds:
        dset = BratsVolumeDataset(
            root=data_root,
            val_fold=f,
            view=view,
            subjects_per_fold=subjects_per_fold,
            cache_root=cache_root,
            cache_required=cache_required,
            label_format=exp_cfg["label_format"],
        )
        train_subjects.append(dset)
    if not train_subjects:
        raise RuntimeError("No training subjects found.")
    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset(train_subjects)

    val_ds = BratsVolumeDataset(
        root=data_root,
        val_fold=val_fold,
        view=view,
        subjects_per_fold=subjects_per_fold,
        cache_root=cache_root,
        cache_required=cache_required,
        label_format=exp_cfg["label_format"],
    )
    print(f"Train subjects: {len(train_ds)} | Val subjects: {len(val_ds)}")
    if overfit_subjects is not None:
        train_ds, val_ds = make_overfit_datasets(train_ds, view, overfit_subjects)
        print(
            "\n=== OVERFIT DEBUG MODE ===\n"
            f"Training and evaluating on the same {overfit_subjects} subject(s).\n"
            "Use this only to verify that the model can memorize a tiny dataset.\n"
        )
        print(f"Overfit train subjects: {len(train_ds)} | Overfit val subjects: {len(val_ds)}")

    reference_sizes = None
    if loss_function == "wbce_fdice":
        saved_references = resume_checkpoint_config.get("reference_lesion_sizes")
        reference_payload = [saved_references if accelerator.is_main_process else None]
        if accelerator.is_main_process and saved_references is None:
            computed = compute_reference_lesion_sizes(train_ds, k)
            reference_payload[0] = {
                name.upper(): float(computed[index])
                for index, name in enumerate(OUT_TOKENS)
            }
        broadcast_object_list(reference_payload, from_process=0)
        reference_map = reference_payload[0]
        reference_sizes = torch.tensor(
            [reference_map[name] for name in ("ET", "TC", "WT")],
            dtype=torch.float32,
        )
        config["reference_lesion_sizes"] = reference_map
        print(
            "Loss configuration -> wbce_fdice | "
            f"size_aware_bce={size_aware_bce_cfg} | "
            f"foreground_dice={foreground_dice_cfg}"
        )
        print(
            "Reference positive voxels -> "
            + " ".join(f"M_{name}={reference_map[name]:.1f}" for name in ("ET", "TC", "WT"))
        )
    else:
        print("Loss configuration -> bce_dice (exact baseline reduction)")

    if accelerator.is_main_process:
        with open(hyperparams_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    g = torch.Generator()
    g.manual_seed(SEED)

    loader_options = {
        "num_workers": loader_workers,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": loader_prefetch_factor,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_subjects,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_training_subjects,
        **loader_options,
    )
    train_eval_ds = ViewSubset(train_ds, list(range(len(train_ds))), view)
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=eval_batch_subjects, shuffle=False,
        collate_fn=collate_training_subjects, **loader_options
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_subjects,
        shuffle=False,
        collate_fn=collate_training_subjects,
        **loader_options,
    )

    print(f"Loss weights -> lambda_bce={lambda_bce}, lambda_dice={lambda_dice}")

    model = build_model(
        model_name,
        out_channels=3,
        patch_size=patch_size,
        linear_projection=linear_projection,
        residual_connections=residual_connections,
        dwconv2d_spiking=dwconv2d_spiking,
        patch_embedding_spiking=patch_embedding_spiking,
        input_skip=input_skip,
    )

    if accelerator.is_main_process:
        print_model_info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = create_training_scheduler(optimizer, scheduler_name)
    # Firing-rate monitoring is intentionally disabled because its forward
    # hooks call ``Tensor.item()`` repeatedly and synchronize CPU and GPU.
    # Keep the implementation above available for targeted diagnostics:
    # spkmon = FiringRateMonitor(model)
    spkmon = None
    model, optimizer, train_loader, train_eval_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, train_eval_loader, val_loader, scheduler
    )

    base_model = accelerator.unwrap_model(model)
    init_running_params(base_model)

    best_dice = -1.0
    best_dice_epoch = 0
    start_epoch = 1
    last_checkpoint_path = run_dir / "checkpoint_last.pt"
    best_checkpoint_path = run_dir / "checkpoint_best.pt"

    if resume_from is not None:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {last_checkpoint_path}"
            )
        checkpoint = load_training_checkpoint(
            accelerator,
            model,
            optimizer,
            scheduler,
            last_checkpoint_path,
            g,
            resume_scheduler=resume_scheduler,
            use_fptt=use_fptt,
            resume_lr=resume_lr,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint["best_dice"])
        best_dice_epoch = int(checkpoint["best_dice_epoch"])
        print(
            f"Resumed from {last_checkpoint_path} at epoch "
            f"{checkpoint['epoch']}; next epoch: {start_epoch}"
        )
        scheduler_status = (
            "restored from checkpoint"
            if resume_scheduler
            else (
                "reset with configured resume_lr"
                if resume_lr is not None
                else "reset from checkpoint optimizer LR"
            )
        )
        print(f"  Scheduler state: {scheduler_status}")
        if resume_lr is not None:
            print(f"  Learning rate overridden by resume_lr: {resume_lr:.8g}")
        accelerator.wait_for_everyone()

    append_metrics = (
        resume_from is not None
        and metrics_path.is_file()
        and metrics_path.stat().st_size > 0
    )
    metrics_mode = "a" if append_metrics else "w"
    metrics_target = metrics_path if accelerator.is_main_process else os.devnull
    # Separate CSV preserves the schema of epoch_metrics.csv in existing runs.
    loss_path = run_dir / "loss_components.csv"
    append_losses = resume_from is not None and loss_path.exists() and loss_path.stat().st_size > 0
    loss_target = loss_path if accelerator.is_main_process else os.devnull
    train_dice_path = run_dir / "training_dice.csv"
    append_train_dice = resume_from is not None and train_dice_path.exists() and train_dice_path.stat().st_size > 0
    train_dice_target = train_dice_path if accelerator.is_main_process else os.devnull
    diagnostics_path = run_dir / "loss_diagnostics.csv"
    append_diagnostics = resume_from is not None and diagnostics_path.exists() and diagnostics_path.stat().st_size > 0
    diagnostics_target = diagnostics_path if accelerator.is_main_process else os.devnull
    with open(loss_target, "a" if append_losses else "w", newline="", encoding="utf-8") as loss_file, open(metrics_target, metrics_mode, newline="", encoding="utf-8") as metrics_file, open(train_dice_target, "a" if append_train_dice else "w", newline="", encoding="utf-8") as train_dice_file, open(diagnostics_target, "a" if append_diagnostics else "w", newline="", encoding="utf-8") as diagnostics_file:
        loss_writer = csv.DictWriter(loss_file, fieldnames=["epoch", "use_fptt", "task_loss", "reg_loss", "total_loss"])
        if accelerator.is_main_process and not append_losses:
            loss_writer.writeheader()
        train_dice_writer = csv.DictWriter(train_dice_file, fieldnames=[
            "epoch", "dice_ET", "dice_TC", "dice_WT", "dice_mean", "n_subjects",
        ])
        if accelerator.is_main_process and not append_train_dice:
            train_dice_writer.writeheader()
        diagnostic_fields = [
            "epoch", "loss_function", "bce_loss", "dice_loss",
            "weighted_bce_loss", "foreground_dice_loss", "reg_loss", "total_loss",
        ] + [
            f"{name}_{metric}"
            for name in ("ET", "TC", "WT")
            for metric in ("mean_positive_weight", "max_weight",
                           "fraction_at_max_weight", "positive_windows", "absent_windows")
        ]
        diagnostics_writer = csv.DictWriter(diagnostics_file, fieldnames=diagnostic_fields)
        if accelerator.is_main_process and not append_diagnostics:
            diagnostics_writer.writeheader()
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=[
            "epoch",
            "train_loss",
            "dice_ET",
            "dice_TC",
            "dice_WT",
            "dice_mean",
            "n_subjects",
            "best_dice_mean",
            "best_dice_epoch",
            "checkpoint_path",
        ])
        if accelerator.is_main_process and not append_metrics:
            metrics_writer.writeheader()

        epoch_bar = tqdm(
            range(start_epoch, epochs + 1),
            desc="epochs",
            disable=not accelerator.is_local_main_process,
        )
        for epoch in epoch_bar:
            print(f"\nEpoch {epoch}/{epochs}")
            loss_metrics = {}
            tr_loss = train_epoch_snn_tbptt(model, train_loader, optimizer, accelerator,
                                    k, lambda_bce, lambda_dice,
                                    grad_clip, spkmon,
                                    alpha_fptt, beta_fptt,
                                    rho_fptt, lambda_fptt,
                                    use_fptt=use_fptt, loss_metrics=loss_metrics,
                                    loss_function=loss_function,
                                    reference_sizes=reference_sizes,
                                    size_aware_bce=size_aware_bce_cfg,
                                    foreground_dice=foreground_dice_cfg)
            step_training_scheduler(scheduler, tr_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  train_loss: {tr_loss:.4f} | lr: {current_lr:.8g}")
            bce_label = "weighted_bce" if loss_function == "wbce_fdice" else "bce"
            dice_label = "foreground_dice" if loss_function == "wbce_fdice" else "dice"
            print(
                f"  loss components: {bce_label}={loss_metrics['bce_loss']:.4f}  "
                f"{dice_label}={loss_metrics['dice_loss']:.4f}  "
                f"reg={loss_metrics['reg_loss']:.4f}  total={loss_metrics['total_loss']:.4f}"
            )
            if loss_function == "wbce_fdice":
                for name in ("ET", "TC", "WT"):
                    print(
                        f"  {name} weights: mean={loss_metrics[f'{name}_mean_positive_weight']:.3f} "
                        f"max={loss_metrics[f'{name}_max_weight']:.3f} "
                        f"at_cap={loss_metrics[f'{name}_fraction_at_max_weight']:.1%} "
                        f"positive={loss_metrics[f'{name}_positive_windows']} "
                        f"absent={loss_metrics[f'{name}_absent_windows']}"
                    )

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": float(tr_loss),
                "dice_ET": None,
                "dice_TC": None,
                "dice_WT": None,
                "dice_mean": None,
                "n_subjects": 0,
                "best_dice_mean": float(best_dice),
                "best_dice_epoch": int(best_dice_epoch),
                "checkpoint_path": str(last_checkpoint_path),
            }
            best_improved = False

            train_dice_metrics = None
            if epoch % eval_every == 0:
                # train_dice_metrics = evaluate_3d_snn(
                #     model, train_eval_loader, accelerator,
                #     prob_threshold=prob_threshold, k=eval_batch_slices, spkmon=None,
                # )
                # print(f"  train dice: "
                #     f"ET={train_dice_metrics['dice_ET']:.4f}  "
                #     f"TC={train_dice_metrics['dice_TC']:.4f}  "
                #     f"WT={train_dice_metrics['dice_WT']:.4f}  "
                #     f"mean={train_dice_metrics['dice_mean']:.4f}  "
                #     f"(N={train_dice_metrics['n_subjects']})")

                metrics = evaluate_3d_snn(model,
                                        val_loader,
                                        accelerator,
                                        prob_threshold=prob_threshold,
                                        k=eval_batch_slices,
                                        spkmon=spkmon)

                print(f"  val dice: "
                    f"ET={metrics['dice_ET']:.4f}  "
                    f"TC={metrics['dice_TC']:.4f}  "
                    f"WT={metrics['dice_WT']:.4f}  "
                    f"mean={metrics['dice_mean']:.4f}  "
                    f"(N={metrics['n_subjects']})")

                epoch_metrics.update({
                    "dice_ET": metrics["dice_ET"],
                    "dice_TC": metrics["dice_TC"],
                    "dice_WT": metrics["dice_WT"],
                    "dice_mean": metrics["dice_mean"],
                    "n_subjects": metrics["n_subjects"],
                })

                if metrics["dice_mean"] > best_dice:
                    best_dice = metrics["dice_mean"]
                    best_dice_epoch = epoch
                    best_improved = True

            epoch_metrics["best_dice_mean"] = float(best_dice)
            epoch_metrics["best_dice_epoch"] = int(best_dice_epoch)

            if best_improved and save_training_checkpoint_if_main(
                accelerator,
                model,
                optimizer,
                scheduler,
                best_checkpoint_path,
                epoch,
                best_dice,
                best_dice_epoch,
                config,
                g,
            ):
                print(f"  Saved best checkpoint -> {best_checkpoint_path}")

            if save_training_checkpoint_if_main(
                accelerator,
                model,
                optimizer,
                scheduler,
                last_checkpoint_path,
                epoch,
                best_dice,
                best_dice_epoch,
                config,
                g,
            ):
                print(f"  Saved last checkpoint -> {last_checkpoint_path}")
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                loss_writer.writerow({
                    "epoch": epoch, "use_fptt": use_fptt,
                    "task_loss": loss_metrics["task_loss"],
                    "reg_loss": loss_metrics["reg_loss"],
                    "total_loss": loss_metrics["total_loss"],
                })
                loss_file.flush()
                diagnostic_row = {
                    "epoch": epoch, "loss_function": loss_function,
                    **{key: loss_metrics.get(key) for key in diagnostic_fields[2:]},
                }
                diagnostics_writer.writerow(diagnostic_row)
                diagnostics_file.flush()
                if train_dice_metrics is not None:
                    train_dice_writer.writerow({"epoch": epoch, **train_dice_metrics})
                    train_dice_file.flush()
                metrics_writer.writerow(epoch_metrics)
                metrics_file.flush()

            if accelerator.is_local_main_process:
                progress_metrics = {"train_loss": f"{tr_loss:.4f}"}
                if epoch_metrics["dice_mean"] is not None:
                    progress_metrics["dice_mean"] = f"{epoch_metrics['dice_mean']:.4f}"
                epoch_bar.set_postfix(progress_metrics)

    print(f"\nBest dice: {best_dice}, epoch {best_dice_epoch}")
    accelerator.wait_for_everyone()
    if log_file is not None:
        log_file.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

# -----------------------------
# FPTT
# -----------------------------

# init before training, lambdas is the gradient \Delta l_t(W_{t+1}, avg_weights is \overline{w}_t.
def init_running_params(model):
        model.avg_weights = {}
        model.lambdas = {}
        for name, param in model.named_parameters():
            model.avg_weights[name] = param.detach().clone().type_as(param)
            model.lambdas[name] = 0.0 * param.detach().clone().type_as(param)

# reset after each epoch
def reset_running_params(model):
    for name, param in model.named_parameters():
        param.data.copy_(model.avg_weights[name].data)

# add a loss item
def regularizer_loss(model, reg_loss, alpha, rho=0.0, _lambda=2.0,):
    # print(f"\nalpha: {model.alpha}, beta: {model.beta}, rho: {rho}, _lambda: {_lambda}")
    for name, param in model.named_parameters():
        reg_loss += (rho-1.) * torch.sum(param * model.lambdas[name])
        reg_loss += _lambda * 0.5 * alpha * torch.sum(torch.square(param - model.avg_weights[name]))
    return reg_loss

# update after each parameter udpate
def update_running_params(model, alpha, beta):
    for name, param in model.named_parameters():
        model.lambdas[name].data.add_(-alpha * (param - model.avg_weights[name]))
        model.avg_weights[name].data.mul_((1.0-beta))
        model.avg_weights[name].data.add_(beta*param-(beta/alpha)*model.lambdas[name])


# -----------------------------
# SNN Train / Eval (TBPTT over per-subject sequences)
# -----------------------------
def train_epoch_snn_tbptt(model,
                          loader,
                          optimizer,
                          accelerator,
                          k: int,
                          lambda_bce: float,
                          lambda_dice: float,
                          grad_clip: Optional[float] = 1.0,
                          spkmon: Optional[FiringRateMonitor] = None,
                          alpha=0.5, beta=0.5,  #fptt
                          rho=0.0, lmbda=2.0,
                          use_fptt: bool = True,
                          loss_metrics: Optional[Dict] = None,
                          loss_function: str = "bce_dice",
                          reference_sizes: Optional[torch.Tensor] = None,
                          size_aware_bce: Optional[Dict] = None,
                          foreground_dice: Optional[Dict] = None):
    """
    Train one epoch with TBPTT over per-subject slice sequences.

    Args:
        model: SNN model; forward(x_win[B,k,4,H,W], t0) -> logits[B,3,k,H,W]
        loader: DataLoader yielding (xs[S,4,H,W], ys[S,3,H,W], meta)
        optimizer: torch optimizer
        accelerator: Hugging Face Accelerator controlling device placement,
            backward propagation, clipping, and global reductions
        k: TBPTT window size (slices per window)
        lambda_bce, lambda_dice: loss weights
        grad_clip: max grad-norm (None to disable)
        use_fptt: enable regularization, auxiliary updates and epoch weight reset
        loss_metrics: optional output mapping for task/reg/total epoch losses
        spkmon: optional FiringRateMonitor to track firing rates

    Returns:
        epoch_loss (float)
    """
    model.train()
    base_model = accelerator.unwrap_model(model)
    if spkmon is not None:
        spkmon.reset()

    running_loss = torch.zeros((), device=accelerator.device)
    task_loss_sum = torch.zeros((), device=accelerator.device)
    reg_loss_sum = torch.zeros((), device=accelerator.device)
    bce_loss_sum = torch.zeros((), device=accelerator.device)
    dice_loss_sum = torch.zeros((), device=accelerator.device)
    update_count = torch.zeros((), device=accelerator.device)
    weight_sum = torch.zeros(3, device=accelerator.device)
    weight_max = torch.zeros(3, device=accelerator.device)
    saturated_count = torch.zeros(3, device=accelerator.device)
    positive_count = torch.zeros(3, device=accelerator.device)
    absent_count = torch.zeros(3, device=accelerator.device)
    for xs, ys, meta in loader:
        xs = xs.to(accelerator.device, non_blocking=True)  # (B,S,4,H,W)
        ys = ys.to(accelerator.device, non_blocking=True)  # (B,S,3,H,W)
        B, S, C, H, W = xs.shape

        # Walk along the sequence in windows of length k
        for t0 in range(0, S, k):
            t1 = min(t0 + k, S)
            x_win = xs[:, t0:t1, ...]                 # (B,k,4,H,W)
            y_win = ys[:, t0:t1, ...]                 # (B,k,3,H,W)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_win, t0=t0)              # (B,3,k,H,W)

            # Targets to channel-first to match logits
            y_win_ck = y_win.permute(0, 2, 1, 3, 4).contiguous()  # (B,3,k,H,W)
            # ---- supervised loss ----
            task_loss, bce, dice, class_weights = compute_supervised_loss(
                logits, y_win_ck, loss_function, lambda_bce, lambda_dice,
                reference_sizes=reference_sizes,
                size_aware_bce=size_aware_bce,
                foreground_dice=foreground_dice,
            )
            if class_weights is not None:
                n_pos = y_win_ck.sum(dim=(2, 3, 4))
                present = n_pos > 0
                weight_sum += (class_weights * present).sum(dim=0).detach()
                positive_count += present.sum(dim=0).detach()
                absent_count += (~present).sum(dim=0).detach()
                present_weights = torch.where(
                    present, class_weights, torch.zeros_like(class_weights)
                )
                weight_max = torch.maximum(
                    weight_max, present_weights.max(dim=0).values.detach()
                )
                max_weight = float((size_aware_bce or {}).get("max_weight", 5.0))
                saturated_count += (present & (class_weights >= max_weight)).sum(dim=0).detach()

            # fptt
            reg_loss_value = torch.zeros([]).type_as(logits)
            reg_loss = (
                regularizer_loss(base_model, reg_loss_value, alpha, rho, lmbda)
                if use_fptt else reg_loss_value
            )
            loss = task_loss + reg_loss
            # --------------

            accelerator.backward(loss)
            if grad_clip is not None:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            update_count += 1.0
            
            # fptt
            if use_fptt:
                update_running_params(base_model, alpha, beta)
            if hasattr(base_model, "detach_states"):
                base_model.detach_states()

            

            running_loss += loss.detach() * xs.size(0)
            task_loss_sum += task_loss.detach() * xs.size(0)
            reg_loss_sum += reg_loss.detach() * xs.size(0)
            bce_loss_sum += bce.detach() * xs.size(0)
            dice_loss_sum += dice.detach() * xs.size(0)

    epoch_loss = reduce_loss_totals(accelerator, running_loss, update_count)
    

    # ---- firing-rate report ----
    if spkmon is not None and accelerator.is_local_main_process:
        spkmon.report(tag="Train")

    # ---- fptt ----
    if use_fptt:
        reset_running_params(base_model)

    if loss_metrics is not None:
        loss_metrics.update(
            task_loss=reduce_loss_totals(accelerator, task_loss_sum, update_count),
            bce_loss=reduce_loss_totals(accelerator, bce_loss_sum, update_count),
            dice_loss=reduce_loss_totals(accelerator, dice_loss_sum, update_count),
            reg_loss=reduce_loss_totals(accelerator, reg_loss_sum, update_count),
            total_loss=epoch_loss,
        )
        if loss_function == "wbce_fdice":
            global_weight_sum = accelerator.reduce(weight_sum, reduction="sum")
            global_weight_max = accelerator.reduce(weight_max, reduction="max")
            global_saturated = accelerator.reduce(saturated_count, reduction="sum")
            global_positive = accelerator.reduce(positive_count, reduction="sum")
            global_absent = accelerator.reduce(absent_count, reduction="sum")
            loss_metrics["weighted_bce_loss"] = loss_metrics["bce_loss"]
            loss_metrics["foreground_dice_loss"] = loss_metrics["dice_loss"]
            for index, name in enumerate(("ET", "TC", "WT")):
                count = global_positive[index].clamp_min(1)
                loss_metrics[f"{name}_mean_positive_weight"] = (
                    global_weight_sum[index] / count
                ).item()
                loss_metrics[f"{name}_max_weight"] = global_weight_max[index].item()
                loss_metrics[f"{name}_fraction_at_max_weight"] = (
                    global_saturated[index] / count
                ).item()
                loss_metrics[f"{name}_positive_windows"] = int(global_positive[index].item())
                loss_metrics[f"{name}_absent_windows"] = int(global_absent[index].item())
    return epoch_loss


@torch.no_grad()
def evaluate_3d_snn(model,
                    loader,
                    accelerator,
                    prob_threshold: float = 0.5,
                    k: int = 16,
                    spkmon: Optional[FiringRateMonitor] = None):
    """
    Evaluate batches of subjects with sequential slices and device-side Dice.
    Accumulate voxel counts over all windows separately for each subject, then
    average subject Dice scores (not window Dice or pooled batch counts).

    Args:
        model: SNN model; forward(x_win[B,k,4,H,W], t0) -> logits[B,3,k,H,W]
        loader: DataLoader yielding xs[B,S,4,H,W], ys[B,S,3,H,W], metadata.
            Volumes have matching fixed spatial shapes, without batch padding.
        accelerator: Hugging Face Accelerator controlling device placement
            and distributed metric gathering
        prob_threshold: binarization threshold for predictions
        k: TBPTT window size
        spkmon: optional FiringRateMonitor to track firing rates

    Returns:
        dict with dice_ET, dice_TC, dice_WT, dice_mean, n_subjects
    """
    model.eval()
    if spkmon is not None:
        spkmon.reset()

    dices = []
    for xs, ys, _meta in loader:
        xs = xs.to(accelerator.device)   # (B,S,4,H,W)
        ys = ys.to(accelerator.device)   # (B,S,3,H,W)
        S = xs.shape[1]
        intersection = torch.zeros(
            (xs.shape[0], ys.shape[2]), dtype=torch.int64, device=xs.device
        )
        denominator = torch.zeros_like(intersection)
        # t0=0 resets neuron states for every batch, including a shorter tail.
        # Counting voxels in slice order is equivalent to reconstructing 3D.
        for t0 in range(0, S, k):
            t1 = min(t0 + k, S)
            logits = model(xs[:, t0:t1], t0=t0)     # (B,3,k,H,W)
            predicted = torch.sigmoid(logits) >= prob_threshold
            target = ys[:, t0:t1].permute(0, 2, 1, 3, 4).bool()
            axes = (2, 3, 4)
            intersection += (predicted & target).sum(dim=axes)
            denominator += predicted.sum(dim=axes) + target.sum(dim=axes)

        batch_dices = (2 * intersection.to(torch.float64) + 1e-6) / (
            denominator.to(torch.float64) + 1e-6
        )
        # Keep Accelerate's removal of duplicated subjects in the final batch.
        dices.append(accelerator.gather_for_metrics(batch_dices))

    # Only the small per-subject Dice table leaves the device.
    dices = torch.cat(dices).cpu().numpy() if dices else np.zeros((0, 3), dtype=np.float64)
    mean_per_class = dices.mean(axis=0) if len(dices) else np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # ---- firing-rate report ----
    if spkmon is not None and accelerator.is_local_main_process:
        spkmon.report(tag="Eval")

    return {
        "dice_ET": float(mean_per_class[0]),
        "dice_TC": float(mean_per_class[1]),
        "dice_WT": float(mean_per_class[2]),
        "dice_mean": float(mean_per_class.mean()),
        "n_subjects": int(len(dices)),
    }
    


# -----------------------------
# Main / Config
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="3-view SNN training driven by a YAML experiment file.")
    ap.add_argument("--config", type=str, default=DEFAULT_EXPERIMENTS_YAML,
                    help="Path to YAML experiment file")
    args = ap.parse_args()

    exp_cfg = load_experiment_from_yaml(args.config)
    run_experiment(exp_cfg, args.config)

    print("Done.")
