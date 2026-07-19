#!/usr/bin/env python3
# dnn_3ch.py — SNN (TBPTT) 2D-train / 3D-eval for BraTS (ET/TC/WT multilabel)

import os
import re
import random
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import csv
import shutil

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


def save_checkpoint_if_main(accelerator: Accelerator,
                            model: nn.Module,
                            checkpoint_path: Path,
                            epoch: int,
                            dice_mean: float,
                            config: Dict) -> bool:
    if not accelerator.is_main_process:
        return False
    accelerator.save({
        "model": accelerator.unwrap_model(model).state_dict(),
        "epoch": epoch,
        "dice_mean": dice_mean,
        "config": config,
    }, checkpoint_path)
    return True


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


def brats_to_multilabel(mask3d: np.ndarray) -> np.ndarray:
    """
    Convert BraTS integer labels to multilabel [ET, TC, WT].
    Automatically detects BraTS17 ({0,1,2,4}) or BraTS23 ({0,1,2,3,4}) format.
    Returns (3, X, Y, Z) float32 in {0,1}.
    """
    m = mask3d.astype(np.int32)
    unique_labels = np.unique(m)

    # --- detect version ---
    if 3 in unique_labels:
        # BraTS23
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
      meta:  dict with 'sid' and 'xyz'
    """
    def __init__(self, root: str, val_fold: int, view: str, subjects_per_fold: Optional[int] = None):
        rootp = ensure_train_root(Path(root))
        self.view = view
        if view not in VALID_VIEWS:
            raise ValueError(f"view must be one of {VALID_VIEWS}")
        subjects = find_subject_dirs(rootp, [val_fold])
        self.subjects = limit_subject_dirs(subjects, subjects_per_fold, f"fold {val_fold}")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx: int):
        subj_dir = self.subjects[idx]
        img_paths_by_mod, seg_path = load_subject_nii_and_pngs(subj_dir, self.view)

        # load seg → multilabel → (S,3,H,W)
        seg_img = nib.load(str(seg_path))
        seg = seg_img.get_fdata(dtype=np.float32)
        seg = np.rint(seg).astype(np.int16)
        if seg.shape != TARGET_SHAPE:
            x, y, z = seg.shape
            tx, ty, tz = TARGET_SHAPE
            xs, ys, zs = ((x - tx)//2, (y - ty)//2, (z - tz)//2)
            seg = seg[xs:xs+tx, ys:ys+ty, zs:zs+tz]
        ml = brats_to_multilabel(seg)    # (3,X,Y,Z)
        xyz = ml.shape[1:]
        ys = take_view(ml, self.view)    # (S,3,H,W)

        # images
        D = len(next(iter(img_paths_by_mod.values())))
        frames = []
        for i in range(D):
            chans = []
            for m in MOD_ORDER:
                path = img_paths_by_mod[m][i]
                # ensure file is closed after loading
                with Image.open(path) as im:
                    im = im.convert("L")
                    t = np.array(im, dtype=np.float32) / 255.0
                chans.append(t)
            frames.append(np.stack(chans, axis=0))
        xs = np.stack(frames, axis=0)                   # (S,4,H,W)

        return torch.from_numpy(xs).float(), torch.from_numpy(ys).float(), {"sid": subj_dir.name, "xyz": xyz}


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

    return raw


def run_experiment(exp_cfg: Dict, config_path: Optional[str] = None):
    accelerator = Accelerator()
    exp_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", exp_cfg["name"])
    run_metadata = [
        datetime.now().strftime("%Y%m%d_%H%M%S")
        if accelerator.is_main_process else None,
        os.environ.get("SNN_FPTT_RUN_ID")
        if accelerator.is_main_process else None,
    ]
    broadcast_object_list(run_metadata, from_process=0)
    start_time, configured_run_id = run_metadata
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
    print(f"Run directory: {run_dir.resolve()}")
    print(json.dumps(exp_cfg, indent=2, default=str))

    data_root = exp_cfg["data_root"]
    val_fold = int(exp_cfg["val_fold"])
    view = exp_cfg["view"]
    model_name = exp_cfg["model"]
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
    prob_threshold = float(exp_cfg["prob_threshold"])
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
        "data_root": data_root,
        "val_fold": val_fold,
        "view": view,
        "model": model_name,
        "epochs": epochs,
        "batch_size_subjects": batch_size_subjects,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "tbptt_k": k,
        "alpha_fptt": alpha_fptt,
        "beta_fptt": beta_fptt,
        "rho_fptt": rho_fptt,
        "lambda_fptt": lambda_fptt,
        "lambda_bce": lambda_bce,
        "lambda_dice": lambda_dice,
        "eval_every": eval_every,
        "eval_batch_slices": eval_batch_slices,
        "prob_threshold": prob_threshold,
        "subjects_per_fold": subjects_per_fold,
        "overfit_subjects": overfit_subjects,
    }

    print("\n=== CONFIG (SNN) ===")
    print(json.dumps(config, indent=2))

    if accelerator.is_main_process:
        with open(hyperparams_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

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
        dset = BratsVolumeDataset(root=data_root, val_fold=f, view=view, subjects_per_fold=subjects_per_fold)
        train_subjects.append(dset)
    if not train_subjects:
        raise RuntimeError("No training subjects found.")
    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset(train_subjects)

    val_ds = BratsVolumeDataset(root=data_root, val_fold=val_fold, view=view, subjects_per_fold=subjects_per_fold)
    print(f"Train subjects: {len(train_ds)} | Val subjects: {len(val_ds)}")
    if overfit_subjects is not None:
        train_ds, val_ds = make_overfit_datasets(train_ds, view, overfit_subjects)
        print(
            "\n=== OVERFIT DEBUG MODE ===\n"
            f"Training and evaluating on the same {overfit_subjects} subject(s).\n"
            "Use this only to verify that the model can memorize a tiny dataset.\n"
        )
        print(f"Overfit train subjects: {len(train_ds)} | Overfit val subjects: {len(val_ds)}")

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size_subjects, shuffle=True,
                              num_workers=2, pin_memory=False, drop_last=False,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Loss weights -> lambda_bce={lambda_bce}, lambda_dice={lambda_dice}")

    if model_name == "orig":
        model = SNNBraTS(out_channels=3)
    elif model_name == "shallow":
        model = SNNBraTSUNetShallow(out_channels=3)
    elif model_name == "medium":
        model = SNNBraTSUNetMedium(out_channels=3)
    elif model_name == "deep":
        model = SNNBraTSUNetDeep(out_channels=3)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if accelerator.is_main_process:
        print_model_info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    spkmon = FiringRateMonitor(model)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_dice = -1.0
    best_dice_epoch = 0

    metrics_target = metrics_path if accelerator.is_main_process else os.devnull
    with open(metrics_target, "w", newline="", encoding="utf-8") as metrics_file:
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
        if accelerator.is_main_process:
            metrics_writer.writeheader()

        init_running_params(accelerator.unwrap_model(model))

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            tr_loss = train_epoch_snn_tbptt(model, train_loader, optimizer, accelerator,
                                    k, lambda_bce, lambda_dice,
                                    grad_clip, spkmon,
                                    alpha_fptt, beta_fptt,
                                    rho_fptt, lambda_fptt)
            scheduler.step(tr_loss)
            print(f"  train_loss: {tr_loss:.4f}")

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
                "checkpoint_path": "",
            }

            if epoch % eval_every == 0:
                metrics = evaluate_3d_snn(model,
                                        val_loader,
                                        accelerator,
                                        prob_threshold=prob_threshold,
                                        k=k,
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
                    ckpt_path = run_dir / f"checkpoint_{run_id}.pt"
                    if save_checkpoint_if_main(
                        accelerator, model, ckpt_path, epoch, best_dice, config
                    ):
                        epoch_metrics["checkpoint_path"] = str(ckpt_path)
                        print(f"  Saved best model -> {ckpt_path}")
                    accelerator.wait_for_everyone()

            epoch_metrics["best_dice_mean"] = float(best_dice)
            epoch_metrics["best_dice_epoch"] = int(best_dice_epoch)
            if accelerator.is_main_process:
                metrics_writer.writerow(epoch_metrics)
                metrics_file.flush()

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
                          rho=0.0, lmbda=2.0):  # fptt
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
        spkmon: optional FiringRateMonitor to track firing rates

    Returns:
        epoch_loss (float)
    """
    model.train()
    base_model = accelerator.unwrap_model(model)
    if spkmon is not None:
        spkmon.reset()

    running_loss = torch.zeros((), device=accelerator.device)
    update_count = torch.zeros((), device=accelerator.device)
    pbar = tqdm(loader, desc=f"train (TBPTT k={k})",
                disable=not accelerator.is_local_main_process)
    for xs, ys, meta in pbar:
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
            # ---- loss ----
            bce = F.binary_cross_entropy_with_logits(logits, y_win_ck) if lambda_bce > 0 else torch.tensor(0., device=logits.device)
            # Soft Dice
            probs = torch.sigmoid(logits)
            reduce_dims = tuple(i for i in range(probs.ndim) if i != 1)  # sum over all except channel
            intersect = (probs * y_win_ck).sum(dim=reduce_dims)
            denom = probs.sum(dim=reduce_dims) + y_win_ck.sum(dim=reduce_dims)
            dice = 1 - ((2 * intersect + 1e-6) / (denom + 1e-6)).mean()
            
            # fptt
            reg_loss_value = torch.zeros([]).type_as(logits)
            reg_loss = regularizer_loss(base_model, reg_loss_value, alpha, rho, lmbda)
            loss = lambda_bce * bce + lambda_dice * dice + reg_loss
            # --------------

            accelerator.backward(loss)
            if grad_clip is not None:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            update_count += 1.0
            
            # fptt
            update_running_params(base_model, alpha, beta)
            if hasattr(base_model, "detach_states"):
                base_model.detach_states()

            

            running_loss += loss.detach() * xs.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = reduce_loss_totals(accelerator, running_loss, update_count)
    

    # ---- firing-rate report ----
    if spkmon is not None and accelerator.is_local_main_process:
        spkmon.report(tag="Train")

    # ---- fptt ----
    reset_running_params(base_model)

    return epoch_loss


@torch.no_grad()
def evaluate_3d_snn(model,
                    loader,
                    accelerator,
                    prob_threshold: float = 0.5,
                    k: int = 16,
                    spkmon: Optional[FiringRateMonitor] = None):
    """
    Evaluate per subject:
      - run TBPTT over slices
      - stack per-slice predictions back to 3D volumes
      - compute Dice for ET/TC/WT

    Args:
        model: SNN model; forward(x_win[B,k,4,H,W], t0) -> logits[B,3,k,H,W]
        loader: DataLoader yielding (xs[S,4,H,W], ys[S,3,H,W], meta{'xyz',...})
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
    for xs, ys, meta in tqdm(loader, desc="eval", leave=False,
                             disable=not accelerator.is_local_main_process):
        xs = xs.to(accelerator.device)   # (1,S,4,H,W)
        ys = ys.to(accelerator.device)   # (1,S,3,H,W)
        S = xs.shape[1]
        xyz = meta["xyz"]

        preds_seq = []
        # Iterate windows, collect per-slice probs
        for t0 in range(0, S, k):
            t1 = min(t0 + k, S)
            x_win = xs[:, t0:t1, ...]                # (1,k,4,H,W)
            logits = model(x_win, t0=t0)             # (1,3,k,H,W)
            probs  = torch.sigmoid(logits).cpu().numpy()   # (1,3,k,H,W)
            probs = np.transpose(probs, (0, 2, 1, 3, 4))   # (1,k,3,H,W)
            preds_seq.append(probs[0])               # (k,3,H,W)
        preds = np.concatenate(preds_seq, axis=0)     # (S,3,H,W)

        # Reassemble to 3D
        target_np = ys.cpu().numpy()[0]               # (S,3,H,W)
        vol_pred = stack_back(preds, loader.dataset.view, xyz)   # (3,X,Y,Z)
        vol_gt   = stack_back(target_np, loader.dataset.view, xyz)

        # Binarize predictions; GT already {0,1}
        vol_bin  = (vol_pred >= prob_threshold).astype(np.uint8)
        vol_gtb  = vol_gt.astype(np.uint8)

        # Dice per channel
        K = vol_bin.shape[0]
        d = []
        for ch in range(K):
            p = vol_bin[ch].reshape(-1).astype(np.uint8)
            t = vol_gtb[ch].reshape(-1).astype(np.uint8)
            inter = (p & t).sum()
            denom = p.sum() + t.sum()
            d.append((2 * inter + 1e-6) / (denom + 1e-6))
        batch_dices = torch.tensor([d], dtype=torch.float64,
                                    device=accelerator.device)
        gathered_dices = accelerator.gather_for_metrics(batch_dices)
        dices.extend(gathered_dices.cpu().numpy())

    dices = np.array(dices) if len(dices) else np.zeros((0, 3), dtype=np.float64)
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
