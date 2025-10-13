#!/usr/bin/env python3
# dnn_3ch.py — SNN (TBPTT) 2D-train / 3D-eval for BraTS (ET/TC/WT multilabel)

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

import numpy as np
import nibabel as nib
from tqdm import tqdm

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ==== use your spiking UNet-like model ====
# SNNBraTS: forward(x_win[B,k,4,H,W], t0) -> (B, out_channels, k, H, W)
from model import SNNBraTS  # mirrors your SNN implementation with PLIF nodes

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
OUT_TOKENS = ["et", "tc", "wt"]             # 3 output channels
TARGET_SHAPE = (160, 192, 152)              # (x,y,z) used by preprocessing
FOLD_NAMES = {"1", "2", "3", "4", "5"}


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


def brats_to_multilabel(mask3d: np.ndarray) -> np.ndarray:
    """
    BraTS integer labels {0,1,2,4} -> multilabel [ET,TC,WT]
    Returns (3, X, Y, Z) float32 in {0,1}.
    """
    m = mask3d.astype(np.int32)
    et = (m == 4)
    tc = (m == 1) | (m == 4)
    wt = (m == 1) | (m == 2) | (m == 4)
    return np.stack([et, tc, wt], axis=0).astype(np.float32)


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
        m: sorted(view_dir.glob(f"Brats17_*_{m}_*.png")) for m in MOD_ORDER
    }
    if not all(img_paths_by_mod[m] for m in MOD_ORDER):
        raise RuntimeError(f"Incomplete modalities in {view_dir}")

    seg_candidates = list(subj_dir.glob("*_seg.nii")) + list(subj_dir.glob("*_seg.nii.gz"))
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
    def __init__(self, root: str, val_fold: int, view: str):
        rootp = ensure_train_root(Path(root))
        self.view = view
        if view not in VALID_VIEWS:
            raise ValueError(f"view must be one of {VALID_VIEWS}")
        self.subjects = find_subject_dirs(rootp, [val_fold])

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

# -----------------------------
# FPTT
# -----------------------------

# init before training
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
                          device,
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
        device: torch.device
        k: TBPTT window size (slices per window)
        lambda_bce, lambda_dice: loss weights
        grad_clip: max grad-norm (None to disable)
        spkmon: optional FiringRateMonitor to track firing rates

    Returns:
        epoch_loss (float)
    """
    model.train()
    if spkmon is not None:
        spkmon.reset()

    running_loss = 0.0
    update_count = 0
    pbar = tqdm(loader, desc=f"train (TBPTT k={k})")
    for xs, ys, meta in pbar:
        xs = xs.to(device, non_blocking=True)  # (B,S,4,H,W)
        ys = ys.to(device, non_blocking=True)  # (B,S,3,H,W)
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
            reg_loss = regularizer_loss(model, reg_loss_value, alpha, rho, lmbda)
            loss = lambda_bce * bce + lambda_dice * dice + reg_loss
            # --------------

            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            update_count += 1
            
            # fptt
            update_running_params(model, alpha, beta)
            if hasattr(model, "detach_states"):
                model.detach_states()

            

            running_loss += loss.item() * xs.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / update_count
    

    # ---- firing-rate report ----
    if spkmon is not None:
        spkmon.report(tag="Train")

    # ---- fptt ----
    reset_running_params(model)

    return epoch_loss


@torch.no_grad()
def evaluate_3d_snn(model,
                    loader,
                    device,
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
        device: torch.device
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
    for xs, ys, meta in tqdm(loader, desc="eval", leave=False):
        xs = xs.to(device)   # (1,S,4,H,W)
        ys = ys.to(device)   # (1,S,3,H,W)
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
        dices.append(np.array(d, dtype=np.float64))

    dices = np.array(dices) if len(dices) else np.zeros((0, 3), dtype=np.float64)
    mean_per_class = dices.mean(axis=0) if len(dices) else np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # ---- firing-rate report ----
    if spkmon is not None:
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
    # ---- Config (edit here) ----
    data_root = "data/BRATS2017_preprocessed/Brats17TrainingData"

    val_fold = 1              # int in {1..5}, used as validation
    view = "sagittal"            # 'sagittal' | 'coronal' | 'axial'

    # training
    epochs = 100
    batch_size_subjects = 8   # subjects per batch (each provides a sequence of slices)
    # Adadelta defaults from your prior config (works well with TBPTT)
    # lr = 1.0
    # rho = 0.95
    # eps = 1e-8
    # weight_decay = 1e-5
    # grad_clip = 1.0
    # Adam
    lr = 0.001
    rho = None
    eps = None
    weight_decay = 1e-5
    grad_clip = 0.3
    
    # FPTT
    k = 1  # number of slices per window, not number of updates
    alpha_fptt = 0.1
    beta_fptt = 0.15
    rho_fptt = 0.0
    lambda_fptt = 1.5

    # loss weights: L = λ_bce * BCEWithLogits + λ_dice * SoftDice
    lambda_bce = 0.5
    lambda_dice = 0.5

    # evaluation
    eval_every = 1
    eval_batch_slices = 16
    prob_threshold = 0.5

    config = {
        "data_root": data_root,
        "val_fold": val_fold,
        "view": view,
        "epochs": epochs,
        "batch_size_subjects": batch_size_subjects,
        "lr": lr,
        "rho": rho,
        "eps": eps,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "tbptt_k": k,
        "alpha_fptt": alpha_fptt,
        "beta_fptt": beta_fptt,
        "rho_fptt" : rho_fptt,
        "lambda_fptt": lambda_fptt,
        "lambda_bce": lambda_bce,
        "lambda_dice": lambda_dice,
        "eval_every": eval_every,
        "eval_batch_slices": eval_batch_slices,
        "prob_threshold": prob_threshold
    }

    print("\n=== CONFIG (SNN) ===")
    print(json.dumps(config, indent=2))

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    print(f"Using device: {device}")

    # ---- Fold selection (no train.txt) ----
    all_folds = {1, 2, 3, 4, 5}
    if val_fold not in all_folds:
        raise ValueError("val_fold must be in {1,2,3,4,5}")
    train_folds = sorted(list(all_folds - {val_fold}))
    print(f"Train folds: {train_folds} | Val fold: {val_fold}")

    # ---- Quick preflight: how many subjects per fold? ----
    _train_root = ensure_train_root(Path(data_root))
    for f in train_folds + [val_fold]:
        fdir = _train_root / str(f)
        n_subj = len([p for p in fdir.iterdir() if p.is_dir()]) if fdir.exists() else 0
        tag = "VAL" if f == val_fold else "TRN"
        print(f"[SCAN] Fold {f} ({tag}): {n_subj} subject dirs under {fdir}")

    # ---- Data ----
    # For SNN TBPTT, use per-subject sequences (BratsVolumeDataset) for both train and val.
    train_subjects = []
    for f in train_folds:
        dset = BratsVolumeDataset(root=data_root, val_fold=f, view=view)
        train_subjects.append(dset)
    if not train_subjects:
        raise RuntimeError("No training subjects found.")
    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset(train_subjects)

    val_ds = BratsVolumeDataset(root=data_root, val_fold=val_fold, view=view)
    print(f"Train subjects: {len(train_ds)} | Val subjects: {len(val_ds)}")

    # seeded train loader
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size_subjects, shuffle=True,
                              num_workers=2, pin_memory=False, drop_last=False,
                              worker_init_fn=seed_worker, generator=g)

    # train_loader = DataLoader(train_ds, batch_size=batch_size_subjects, shuffle=True,
    #                           num_workers=2, pin_memory=False, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"Loss weights -> lambda_bce={lambda_bce}, lambda_dice={lambda_dice}")

    # ---- Model/Optim ----
    model = SNNBraTS(out_channels=3).to(device)  # ET/TC/WT
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    spkmon = FiringRateMonitor(model)

    # ---- Train/Eval Loop ----
    best_dice = -1.0
    best_dice_epoch = 0
    
    # ---- fptt ----
    init_running_params(model)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        tr_loss = train_epoch_snn_tbptt(model, train_loader, optimizer, device,
                                k, lambda_bce, lambda_dice,
                                grad_clip, spkmon, 
                                alpha_fptt, beta_fptt,
                                rho_fptt, lambda_fptt)
        scheduler.step(tr_loss)
        print(f"  train_loss: {tr_loss:.4f}")

        # ---- Validation ----
        if epoch % eval_every == 0:
            metrics = evaluate_3d_snn(model,
                                    val_loader,
                                    device,
                                    prob_threshold=prob_threshold,
                                    k=k,
                                    spkmon=spkmon)   # <- pass the monitor here

            print(f"  val dice: "
                f"ET={metrics['dice_ET']:.4f}  "
                f"TC={metrics['dice_TC']:.4f}  "
                f"WT={metrics['dice_WT']:.4f}  "
                f"mean={metrics['dice_mean']:.4f}  "
                f"(N={metrics['n_subjects']})")

            # save best checkpoint
            if metrics["dice_mean"] > best_dice:
                best_dice = metrics["dice_mean"]
                best_dice_epoch = epoch
                ckpt_path = f"checkpoint_snn_fold{val_fold}_{view}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "dice_mean": best_dice,
                    "config": config
                }, ckpt_path)
                print(f"  Saved best model -> {ckpt_path}")

    print(f"\nBest dice: {best_dice}, epoch {best_dice_epoch}")  
    print("Done.")
