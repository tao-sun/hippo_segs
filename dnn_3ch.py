#!/usr/bin/env python3
# dnn.py — Train 2D / Eval 3D for BraTS (no train.txt; folds discovery)
# Architecture mirrors current SNNBraTS but without spiking.
import re
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Constants & helpers
# -----------------------------
VALID_VIEWS = {"sagittal", "coronal", "axial"}
MOD_ORDER = ["t1c", "t1n", "t2f", "t2w"]   # 4 input channels
OUT_TOKENS = ["et", "tc", "wt"]             # 3 output channels
MOD_ALIASES = {
    "t1c": ["t1c", "t1ce", "t1"],
    "t1n": ["t1n", "t1"],
    "t2f": ["t2f", "flair", "t2"],
    "t2w": ["t2w", "t2"],
}

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
    et = (m == 3)
    tc = (m == 1) | (m == 3)
    wt = (m == 1) | (m == 2) | (m == 3)
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
            if p.stem.lower().startswith("bra") and
            any(
                re.search(rf"(?<![A-Za-z0-9]){alias}(?![A-Za-z0-9])", p.stem.lower())
                for alias in MOD_ALIASES.get(m, [m])
            )
        ])
        for m in MOD_ORDER
    }
    if not all(img_paths_by_mod[m] for m in MOD_ORDER):
        raise RuntimeError(f"Incomplete modalities in {view_dir}")

    # robust seg detection: match any .nii / .nii.gz file containing 'seg' in the stem
    # search recursively in subject dir (handles nested subfolders)
    seg_candidates = sorted([p for p in subj_dir.rglob("*seg*.nii*") if p.suffix in (".nii", ".gz")])
    if not seg_candidates:
        raise RuntimeError(f"No *_seg.nii(.gz) in {subj_dir}")
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
    # ensure shape
    return vol[:, :ref_shape[0], :ref_shape[1], :ref_shape[2]]


# -----------------------------
# Datasets
# -----------------------------
class BratsSliceDataset(Dataset):
    """
    Training dataset yielding 2D slices from all training folds (other than val_fold).
    Each sample:
      x: (4, H, W) float32
      y: (3, H, W) float32
    """
    def __init__(self, root: str, train_folds: List[int], view: str, drop_empty: bool = False):
        rootp = ensure_train_root(Path(root))
        self.view = view
        if view not in VALID_VIEWS:
            raise ValueError(f"view must be one of {VALID_VIEWS}")
        subjects = find_subject_dirs(rootp, train_folds, verbose=True)

        kept, skipped = 0, 0
        xs_all, ys_all = [], []

        for subj_dir in tqdm(subjects):
            try:
                view_dir = subj_dir / view
                if not view_dir.exists():
                    print(f"[SKIP] No '{view}' view dir for {subj_dir.name}")
                    skipped += 1
                    continue

                img_paths_by_mod: Dict[str, List[Path]] = {
                    m: sorted([
                        p for p in view_dir.glob("*.[Pp][Nn][Gg]")
                        if p.stem.lower().startswith("bra") and
                        any(
                            re.search(rf"(?<![A-Za-z0-9]){alias}(?![A-Za-z0-9])", p.stem.lower())
                            for alias in MOD_ALIASES.get(m, [m])
                        )
                    ])
                    for m in MOD_ORDER
                }
                if not all(img_paths_by_mod[m] for m in MOD_ORDER):
                    print(f"[SKIP] Missing modality PNGs in {view_dir} (modalities with 0 files: "
                          f"{[m for m in MOD_ORDER if not img_paths_by_mod[m]]})")
                    skipped += 1
                    continue

                counts = [len(v) for v in img_paths_by_mod.values()]
                if len(set(counts)) != 1:
                    print(f"[SKIP] Unequal slice counts across modalities in {view_dir}: {counts}")
                    skipped += 1
                    continue
                D = counts[0]

                # robust seg detection: match any .nii / .nii.gz file containing 'seg' in the stem
                # search recursively in subject dir (handles nested subfolders)
                seg_candidates = sorted([p for p in subj_dir.rglob("*seg*.nii*") if p.suffix in (".nii", ".gz")])
                if not seg_candidates:
                    print(f"[SKIP] No *_seg.nii(.gz) in {subj_dir}")
                    skipped += 1
                    continue
                seg_path = seg_candidates[0]

                seg_img = nib.load(str(seg_path))
                seg = seg_img.get_fdata(dtype=np.float32)
                seg = np.rint(seg).astype(np.int16)
                if seg.shape != TARGET_SHAPE:
                    x, y, z = seg.shape
                    tx, ty, tz = TARGET_SHAPE
                    xs0, ys0, zs0 = ((x - tx)//2, (y - ty)//2, (z - tz)//2)
                    seg = seg[xs0:xs0+tx, ys0:ys0+ty, zs0:zs0+tz]
                ml = brats_to_multilabel(seg)        # (3,X,Y,Z)
                ys = take_view(ml, view)              # (S,3,H,W)

                frames = []
                from PIL import Image
                for i in range(D):
                    chans = []
                    for m in MOD_ORDER:
                        p = img_paths_by_mod[m][i]
                        t = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
                        chans.append(t)
                    frames.append(np.stack(chans, axis=0))  # (4,H,W)
                xs = np.stack(frames, axis=0)               # (S,4,H,W)

                if drop_empty:
                    keep = (ys.reshape(ys.shape[0], -1).sum(axis=1) > 0)
                    if keep.sum() == 0:
                        print(f"[SKIP] All slices empty GT for {subj_dir.name} ({view}), dropping subject")
                        skipped += 1
                        continue
                    xs = xs[keep]
                    ys = ys[keep]

                xs_all.append(xs)
                ys_all.append(ys)
                kept += 1

            except Exception as e:
                print(f"[SKIP] {subj_dir.name}: {type(e).__name__}: {e}")
                skipped += 1
                continue

        if not xs_all:
            raise RuntimeError(f"No training slices found after scanning folds {train_folds} in {rootp}. "
                               f"Kept={kept}, Skipped={skipped} — see logs above for reasons.")

        self.x = torch.from_numpy(np.concatenate(xs_all, axis=0)).float()  # (N,4,H,W)
        self.y = torch.from_numpy(np.concatenate(ys_all, axis=0)).float()  # (N,3,H,W)
        print(f"[OK] Training slices prepared: {self.x.shape[0]} slices from {kept} subjects "
              f"(skipped {skipped}).")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class BratsVolumeDataset(Dataset):
    """
    Validation/Test dataset yielding full subjects (as slice stacks) from a single fold.
    Returns:
      x_vol: (S, 4, H, W) float32
      y_vol: (S, 3, H, W) float32
      meta:  dict with 'sid' and 'xyz'

    Optional RAM guard: set max_ram_gb to emit a warning (default) or raise if the
    estimated in-memory size exceeds the threshold. The estimate assumes float32
    tensors for both inputs (4 channels) and labels (3 channels).
    """
    def __init__(self, root: str, val_fold: int, view: str,
                 max_ram_gb: Optional[float] = None, raise_if_exceeds: bool = False):
        rootp = ensure_train_root(Path(root))
        self.view = view
        if view not in VALID_VIEWS:
            raise ValueError(f"view must be one of {VALID_VIEWS}")
        self.subjects = find_subject_dirs(rootp, [val_fold])

        # Lightweight estimate of how much RAM the whole dataset would occupy if fully materialized.
        self.estimated_total_gb = 0.0
        self.estimated_per_subject_gb = 0.0
        if self.subjects:
            est_total_gb, est_per_subj_gb, slices, hw = self._estimate_ram_gb()
            self.estimated_total_gb = est_total_gb
            self.estimated_per_subject_gb = est_per_subj_gb
            print(
                f"[INFO] BratsVolumeDataset ≈ {est_total_gb:.2f} GB "
                f"({est_per_subj_gb:.2f} GB/subject, {len(self.subjects)} subjects, "
                f"S={slices}, HxW={hw[0]}x{hw[1]})"
            )

            if max_ram_gb is not None and est_total_gb > max_ram_gb:
                msg = (
                    f"Estimated RAM {est_total_gb:.2f} GB exceeds limit {max_ram_gb:.2f} GB "
                    f"for fold {val_fold} ({len(self.subjects)} subjects, view={view})."
                )
                if raise_if_exceeds:
                    raise MemoryError(msg)
                else:
                    print(f"[WARN] {msg}")

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
        ml = brats_to_multilabel(seg).astype(np.float32)    # (3,X,Y,Z)
        xyz = ml.shape[1:]
        ys = take_view(ml, self.view)    # (S,3,H,W)

        # images
        D = len(next(iter(img_paths_by_mod.values())))
        frames = []
        from PIL import Image
        for i in range(D):
            chans = []
            for m in MOD_ORDER:
                t = np.array(Image.open(img_paths_by_mod[m][i]).convert("L"), dtype=np.float32) / np.float32(255.0)
                chans.append(t)
            frames.append(np.stack(chans, axis=0))      # (4,H,W)
        xs = np.stack(frames, axis=0)                   # (S,4,H,W)

        return torch.from_numpy(xs).float(), torch.from_numpy(ys).float(), {"sid": subj_dir.name, "xyz": xyz}

    def _estimate_ram_gb(self):
        """Return (total_gb, per_subject_gb, num_slices, (H, W)) estimate without loading data."""
        # Default to target shape if probing fails.
        slices, H, W = expected_DHW_for_view(self.view)
        try:
            img_paths_by_mod, _ = load_subject_nii_and_pngs(self.subjects[0], self.view)
            slices = len(next(iter(img_paths_by_mod.values()))) or slices
            from PIL import Image
            sample_w, sample_h = Image.open(img_paths_by_mod[MOD_ORDER[0]][0]).size
            H, W = sample_h, sample_w
        except Exception:
            pass

        per_subject_bytes = slices * H * W * (len(MOD_ORDER) + len(OUT_TOKENS)) * 4  # float32
        total_bytes = per_subject_bytes * len(self.subjects)
        to_gb = lambda b: b / (1024 ** 3)
        return to_gb(total_bytes), to_gb(per_subject_bytes), slices, (H, W)


# -----------------------------
# Model (same layout as current SNNBraTS, but DNN)
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
        )
        self.dropout = dropout

    def forward(self, x):
        x = self.net(x)
        if self.dropout and self.training:
            x = F.dropout(x, self.dropout)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout

    def forward(self, x):
        x = self.up(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout and self.training:
            x = F.dropout(x, self.dropout)
        return x


class UNetLike2D(nn.Module):
    """
    Mirrors your SNNBraTS:
      Enc: 4→32 → 32→64 → 64→128 with MaxPool after each block (3 downs total)
      Dec: 128→128 (up), concat with enc feature, 2 convs ...
      Final 1×1 conv to 3 channels (ET/TC/WT)
    """
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.conv_block1 = ConvBlock(in_channels, 32)
        self.conv_block2 = ConvBlock(32, 64)
        self.conv_block3 = ConvBlock(64, 128)

        # Decoder stage 1 (mirrors: deconv + conv + concat + conv)
        self.deconv_block1 = DeconvBlock(128, 128)
        self.deconv1_conv = ConvBlock(128, 128)
        self.concat1_conv = ConvBlock(128 + 64, 128)  # concat with pool2 (64 ch)

        # Decoder stage 2
        self.deconv_block2 = DeconvBlock(128, 128)
        self.deconv2_conv = ConvBlock(128, 128)
        self.concat2_conv = ConvBlock(128 + 32, 128)  # concat with pool1 (32 ch)

        # Decoder stage 3
        self.deconv_block3 = DeconvBlock(128, 128)
        self.deconv3_conv = ConvBlock(128, 128)

        # Classifier
        self.outc = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B,4,H,W) → logits: (B,3,H,W)
        x1 = self.conv_block1(x)       # 32
        p1 = self.pool(x1)             # ↓

        x2 = self.conv_block2(p1)      # 64
        p2 = self.pool(x2)             # ↓

        x3 = self.conv_block3(p2)      # 128
        p3 = self.pool(x3)             # ↓

        y = self.deconv_block1(p3)     # 128
        y = self.deconv1_conv(y)       # 128
        y = torch.cat([p2, y], dim=1)  # 64+128
        y = self.concat1_conv(y)       # 128

        y = self.deconv_block2(y)      # 128
        y = self.deconv2_conv(y)       # 128
        y = torch.cat([p1, y], dim=1)  # 32+128
        y = self.concat2_conv(y)       # 128

        y = self.deconv_block3(y)      # 128
        y = self.deconv3_conv(y)       # 128

        return self.outc(y)            # (B,3,H,W)


# -----------------------------
# Losses & Metrics
# -----------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersect = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2 * intersect + self.eps) / (denom + self.eps)
        return 1 - dice.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor,
                  lambda_bce: float, lambda_dice: float) -> torch.Tensor:
    # BCE over all elements (includes the 3 channels) + Dice averaged over channels
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = SoftDiceLoss()(logits, targets)
    return lambda_bce * bce + lambda_dice * dice


@torch.no_grad()
def dice_per_channel(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    """
    pred, target: (K, X, Y, Z) in {0,1}
    returns (K,) dice
    """
    K = pred.shape[0]
    d = []
    for k in range(K):
        p = pred[k].reshape(-1).astype(np.uint8)
        t = target[k].reshape(-1).astype(np.uint8)
        inter = (p & t).sum()
        denom = p.sum() + t.sum()
        d.append(float((2 * inter + eps) / (denom + eps)))
    return np.array(d)


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, opt, device, lambda_bce, lambda_dice, grad_clip: Optional[float] = None):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"train")
    for x, y in pbar:
        x = x.to(device)  # (B,4,H,W)
        y = y.to(device)  # (B,3,H,W)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = combined_loss(logits, y, lambda_bce, lambda_dice)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        running += loss.item() * x.size(0)
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate_3d(model, loader, device, batch_size_eval: int = 8, thresh: float = 0.5):
    """
    Per subject: run per-slice inference, stack back into 3D, compute Dice for ET/TC/WT.
    """
    model.eval()
    dices = []
    for xs, ys, meta in tqdm(loader, desc="eval", leave=False):
        xs = xs[0].to(device)   # (S,4,H,W)
        ys = ys[0].to(device)   # (S,3,H,W)
        S = xs.shape[0]

        preds = []
        for i in range(0, S, batch_size_eval):
            xb = xs[i:i+batch_size_eval]
            pb = torch.sigmoid(model(xb))
            preds.append(pb.cpu())
        preds = torch.cat(preds, dim=0).numpy()  # (S,3,H,W)
        target_np = ys.cpu().numpy()

        vol_pred = stack_back(preds, loader.dataset.view, meta["xyz"])  # (3,X,Y,Z)
        vol_gt = stack_back(target_np, loader.dataset.view, meta["xyz"])
        vol_bin = (vol_pred >= thresh).astype(np.uint8)
        vol_gtb = (vol_gt >= 0.5).astype(np.uint8)

        d = dice_per_channel(vol_bin, vol_gtb)  # (3,)
        dices.append(d)

    dices = np.array(dices) if len(dices) else np.zeros((0, 3))
    mean_per_class = dices.mean(axis=0) if len(dices) else np.array([0.0, 0.0, 0.0])
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
    # Point to either:
    #   data/BRATS2017_preprocessed/Brats17TrainingData
    # or data/BRATS2017_preprocessed  (the code will append Brats17TrainingData automatically)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Start time: {start_time}")

    ap = argparse.ArgumentParser(description="3-view SNN training.")
    ap.add_argument("--val-fold", type=int, required=True,
                    help="1,2,3,4,5")
    ap.add_argument("--view", type=str, required=True,
                    help="'sagittal', 'coronal','axial'")
    ap.add_argument("--model", choices=["orig", "shallow", "medium", "deep"], default="orig",
                    help="Choose the BraTS model architecture")
    ap.add_argument("--batch", type=int, default=8, help="8 or 16")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="1e-3 or 5e-4")
    ap.add_argument("--year", type=str, required=True,
                    help="17 or 23")
    args = ap.parse_args()

    # ---- Config (edit here) ----
    if args.year == "17":
        data_root = "data/BRATS2017_preprocessed/Brats17TrainingData"
    elif args.year == "23":
        data_root = "/gpfs/scratch1/shared/apiaghiardelli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    val_fold = args.val_fold              # int in {1..5}, used as validation
    view = args.view            # 'sagittal' | 'coronal' | 'axial'
    # training
    epochs = 100
    batch_size = args.batch   # subjects per batch (each provides a sequence of slices)
    # Adadelta (as requested): lr=1.0, rho=0.95, eps=1e-8
    lr = args.lr
    rho = None
    eps = None
    weight_decay = 1e-5
    grad_clip = 0.3

    drop_empty_slices = True  # skip all-zero GT slices in training

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
        "batch_size": batch_size,
        "lr": lr,
        "rho": rho,
        "eps": eps,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "drop_empty_slices": drop_empty_slices,
        "lambda_bce": lambda_bce,
        "lambda_dice": lambda_dice,
        "eval_every": eval_every,
        "eval_batch_slices": eval_batch_slices,
        "prob_threshold": prob_threshold
    }

    print("\n=== CONFIG ===")
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
    train_ds = BratsSliceDataset(root=data_root, train_folds=train_folds, view=view, drop_empty=drop_empty_slices)
    val_ds = BratsVolumeDataset(root=data_root, val_fold=val_fold, view=view)
    print(f"Train slices: {len(train_ds)} | Val subjects: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Loss weights -> lambda_bce={lambda_bce}, lambda_dice={lambda_dice}")

    # ---- Model/Optim ----
    model = UNetLike2D(in_channels=4, out_channels=3).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Train/Eval Loop ----
    best_dice = -1.0
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, lambda_bce, lambda_dice, grad_clip)
        print(f"  train_loss: {tr_loss:.4f}")

        if epoch % eval_every == 0:
            metrics = evaluate_3d(model, val_loader, device, batch_size_eval=eval_batch_slices, thresh=prob_threshold)
            print(f"  val dice: ET={metrics['dice_ET']:.4f}  TC={metrics['dice_TC']:.4f}  WT={metrics['dice_WT']:.4f}  mean={metrics['dice_mean']:.4f}  (N={metrics['n_subjects']})")
            if metrics["dice_mean"] > best_dice:
                best_dice = metrics["dice_mean"]
                ckpt_path = f"checkpoint_fold{val_fold}_{view}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "dice_mean": best_dice,
                    "config": {
                        "val_fold": val_fold, "view": view,
                        "lambda_bce": lambda_bce, "lambda_dice": lambda_dice
                    }}, ckpt_path)
                print(f"  Saved best model -> {ckpt_path}")

    print("Done.")
