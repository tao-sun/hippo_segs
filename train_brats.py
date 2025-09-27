#!/usr/bin/env python3
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from model import SNNBraTS
from data import BratsDataset

LABEL_NAMES = ["ET", "TC", "WT"]


def dice_per_channel(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    B, K, k, H, W = pred_bin.shape
    p = pred_bin.reshape(B, K, k, -1).float()
    t = target_bin.reshape(B, K, k, -1).float()
    inter = (p * t).sum(dim=(0, 2, 3))
    p_sum = p.sum(dim=(0, 2, 3))
    t_sum = t.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + eps) / (p_sum + t_sum + eps)
    return dice


def train_epoch_tbptt(model, loader, optimizer, criterion, device, k: int, grad_clip: float | None = 1.0):
    model.train()
    running_loss = 0.0
    dice_sum = torch.zeros(3, dtype=torch.float64)
    dice_batches = 0

    pbar = tqdm(loader, desc=f"train (k={k})")
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)    # (B,D,4,H,W)
        y = batch["label"].to(device, non_blocking=True)    # (B,D,3,H,W)
        B, D, C, H, W = x.shape

        # Collect predictions across D
        preds_all = []
        targets_all = []

        for t0 in range(0, D, k):
            t1 = min(t0 + k, D)
            x_win = x[:, t0:t1, ...]
            y_win = y[:, t0:t1, ...]

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_win, t0=t0)                    # (B,3,k,H,W)
            target = y_win.permute(0, 2, 1, 3, 4).contiguous()

            loss = criterion(logits, target)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            model.detach_states()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            preds_all.append(torch.sigmoid(logits).detach().cpu())
            targets_all.append(target.detach().cpu())

        # concat over D dimension
        preds_all = torch.cat(preds_all, dim=2)   # (B,3,D,H,W)
        targets_all = torch.cat(targets_all, dim=2)

        preds_bin = (preds_all > 0.5).to(targets_all.dtype)
        dice = dice_per_channel(preds_bin, targets_all)
        dice_sum += dice.double()
        dice_batches += 1

    epoch_loss = running_loss / max(1, len(loader))
    epoch_dice = (dice_sum / max(1, dice_batches)).float()
    return epoch_loss, epoch_dice


@torch.no_grad()
def eval_epoch_tbptt(model, loader, criterion, device, k: int):
    # --- BPTT-style EVAL: single full-sequence forward, no window loop ---
    model.eval()
    running_loss = 0.0
    dice_sum = torch.zeros(3, dtype=torch.float64)
    dice_batches = 0

    pbar = tqdm(loader, desc=f"eval  (BPTT)")
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)    # (B,D,4,H,W)
        y = batch["label"].to(device, non_blocking=True)    # (B,D,3,H,W)

        # Full sequence forward pass; t0=0 to signal sequence start
        logits = model(x, t0=0)                             # (B,3,D,H,W)
        target = y.permute(0, 2, 1, 3, 4).contiguous()      # (B,3,D,H,W)

        loss = criterion(logits, target)
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Dice over the entire D
        probs = torch.sigmoid(logits)
        preds_bin = (probs > 0.5).to(target.dtype)
        dice = dice_per_channel(preds_bin.cpu(), target.cpu())
        dice_sum += dice.double()
        dice_batches += 1

        # avoid any hidden-state leakage across batches
        model.detach_states()

    epoch_loss = running_loss / max(1, len(loader))
    epoch_dice = (dice_sum / max(1, dice_batches)).float()
    return epoch_loss, epoch_dice


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ----- config -----
    data_root = "data/BRATS2017_preprocessed/Brats17TrainingData"
    view = "sagittal"         # or "coronal" / "axial"
    val_fold = 1              # choose 1..5 for validation
    batch_size = 8
    epochs = 50
    k = 8                     # TBPTT window length
    learning_rate = 1e-2

    # ----- data -----
    # validation dataset
    val_ds = BratsDataset(root=data_root, fold=val_fold, view=view)
    # training datasets: all folds except val_fold
    train_folds = [f for f in range(1, 6) if f != val_fold]
    train_ds = ConcatDataset([BratsDataset(root=data_root, fold=f, view=view) for f in train_folds])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ----- model / loss / opt -----
    model = SNNBraTS(out_channels=3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    for epoch in range(epochs):
        tr_loss, tr_dice = train_epoch_tbptt(model, train_loader, optimizer, criterion, device, k)
        va_loss, va_dice = eval_epoch_tbptt (model, val_loader,   criterion, device, k)
        scheduler.step(va_loss)

        tr_dice_str = ", ".join(f"{n}:{v:.3f}" for n, v in zip(LABEL_NAMES, tr_dice))
        va_dice_str = ", ".join(f"{n}:{v:.3f}" for n, v in zip(LABEL_NAMES, va_dice))
        print(f"Epoch {epoch:03d} | "
              f"train loss {tr_loss:.4f} | train Dice [{tr_dice_str}] | "
              f"val loss {va_loss:.4f} | val Dice [{va_dice_str}]")


if __name__ == "__main__":
    main()


