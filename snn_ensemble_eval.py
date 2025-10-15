#!/usr/bin/env python3
"""
3-view SNN TBPTT inference + per-view metrics + ensemble metrics.

- Mirrors evaluate_3d_snn() windowing (k, t0 stepping), then stack_back to 3D. :contentReference[oaicite:1]{index=1}
- Prints, per subject:
    * Sagittal: Dice(ET/TC/WT), NLL
    * Coronal : Dice(ET/TC/WT), NLL
    * Axial   : Dice(ET/TC/WT), NLL
    * Ensemble: Dice(ET/TC/WT), NLL
- Prints fold-level means for each of the above and overall across folds.

Usage
-----
python ensemble_eval_snn.py \
  --data-root data/BRATS2017_preprocessed/Brats17TrainingData \
  --folds 1 2 3 4 5 \
  --ckpt-sag checkpoints/checkpoint_snn_fold{fold}_sagittal.pt \
  --ckpt-cor checkpoints/checkpoint_snn_fold{fold}_coronal.pt \
  --ckpt-axi checkpoints/checkpoint_snn_fold{fold}_axial.pt \
  --k 16 --threshold 0.5 --device auto
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model import SNNBraTS
# Use the exact per-subject dataset & stacking used by evaluate_3d_snn(). :contentReference[oaicite:2]{index=2}
from snn_fptt import BratsVolumeDataset, stack_back   # :contentReference[oaicite:3]{index=3}

# ------------------ utils ------------------

def get_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def load_model(ckpt: Path, device: torch.device) -> SNNBraTS:
    m = SNNBraTS(out_channels=3).to(device)
    sd = torch.load(str(ckpt), map_location=device)
    # Your checkpoints from snn_fptt.py save {"model": state_dict, ...}. :contentReference[oaicite:4]{index=4}
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def dice_per_channel(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    """pred_bin, gt_bin: (3, X, Y, Z) in {0,1} -> returns (3,) Dice."""
    eps = 1e-6
    dices = []
    for c in range(pred_bin.shape[0]):
        p = pred_bin[c].astype(np.uint8).reshape(-1)
        g = gt_bin[c].astype(np.uint8).reshape(-1)
        inter = int((p & g).sum())
        denom = int(p.sum()) + int(g.sum())
        dices.append((2 * inter + eps) / (denom + eps))
    return np.array(dices, dtype=np.float64)


def nll_from_probs(prob: np.ndarray, gt: np.ndarray) -> float:
    """
    Bernoulli NLL with probabilities: -(y*log p + (1-y) log (1-p)).
    prob, gt: (3, X, Y, Z) in [0,1] and {0,1}.
    Returns mean NLL over all voxels & channels.
    """
    eps = 1e-7
    p = np.clip(prob, eps, 1.0 - eps)
    y = gt.astype(np.float32)
    nll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(nll.mean())


@torch.no_grad()
def infer_view_volumes_and_metrics(
    model: SNNBraTS,
    dset: BratsVolumeDataset,
    device: torch.device,
    k: int,
    threshold: float,
    view: str
) -> Dict[str, Dict[str, np.ndarray | float]]:
    """
    TBPTT-style inference per subject for ONE view (like evaluate_3d_snn), returning:
      sid -> {
        "prob": (3,X,Y,Z) float32 in [0,1],
        "gt":   (3,X,Y,Z) float32 {0,1},
        "dice": (3,) np.float64 for ET/TC/WT,
        "nll":  float
      }
    Windowing and stack_back match your reference eval path. :contentReference[oaicite:5]{index=5}
    """
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    results: Dict[str, Dict[str, np.ndarray | float]] = {}

    for xs, ys, meta in tqdm(loader, desc=f"eval_{view}", leave=True):
        # (1,S,4,H,W), (1,S,3,H,W)
        xs = xs.to(device)
        S = xs.shape[1]
        view = dset.view
        xyz = meta["xyz"]
        sid = meta["sid"][0] if isinstance(meta["sid"], list) else meta["sid"]

        # Collect per-slice probs like evaluate_3d_snn() does. :contentReference[oaicite:6]{index=6}
        preds_seq: List[np.ndarray] = []
        for t0 in range(0, S, k):
            t1 = min(t0 + k, S)
            x_win = xs[:, t0:t1, ...]                    # (1,k,4,H,W)
            logits = model(x_win, t0=t0)                 # (1,3,k,H,W) — same call signature as in eval. :contentReference[oaicite:7]{index=7}
            probs  = torch.sigmoid(logits).cpu().numpy() # (1,3,k,H,W)
            probs  = np.transpose(probs, (0, 2, 1, 3, 4))# (1,k,3,H,W)
            preds_seq.append(probs[0])                   # (k,3,H,W)
        preds = np.concatenate(preds_seq, axis=0)        # (S,3,H,W)

        # Reassemble to 3D, identical to reference. :contentReference[oaicite:8]{index=8}
        target_np = ys.cpu().numpy()[0]                  # (S,3,H,W)
        vol_prob = stack_back(preds, view, xyz).astype(np.float32)  # (3,X,Y,Z)
        vol_gt   = stack_back(target_np, view, xyz).astype(np.float32)

        # Metrics (per view)
        vol_bin = (vol_prob >= threshold).astype(np.uint8)
        vol_gtb = vol_gt.astype(np.uint8)
        d = dice_per_channel(vol_bin, vol_gtb)           # (3,)
        nll = nll_from_probs(vol_prob, vol_gt)           # scalar

        results[str(sid)] = {"prob": vol_prob, "gt": vol_gt, "dice": d, "nll": nll}

    return results


def evaluate_fold(
    data_root: Path,
    fold: int,
    ckpt_sag: Path,
    ckpt_cor: Path,
    ckpt_axi: Path,
    k: int,
    threshold: float,
    device: torch.device,
) -> Dict[str, float]:
    # Per-view datasets (same per-subject loader used in your reference). :contentReference[oaicite:9]{index=9}
    ds_sag = BratsVolumeDataset(root=str(data_root), val_fold=fold, view="sagittal")
    ds_cor = BratsVolumeDataset(root=str(data_root), val_fold=fold, view="coronal")
    ds_axi = BratsVolumeDataset(root=str(data_root), val_fold=fold, view="axial")

    # Load models
    m_sag = load_model(ckpt_sag, device)
    m_cor = load_model(ckpt_cor, device)
    m_axi = load_model(ckpt_axi, device)

    # Inference per view (metrics also per view)
    res_sag = infer_view_volumes_and_metrics(m_sag, ds_sag, device, k, threshold, "sagittal")
    res_cor = infer_view_volumes_and_metrics(m_cor, ds_cor, device, k, threshold, "coronal")
    res_axi = infer_view_volumes_and_metrics(m_axi, ds_axi, device, k, threshold, "axial")

    # Subjects in common
    sids = sorted(set(res_sag) & set(res_cor) & set(res_axi))
    if not sids:
        raise RuntimeError(f"No common subjects across views for fold {fold}.")

    # Accumulators
    sag_dices, cor_dices, axi_dices = [], [], []
    sag_nlls, cor_nlls, axi_nlls = [], [], []
    ens_dices, ens_nlls = [], []

    for sid in sids:
        # --- per-view logging ---
        d_s = res_sag[sid]["dice"]; n_s = res_sag[sid]["nll"]
        d_c = res_cor[sid]["dice"]; n_c = res_cor[sid]["nll"]
        d_a = res_axi[sid]["dice"]; n_a = res_axi[sid]["nll"]

        sag_dices.append(np.asarray(d_s)); sag_nlls.append(float(n_s))
        cor_dices.append(np.asarray(d_c)); cor_nlls.append(float(n_c))
        axi_dices.append(np.asarray(d_a)); axi_nlls.append(float(n_a))

        # Ensemble
        vol_s = res_sag[sid]["prob"]; vol_c = res_cor[sid]["prob"]; vol_a = res_axi[sid]["prob"]
        vol_gt = res_sag[sid]["gt"]  # same GT across views

        ens_prob = (vol_s + vol_c + vol_a) / 3.0
        ens_bin  = (ens_prob >= threshold).astype(np.uint8)
        ens_d    = dice_per_channel(ens_bin, vol_gt.astype(np.uint8))
        ens_n    = nll_from_probs(ens_prob, vol_gt)

        ens_dices.append(ens_d)
        ens_nlls.append(ens_n)

        # Print per-subject report
        print(f"[Fold {fold}] {sid} | "
              f"Sag Dice: ET={d_s[0]:.4f} TC={d_s[1]:.4f} WT={d_s[2]:.4f}  NLL={n_s:.5f} | "
              f"Cor Dice: ET={d_c[0]:.4f} TC={d_c[1]:.4f} WT={d_c[2]:.4f}  NLL={n_c:.5f} | "
              f"Ax Dice:  ET={d_a[0]:.4f} TC={d_a[1]:.4f} WT={d_a[2]:.4f}  NLL={n_a:.5f} | "
              f"Ensemble Dice: ET={ens_d[0]:.4f} TC={ens_d[1]:.4f} WT={ens_d[2]:.4f}  NLL={ens_n:.5f}")

    # Fold means
    def _mean(arrs: List[np.ndarray]) -> np.ndarray:
        return np.stack(arrs, axis=0).mean(axis=0) if arrs else np.zeros(3, dtype=np.float64)

    sag_mean = _mean(sag_dices); cor_mean = _mean(cor_dices); axi_mean = _mean(axi_dices)
    ens_mean = _mean(ens_dices)

    print(f"[Fold {fold}] ---- Means over {len(sids)} subjects ----")
    print(f"  Sagittal  Dice: ET={sag_mean[0]:.4f} TC={sag_mean[1]:.4f} WT={sag_mean[2]:.4f}  "
          f"NLL={np.mean(sag_nlls):.5f}")
    print(f"  Coronal   Dice: ET={cor_mean[0]:.4f} TC={cor_mean[1]:.4f} WT={cor_mean[2]:.4f}  "
          f"NLL={np.mean(cor_nlls):.5f}")
    print(f"  Axial     Dice: ET={axi_mean[0]:.4f} TC={axi_mean[1]:.4f} WT={axi_mean[2]:.4f}  "
          f"NLL={np.mean(axi_nlls):.5f}")
    print(f"  Ensemble  Dice: ET={ens_mean[0]:.4f} TC={ens_mean[1]:.4f} WT={ens_mean[2]:.4f}  "
          f"NLL={np.mean(ens_nlls):.5f}")

    return {
        "sag_ET": float(sag_mean[0]), "sag_TC": float(sag_mean[1]), "sag_WT": float(sag_mean[2]), "sag_NLL": float(np.mean(sag_nlls)),
        "cor_ET": float(cor_mean[0]), "cor_TC": float(cor_mean[1]), "cor_WT": float(cor_mean[2]), "cor_NLL": float(np.mean(cor_nlls)),
        "axi_ET": float(axi_mean[0]), "axi_TC": float(axi_mean[1]), "axi_WT": float(axi_mean[2]), "axi_NLL": float(np.mean(axi_nlls)),
        "ens_ET": float(ens_mean[0]), "ens_TC": float(ens_mean[1]), "ens_WT": float(ens_mean[2]), "ens_NLL": float(np.mean(ens_nlls)),
        "ens_MEAN": float(ens_mean.mean()),
        "n_subjects": len(sids),
    }


def main():
    ap = argparse.ArgumentParser(description="3-view SNN ensemble evaluation (per-view + ensemble Dice & NLL).")
    ap.add_argument("--data-root", type=Path,
                    default="data/BRATS2017_preprocessed/Brats17TrainingData",
                    help="Path to BRATS2017_preprocessed/Brats17TrainingData")
    ap.add_argument("--folds", type=int, nargs="+", default=[1,2,3,4,5],
                    help="Fold(s) to evaluate.")
    ap.add_argument("--ckpt-sag", type=str, required=True,
                    help="Checkpoint path/pattern for sagittal (allow '{fold}').")
    ap.add_argument("--ckpt-cor", type=str, required=True,
                    help="Checkpoint path/pattern for coronal (allow '{fold}').")
    ap.add_argument("--ckpt-axi", type=str, required=True,
                    help="Checkpoint path/pattern for axial (allow '{fold}').")
    ap.add_argument("--k", type=int, default=1, help="TBPTT window size for inference.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold.")
    ap.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', 'mps', or 'cpu'.")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Collect per-fold summaries to compute an overall mean
    overall_view_sums = {
        "sag": [], "cor": [], "axi": [], "ens": []
    }  # each will hold arrays of [ET,TC,WT] per fold
    overall_view_nlls = {"sag": [], "cor": [], "axi": [], "ens": []}
    overall_ens_means = []

    for fold in args.folds:
        ck_s = Path(args.ckpt_sag.format(fold=fold)) if "{fold}" in args.ckpt_sag else Path(args.ckpt_sag)
        ck_c = Path(args.ckpt_cor.format(fold=fold)) if "{fold}" in args.ckpt_cor else Path(args.ckpt_cor)
        ck_a = Path(args.ckpt_axi.format(fold=fold)) if "{fold}" in args.ckpt_axi else Path(args.ckpt_axi)
        for p in (ck_s, ck_c, ck_a):
            if not p.exists():
                raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {p}")

        res = evaluate_fold(
            data_root=args.data_root,
            fold=fold,
            ckpt_sag=ck_s, ckpt_cor=ck_c, ckpt_axi=ck_a,
            k=args.k, threshold=args.threshold, device=device
        )

        overall_view_sums["sag"].append(np.array([res["sag_ET"], res["sag_TC"], res["sag_WT"]]))
        overall_view_sums["cor"].append(np.array([res["cor_ET"], res["cor_TC"], res["cor_WT"]]))
        overall_view_sums["axi"].append(np.array([res["axi_ET"], res["axi_TC"], res["axi_WT"]]))
        overall_view_sums["ens"].append(np.array([res["ens_ET"], res["ens_TC"], res["ens_WT"]]))

        overall_view_nlls["sag"].append(res["sag_NLL"])
        overall_view_nlls["cor"].append(res["cor_NLL"])
        overall_view_nlls["axi"].append(res["axi_NLL"])
        overall_view_nlls["ens"].append(res["ens_NLL"])

        overall_ens_means.append(res["ens_MEAN"])

    if overall_ens_means:
        def _m(arrs): return np.stack(arrs, axis=0).mean(axis=0)
        def _std(arrs): return np.stack(arrs, axis=0).std(axis=0)

        print("\n========== Overall (across folds) ==========")
        for tag in ["sag", "cor", "axi", "ens"]:
            m = _m(overall_view_sums[tag])
            s = _std(overall_view_sums[tag])
            n_mean = np.mean(overall_view_nlls[tag])
            n_std = np.std(overall_view_nlls[tag])
            print(
                f"{tag.upper():>7} Dice: "
                f"ET={m[0]:.4f}±{s[0]:.4f}  "
                f"TC={m[1]:.4f}±{s[1]:.4f}  "
                f"WT={m[2]:.4f}±{s[2]:.4f}  "
                f"NLL={n_mean:.5f}±{n_std:.5f}"
            )

        ens_mean = np.mean(overall_ens_means)
        ens_std = np.std(overall_ens_means)
        print(f"\nEnsemble MEAN Dice across folds: {ens_mean:.4f} ± {ens_std:.4f}")


if __name__ == "__main__":
    main()
