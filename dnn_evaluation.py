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
from dnn_3ch import UNetLike2D

# Use the exact per-subject dataset & stacking used by evaluate_3d_snn(). :contentReference[oaicite:2]{index=2}
from dnn_3ch import BratsVolumeDataset, stack_back   # :contentReference[oaicite:3]{index=3}

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


def load_model(ckpt: Path, device: torch.device) -> UNetLike2D:
    m = UNetLike2D(in_channels=4, out_channels=3).to(device)
    sd = torch.load(str(ckpt), map_location=device)
    # Your checkpoints from snn_fptt.py save {"model": state_dict, ...}. :contentReference[oaicite:4]{index=4}
    sd = sd["model"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m

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
    model: UNetLike2D,
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
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    results: Dict[str, Dict[str, np.ndarray | float]] = {}

    for xs, ys, meta in tqdm(loader, desc="eval", leave=False):
        xs = xs[0]   # (S,4,H,W)
        ys = ys[0]   # (S,3,H,W)
        S = xs.shape[0]

        preds = []
        view = dset.view
        xyz = meta["xyz"]
        sid = meta["sid"][0] if isinstance(meta["sid"], list) else meta["sid"]

        # Use k as a slice-batch size to control GPU memory.
        batch_slices = max(int(k), 1)
        for i in tqdm(range(0, S, batch_slices)):
            xb = xs[i:i+batch_slices].to(device, non_blocking=True)
            pb = torch.sigmoid(model(xb))
            preds.append(pb.cpu())
        preds = torch.cat(preds, dim=0).numpy()  # (S,3,H,W)
        print(f"Subject {sid} - stacked preds shape: {preds.shape}")
        print(f"Subject {sid} - stacked gt shape: {ys.shape}")
        print(f"Subject {sid} - view: {view}, xyz: {xyz}")
        target_np = ys.numpy()

        vol_pred = stack_back(preds, view, xyz)  # (3,X,Y,Z)
        vol_gt = stack_back(target_np, view, xyz)
     
        # Metrics (per view)
        vol_bin = (vol_pred >= 0.5).astype(np.uint8)
        vol_gtb = np.rint(vol_gt).astype(np.uint8)
        d = dice_per_channel(vol_bin, vol_gtb)           # (3,)
        nll = nll_from_probs(vol_pred, vol_gt)           # scalar

        results[str(sid)] = {"prob": vol_pred, "gt": vol_gt, "dice": d, "nll": nll}
        print(f"[{view.capitalize()}] Subject {sid}: Dice ET={d[0]:.4f} TC={d[1]:.4f} WT={d[2]:.4f}  NLL={nll:.5f}")
    return results


def evaluate_fold(
    data_root: Path,
    fold: int,
    ckpts: Dict[str, Path],
    k: int,
    threshold: float,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates a single fold for a given set of views.
    `ckpts` is a dictionary like {"sagittal": path, "axial": path}.
    """
    results_per_view = {}
    views_to_process = sorted(list(ckpts.keys()))
    
    print(f"\n--- Processing Fold {fold} for views: {views_to_process} ---")

    view_map = {"sag": "sagittal", "cor": "coronal", "axi": "axial"}

    print(f"\n--- Processing Fold {fold} for views: {[view_map[v] for v in views_to_process]} ---")

    # Run inference for each specified view
    for view in views_to_process:
        full_view_name = view_map[view]
        dset = BratsVolumeDataset(root=str(data_root), val_fold=fold, view=full_view_name)
        model = load_model(ckpts[view], device)
        results_per_view[view] = infer_view_volumes_and_metrics(model, dset, device, k, threshold, view)

    # Find common subjects across the processed views
    if not results_per_view:
        raise ValueError("No views were processed.")
    
    sids = set(next(iter(results_per_view.values())).keys())
    for view in views_to_process[1:]:
        sids.intersection_update(results_per_view[view].keys())
    
    sids = sorted(list(sids))
    if not sids:
        raise RuntimeError(f"No common subjects across views {views_to_process} for fold {fold}.")

    # Accumulators for metrics
    all_dices = {view: [] for view in views_to_process}
    all_nlls = {view: [] for view in views_to_process}
    ens_dices, ens_nlls = [], []

    for sid in sids:
        # Log per-view metrics
        for view in views_to_process:
            dice = results_per_view[view][sid]["dice"]
            nll = results_per_view[view][sid]["nll"]
            all_dices[view].append(np.asarray(dice))
            all_nlls[view].append(float(nll))

        # Ensemble Calculation
        prob_vols = [results_per_view[view][sid]["prob"] for view in views_to_process]
        vol_gt = results_per_view[views_to_process[0]][sid]["gt"] # GT is the same for all

        ens_prob = np.mean(prob_vols, axis=0)
        ens_bin  = (ens_prob >= threshold).astype(np.uint8)
        ens_d    = dice_per_channel(ens_bin, vol_gt.astype(np.uint8))
        ens_n    = nll_from_probs(ens_prob, vol_gt)

        ens_dices.append(ens_d)
        ens_nlls.append(ens_n)

    # Calculate and print fold means
    print(f"[Fold {fold}] ---- Means over {len(sids)} subjects ----")
    def _mean(arrs: List[np.ndarray]) -> np.ndarray:
        return np.stack(arrs, axis=0).mean(axis=0) if arrs else np.zeros(3, dtype=np.float64)

    final_results = {}
    for view in views_to_process:
        print(f"[Fold {fold}] View: {view}")
        mean_dice = _mean(all_dices[view])
        mean_nll = np.mean(all_nlls[view])
        print(f"  {view.capitalize():<9} Dice: ET={mean_dice[0]:.4f} TC={mean_dice[1]:.4f} WT={mean_dice[2]:.4f}  NLL={mean_nll:.5f}")
        final_results[f"{view}_ET"], final_results[f"{view}_TC"], final_results[f"{view}_WT"] = mean_dice
        final_results[f"{view}_NLL"] = mean_nll

    ens_mean = _mean(ens_dices)
    print(f"  Ensemble  Dice: ET={ens_mean[0]:.4f} TC={ens_mean[1]:.4f} WT={ens_mean[2]:.4f}  NLL={np.mean(ens_nlls):.5f}")
    final_results["ens_ET"], final_results["ens_TC"], final_results["ens_WT"] = ens_mean
    final_results["ens_NLL"] = np.mean(ens_nlls)
    final_results["ens_MEAN"] = float(ens_mean.mean())
    final_results["n_subjects"] = len(sids)

    return final_results


def main():
    ap = argparse.ArgumentParser(description="3-view SNN ensemble evaluation (per-view + ensemble Dice & NLL).")
    ap.add_argument("--data-root", type=Path,
                    default="data/BRATS2017_preprocessed/Brats17TrainingData",
                    help="Path to BRATS2017_preprocessed/Brats17TrainingData")
    ap.add_argument("--folds", type=int, nargs="+", default=[1,2,3,4,5],
                    help="Fold(s) to evaluate.")
    ap.add_argument("--ckpt-sag", type=str, required=True,
                    help="Checkpoint path/pattern for sagittal (allow '{fold}').")
    ap.add_argument("--ckpt-cor", type=str, required=False,
                    help="Checkpoint path/pattern for coronal (allow '{fold}').")
    ap.add_argument("--ckpt-axi", type=str, required=False,
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
    # # --- Scenario 1: Sagittal, Coronal, Axial on Fold 1 ---
    print("\n\n========== SCENARIO 1: Sag+Cor+Axi on Fold 1 ==========")
    if all([args.ckpt_sag, args.ckpt_cor, args.ckpt_axi]):
        ckpts_sc1 = {
            "sag": Path(args.ckpt_sag.format(fold=1)),
            "cor": Path(args.ckpt_cor.format(fold=1)),
            "axi": Path(args.ckpt_axi.format(fold=1)),
        }
        res = evaluate_fold(args.data_root, 1, ckpts_sc1, args.k, args.threshold, device)

        overall_view_sums["sag"].append(np.array([res["sag_ET"], res["sag_TC"], res["sag_WT"]]))
        overall_view_sums["axi"].append(np.array([res["axi_ET"], res["axi_TC"], res["axi_WT"]]))
        overall_view_sums["cor"].append(np.array([res["cor_ET"], res["cor_TC"], res["cor_WT"]]))
        overall_view_sums["ens"].append(np.array([res["ens_ET"], res["ens_TC"], res["ens_WT"]]))

        overall_view_nlls["axi"].append(res["axi_NLL"])
        overall_view_nlls["sag"].append(res["sag_NLL"])
        overall_view_nlls["cor"].append(res["cor_NLL"])
        overall_view_nlls["ens"].append(res["ens_NLL"])

        overall_ens_means.append(res["ens_MEAN"])
    else:
        print("Skipping Scenario 1: Missing one or more required checkpoints (sag, cor, axi).")
    # --- Scenario 2: Coronal, Sagittal on Folds 1, 2, 3 ---
    print("\n\n========== SCENARIO 2: Cor+Sag on Folds 1, 2, 3 ==========")
    if all([args.ckpt_cor, args.ckpt_sag]):
        for fold in [2, 3]:
            ckpts_sc2 = {
                "cor": Path(args.ckpt_cor.format(fold=fold)),
                "sag": Path(args.ckpt_sag.format(fold=fold)),
            }
            res = evaluate_fold(args.data_root, fold, ckpts_sc2, args.k, args.threshold, device)

            overall_view_sums["sag"].append(np.array([res["sag_ET"], res["sag_TC"], res["sag_WT"]]))
            overall_view_sums["cor"].append(np.array([res["cor_ET"], res["cor_TC"], res["cor_WT"]]))
            overall_view_sums["ens"].append(np.array([res["ens_ET"], res["ens_TC"], res["ens_WT"]]))

            overall_view_nlls["sag"].append(res["sag_NLL"])
            overall_view_nlls["cor"].append(res["cor_NLL"])
            overall_view_nlls["ens"].append(res["ens_NLL"])

            overall_ens_means.append(res["ens_MEAN"])
    else:
        print("Skipping Scenario 2: Missing one or more required checkpoints (cor, sag).")
    # --- Scenario 3: Axial, Sagittal on Folds 1, 4, 5 ---
    print("\n\n========== SCENARIO 3: Axi+Sag on Folds 1, 4, 5 ==========")
    if all([args.ckpt_axi, args.ckpt_sag]):
        for fold in [4]:
            ckpts_sc3 = {
                "axi": Path(args.ckpt_axi.format(fold=fold)),
                "sag": Path(args.ckpt_sag.format(fold=fold)),
            }
            res = evaluate_fold(args.data_root, fold, ckpts_sc3, args.k, args.threshold, device)
                
            overall_view_sums["sag"].append(np.array([res["sag_ET"], res["sag_TC"], res["sag_WT"]]))
            overall_view_sums["axi"].append(np.array([res["axi_ET"], res["axi_TC"], res["axi_WT"]]))
            overall_view_sums["ens"].append(np.array([res["ens_ET"], res["ens_TC"], res["ens_WT"]]))

            overall_view_nlls["sag"].append(res["sag_NLL"])
            overall_view_nlls["axi"].append(res["axi_NLL"])
            overall_view_nlls["ens"].append(res["ens_NLL"])

            overall_ens_means.append(res["ens_MEAN"])
    else:
        print("Skipping Scenario 3: Missing one or more required checkpoints (axi, sag).")

    print("\n\nAll scenarios completed.")

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
