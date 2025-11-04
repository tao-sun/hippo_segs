#!/usr/bin/env python3
import os
import random
import json
from pathlib import Path
import numpy as np
import nibabel as nib

try:
    import imageio.v2 as imageio
except Exception:
    import imageio

# ==================== CONFIG ====================
IN_ROOT  = Path("BRATS2023")                 # <- change me
OUT_ROOT = Path("BRATS2023_preprocessed")    # <- will be created
TRAIN_DIR_NAME = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
# HGG_DIR_NAME   = "HGG"                                 # ONLY process HGG
TARGET_SHAPE   = (160, 192, 152)                       # (x,y,z)
SEED           = 2025                                  # 5-fold split seed
FOLD_NAMES     = ["1", "2", "3", "4", "5"]
# =================================================

def center_crop3d(arr: np.ndarray, tgt=(160,192,152)) -> np.ndarray:
    assert arr.ndim == 3
    slices = []
    for s, t in zip(arr.shape, tgt):
        if t > s:
            raise ValueError(f"Target {t} > source {s}.")
        st = (s - t) // 2
        slices.append(slice(st, st + t))
    return arr[tuple(slices)]

def minmax(volume: np.ndarray) -> np.ndarray:
    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    if vmax > vmin:
        return (volume - vmin) / (vmax - vmin)
    return np.zeros_like(volume, dtype=np.float32)

def is_nifti(p: Path) -> bool:
    s = p.name.lower()
    return s.endswith(".nii") or s.endswith(".nii.gz")

def is_seg_file(p: Path) -> bool:
    return "-seg.nii.gz" in p.name.lower()

def save_png_views(norm_vol: np.ndarray, subject_out_dir: Path, stem: str) -> None:
    """
    norm_vol: (160,192,152) in (x,y,z)
    Creates sagittal/, coronal/, axial/ under subject_out_dir.
    """
    x, y, z = norm_vol.shape
    assert (x, y, z) == TARGET_SHAPE

    dir_sag = subject_out_dir / "sagittal"  # 160 slices, each (192,152)
    dir_cor = subject_out_dir / "coronal"   # 192 slices, each (160,152)
    dir_axi = subject_out_dir / "axial"     # 152 slices, each (160,192)
    for d in (dir_sag, dir_cor, dir_axi):
        d.mkdir(parents=True, exist_ok=True)

    # sagittal (iterate x): (y,z) = (192,152)
    for i in range(x):
        sl = norm_vol[i, :, :]
        img = np.clip(np.rint(sl * 255.0), 0, 255).astype(np.uint8)
        imageio.imwrite(str(dir_sag / f"{stem}_sagittal_{i:03d}.png"), img)

    # coronal (iterate y): (x,z) = (160,152)
    for i in range(y):
        sl = norm_vol[:, i, :]
        img = np.clip(np.rint(sl * 255.0), 0, 255).astype(np.uint8)
        imageio.imwrite(str(dir_cor / f"{stem}_coronal_{i:03d}.png"), img)

    # axial (iterate z): (x,y) = (160,192)
    for i in range(z):
        sl = norm_vol[:, :, i]
        img = np.clip(np.rint(sl * 255.0), 0, 255).astype(np.uint8)
        imageio.imwrite(str(dir_axi / f"{stem}_axial_{i:03d}.png"), img)

def process_modality_to_pngs(mod_path: Path, subject_out_dir: Path) -> None:
    img = nib.load(str(mod_path))
    data = img.get_fdata(dtype=np.float32)               # load as float32
    cropped = center_crop3d(data, TARGET_SHAPE)
    norm = minmax(cropped).astype(np.float32)
    stem = mod_path.name[:-7] if mod_path.name.endswith(".nii.gz") else mod_path.stem
    save_png_views(norm, subject_out_dir, stem)

def process_seg(seg_path: Path, subject_out_dir: Path) -> None:
    """
    Center-crop *_seg.nii to (160,192,152) and save NIfTI (int16) in the subject folder.
    Label values {0,1,2,4} preserved. Affine/header copied.
    """
    seg_img = nib.load(str(seg_path))
    seg = seg_img.get_fdata(dtype=np.float32)
    seg = np.rint(seg).astype(np.int16)
    seg = center_crop3d(seg, TARGET_SHAPE)

    hdr = seg_img.header.copy()
    hdr.set_data_dtype(np.int16)

    subject_out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(seg, affine=seg_img.affine, header=hdr), str(subject_out_dir / seg_path.name))

def collect_train_subjects(train_root: Path) -> list[Path]:
    """
    Return subject dirs under train_root.
    """
    if not train_root.exists():
        raise FileNotFoundError(f"Missing train_root dir: {train_root}")
    subjects = [p for p in train_root.iterdir() if p.is_dir()]
    # Filter to those with a seg and at least one modality nii
    keep = []
    for subj in subjects:
        niis = [p for p in subj.iterdir() if is_nifti(p)]
        if not niis:
            f"[Warn] Found {len(subjects)} train subjects (expected 1251). Proceeding anyway."
            print(f"[Warn] no niis in {subj}. Proceeding anyway.")
            continue
        if not any(is_seg_file(p) for p in niis):
            print(f"[Warn] no ground truth seg in {subj}. Proceeding anyway.")
            continue
        
        keep.append(subj)
    return keep

def make_folds(subject_dirs: list[Path], seed: int = 2025) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    idxs = list(range(len(subject_dirs)))
    rng.shuffle(idxs)

    n = len(subject_dirs)
    assert n == 1251, f"Expected 1251 train subjects, but got {n}"

    # split into 5 equal folds (1251 → 250 each)
    folds = {name: [] for name in FOLD_NAMES}
    fold_sizes = [n // 5] * 5
    for i in range(n % 5):
        fold_sizes[i] += 1
    start = 0
    for fi, name in enumerate(FOLD_NAMES):
        end = start + fold_sizes[fi]
        for j in idxs[start:end]:
            folds[name].append(subject_dirs[j])
        start = end
    return folds


def main():
    in_train_root = IN_ROOT / TRAIN_DIR_NAME
    print(f"Start processing: {in_train_root}")
    out_train_root = OUT_ROOT / TRAIN_DIR_NAME
    out_train_root.mkdir(parents=True, exist_ok=True)

    subjects = collect_train_subjects(in_train_root)
    if not subjects:
        raise RuntimeError(f"No train subjects found in {in_train_root}")

    if len(subjects) != 1251:
        print(f"[Warn] Found {len(subjects)} train subjects (expected 1251). Proceeding anyway.")

    folds = make_folds(subjects, seed=SEED)

    # Save manifest with train subjects
    (out_train_root / "folds_manifest.json").write_text(
        json.dumps({k: [s.name for s in v] for k, v in folds.items()}, indent=2)
    )

    count_mod = 0
    count_seg = 0

    for fold_name, subj_dirs in folds.items():
        print(f"Fold {fold_name} started")
        fold_num = 0
        fold_out_root = out_train_root / fold_name
        fold_out_root.mkdir(parents=True, exist_ok=True)

        for subj_dir in subj_dirs:
            subject_out_dir = fold_out_root / subj_dir.name

            # Segmentation first
            seg_candidates = list(subj_dir.glob("*_seg.nii")) + list(subj_dir.glob("*-seg.nii.gz"))
            if seg_candidates:
                process_seg(seg_candidates[0], subject_out_dir)
                count_seg += 1
            else:
                print(f"[Warn] No seg file for subject {subj_dir.name}")

            # Modalities → PNG views
            for nii in subj_dir.iterdir():
                if not is_nifti(nii) or is_seg_file(nii):
                    continue
                process_modality_to_pngs(nii, subject_out_dir)
                count_mod += 1
                # if count_mod % 100 == 0:
                #     print(f"{count_mod} modality volumes processed...")

            fold_num += 1
            if fold_num % 50 == 0:
                print(f"  {fold_num} processed!")

        # ---- ASSERT after finishing this fold ----
        expected = len(subj_dirs)
        existing = [p for p in fold_out_root.glob("BraTS-GLI-*") if p.is_dir()]
        n_existing = len(existing)
        assert n_existing == expected, (
            f"Fold {fold_name} mismatch: expected {expected}, found {n_existing} Brats17_* subdirs"
        )
        print(f"  Fold {fold_name} finished: {n_existing} subjects OK")

    print(
        "Done.\n"
        f"  {count_mod} modality volumes → PNG views (train only)\n"
        f"  {count_seg} *_seg.nii cropped & saved\n"
        f"  Folds under: {out_train_root}"
    )


if __name__ == "__main__":
    main()
