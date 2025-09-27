import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class HippoDataset(Dataset):
    def __init__(self, path, subjects, frames):
        self.path = path
        self.frames = frames
        self.subjects = subjects
        return

    def __getitem__(self, index):
        start_idx = index * self.frames + self.subjects[0]
        images, labels = [], []
        for idx in range(start_idx, start_idx+48):
            img = read_image(f"{self.path}/hippo{idx}.png").squeeze()
            # img.type(torch.float)
            image = img * (1. / 255)
            images.append(image)

            label = read_image(f"{self.path}/label{idx}.png").squeeze()
            label = label * (1. / 255)
            label = label.long()
            labels.append(label)
        
        images = torch.stack(images)
        labels = torch.stack(labels)
        # print(f"images: {images.shape}, labels: {labels.dtype}")
        return images, labels

    def __len__(self):
        return len(self.subjects)



from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import nibabel as nib
import numpy as np

MOD_ORDER = ["t1", "t1ce", "t2", "flair"]          # input channels
LABEL_TOKENS = ["et", "tc", "wt"]                  # output channels (multi-label)
VALID_VIEWS = {"sagittal", "coronal", "axial"}
TARGET_SHAPE = (160, 192, 152)                     # (x,y,z)

def _brats_intmask_to_multilabel(mask3d: np.ndarray) -> np.ndarray:
    """BraTS labels {0,1,2,4} -> multilabel channels [ET, TC, WT]; returns (3,x,y,z) float32."""
    m = mask3d.astype(np.int32)
    et = (m == 4)
    tc = (m == 1) | (m == 4)
    wt = (m == 1) | (m == 2) | (m == 4)
    return np.stack([et, tc, wt], axis=0).astype(np.float32)

class BratsDataset(Dataset):
    """
    root: path to BRATS2017_preprocessed/Brats17TrainingData
    fold: int in {1..5}
    view: 'sagittal' | 'coronal' | 'axial'
    Returns per item (one subject):
      {
        'image':  (D, 4, H, W) float32 in [0,1],
        'label':  (D, K, H, W) float32 {0,1},
        'subject': str,
        'fold':    str,
        'view':    str
      }
    """
    def __init__(
        self,
        root: str,
        fold: int,
        view: str,
        label_tokens: Optional[List[str]] = None,
        to_tensor: Optional[callable] = None,
    ) -> None:
        self.root = Path(root)  # .../BRATS2017_preprocessed/Brats17TrainingData
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        if not isinstance(fold, int) or fold < 1 or fold > 5:
            raise ValueError("fold must be an int in {1,2,3,4,5}.")
        self.fold = str(fold)

        if view not in VALID_VIEWS:
            raise ValueError(f"Unknown view '{view}'. Choose from {VALID_VIEWS}.")
        self.view = view

        self.label_tokens = label_tokens or LABEL_TOKENS
        self.to_tensor = to_tensor or transforms.ToTensor()  # PNG uint8 -> (1,H,W) float32 [0,1]

        fold_dir = self.root / self.fold
        if not fold_dir.exists():
            raise FileNotFoundError(fold_dir)

        self.items: List[Dict[str, Any]] = []
        # subjects directly under the fold
        for subj_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir()):
            view_dir = subj_dir / self.view
            if not view_dir.exists():
                continue

            # collect modality PNGs
            img_paths_by_mod: Dict[str, List[Path]] = {
                m: sorted(view_dir.glob(f"Brats17_*_{m}_*.png")) for m in MOD_ORDER
            }
            if not all(img_paths_by_mod[m] for m in MOD_ORDER):
                continue
            counts = [len(v) for v in img_paths_by_mod.values()]
            if len(set(counts)) != 1:
                continue
            D = counts[0]

            # find cropped seg in the same subject dir
            seg_candidates = list(subj_dir.glob("*_seg.nii")) + list(subj_dir.glob("*_seg.nii.gz"))
            if not seg_candidates:
                continue
            seg_path = seg_candidates[0]

            self.items.append({
                "subject": subj_dir.name,
                "fold": self.fold,
                "view_dir": view_dir,
                "img_paths_by_mod": img_paths_by_mod,
                "seg_path": seg_path,
                "D": D
            })

        if not self.items:
            raise RuntimeError(f"No subjects found in fold {self.fold} for view='{self.view}' under {self.root}.")

        # Expected (D,H,W) by view (from TARGET_SHAPE = (x,y,z))
        if self.view == "sagittal":
            self.expected_DHW = (TARGET_SHAPE[0], TARGET_SHAPE[1], TARGET_SHAPE[2])  # (160,192,152)
        elif self.view == "coronal":
            self.expected_DHW = (TARGET_SHAPE[1], TARGET_SHAPE[0], TARGET_SHAPE[2])  # (192,160,152)
        else:  # axial
            self.expected_DHW = (TARGET_SHAPE[2], TARGET_SHAPE[0], TARGET_SHAPE[1])  # (152,160,192)

    def __len__(self) -> int:
        return len(self.items)

    def _load_images(self, img_paths_by_mod: Dict[str, List[Path]], D: int) -> torch.Tensor:
        frames = []
        for i in range(D):
            chans = []
            for m in MOD_ORDER:
                x = Image.open(img_paths_by_mod[m][i]).convert("L")
                t = self.to_tensor(x).squeeze(0)  # (H,W) float32 [0,1]
                chans.append(t)
            frames.append(torch.stack(chans, dim=0))     # (4,H,W)
        vol = torch.stack(frames, dim=0)                # (D,4,H,W)

        # sanity check on expected dims (not enforced)
        Dexp, Hexp, Wexp = self.expected_DHW
        if vol.shape[0] != Dexp or vol.shape[2] != Hexp or vol.shape[3] != Wexp:
            pass
        return vol

    def _load_labels_from_seg(self, seg_path: Path) -> torch.Tensor:
        seg_img = nib.load(str(seg_path))
        seg = seg_img.get_fdata(dtype=np.float32)
        seg = np.rint(seg).astype(np.int16)             # ensure ints
        # If not cropped already, defensively center-crop to TARGET_SHAPE
        if seg.shape != TARGET_SHAPE:
            x, y, z = seg.shape
            tx, ty, tz = TARGET_SHAPE
            xs, ys, zs = ( (x - tx)//2, (y - ty)//2, (z - tz)//2 )
            seg = seg[xs:xs+tx, ys:ys+ty, zs:zs+tz]

        ml = _brats_intmask_to_multilabel(seg)          # (K, x, y, z)
        K, x, y, z = ml.shape

        frames = []
        if self.view == "sagittal":
            for i in range(x):
                frames.append(torch.from_numpy(ml[:, i, :, :]))  # (K,192,152)
        elif self.view == "coronal":
            for i in range(y):
                frames.append(torch.from_numpy(ml[:, :, i, :]))  # (K,160,152)
        else:  # axial
            for i in range(z):
                frames.append(torch.from_numpy(ml[:, :, :, i]))  # (K,160,192)

        labels = torch.stack(frames, dim=0).float()     # (D,K,H,W)
        return labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        D = item["D"]
        images = self._load_images(item["img_paths_by_mod"], D)   # (D,4,H,W)
        labels = self._load_labels_from_seg(item["seg_path"])     # (D,K,H,W)
        return {
            "image": images,
            "label": labels,
            "subject": item["subject"],
            "fold": item["fold"],
            "view": self.view,
        }

