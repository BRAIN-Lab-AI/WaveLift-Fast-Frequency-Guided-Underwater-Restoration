"""Paired underwater image dataset for WaveFlow-UIE."""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(root: str) -> List[Path]:
    """Return image files sorted by name."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(p for p in root_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


def _read_rgb(path: Path) -> np.ndarray:
    """Read image as RGB float32 in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _pad_to_min_size(lq: np.ndarray, gt: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pad paired images so random crop is always valid."""
    h, w = lq.shape[:2]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return lq, gt

    pad = ((0, pad_h), (0, pad_w), (0, 0))
    return np.pad(lq, pad, mode="reflect"), np.pad(gt, pad, mode="reflect")


def _paired_random_crop(lq: np.ndarray, gt: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the same spatial region from both images."""
    lq, gt = _pad_to_min_size(lq, gt, size)
    h, w = lq.shape[:2]
    top = np.random.randint(0, h - size + 1)
    left = np.random.randint(0, w - size + 1)
    return lq[top : top + size, left : left + size], gt[top : top + size, left : left + size]


def _paired_augment(lq: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply simple paired flip augmentations."""
    if np.random.rand() < 0.5:
        lq = np.flip(lq, axis=1)
        gt = np.flip(gt, axis=1)
    if np.random.rand() < 0.5:
        lq = np.flip(lq, axis=0)
        gt = np.flip(gt, axis=0)
    return lq.copy(), gt.copy()


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert HWC RGB float image to CHW tensor."""
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


class PairedUIEDataset(Dataset):
    """Dataset for paired underwater restoration images.

    The input and target directories are matched by filename. If exact filename
    matching fails, the loader falls back to sorted pairing, which is useful for
    datasets whose paired directories use consistent ordering.
    """

    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        patch_size: int = 256,
        is_train: bool = True,
        enlarge_ratio: int = 1,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.patch_size = patch_size
        self.is_train = is_train
        self.enlarge_ratio = max(int(enlarge_ratio), 1)

        input_paths = _list_images(input_dir)
        target_paths = _list_images(target_dir)
        target_by_name = {p.name: p for p in target_paths}

        pairs = []
        for in_path in input_paths:
            if in_path.name in target_by_name:
                pairs.append((in_path, target_by_name[in_path.name]))

        if not pairs and len(input_paths) == len(target_paths):
            pairs = list(zip(input_paths, target_paths))

        if not pairs:
            raise RuntimeError(
                f"No paired images found between {input_dir} and {target_dir}. "
                "Ensure matching filenames or equal sorted directory lengths."
            )

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs) * self.enlarge_ratio if self.is_train else len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pair_index = index % len(self.pairs)
        lq_path, gt_path = self.pairs[pair_index]

        lq = _read_rgb(lq_path)
        gt = _read_rgb(gt_path)

        if lq.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (lq.shape[1], lq.shape[0]), interpolation=cv2.INTER_AREA)

        if self.is_train:
            lq, gt = _paired_random_crop(lq, gt, self.patch_size)
            lq, gt = _paired_augment(lq, gt)

        return {
            "lq": _to_tensor(lq),
            "gt": _to_tensor(gt),
            "lq_path": str(lq_path),
            "gt_path": str(gt_path),
        }
