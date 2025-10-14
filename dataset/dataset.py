import os
import re
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from alive_progress import alive_bar
import torchvision.transforms.functional as F
from .utils.openDAS_h5 import read_febus

DEFAULT_HEIGHT = 400
DEFAULT_WIDTH = 1200
IMAGE_DTYPE = np.float32
TARGET_DTYPE = np.int64
VALID_IMAGE_EXTS = (".npy",)

def _safe_normalize(x: np.ndarray) -> np.ndarray:
    """
    Robust max-abs normalization.
    If max abs is zero, return the array as-is to avoid NaNs/Infs.
    """
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return x
    return x / max_abs


def _extract_id_from_name(filename: str) -> Optional[int]:
    """
    Extract a numeric id from filename.
    Valid examples: '1234_trace.npy', '1234.npy', '1234-anything.npy'
    """
    m = re.match(r"^(\d+)(?:[_\-].*)?\.[A-Za-z0-9]+$", filename)
    return int(m.group(1)) if m else None


def load_and_preprocess_data(
    data_dir: str | os.PathLike,
    targets_dir: str | os.PathLike,
    scale_h: int = DEFAULT_HEIGHT,
    scale_w: int = DEFAULT_WIDTH,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Load .npy inputs from a directory and their matching targets from a directory.
    Resizing is deferred to the Dataset.

    Returns parallel lists: images, targets, ids.
    """
    data_dir = Path(data_dir)
    targets_dir = Path(targets_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not targets_dir.exists():
        targets_dir = Path(targets_dir)

    data_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(VALID_IMAGE_EXTS)])
    images: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    info_ids: List[int] = []

    with alive_bar(len(data_files), title=f"Loading {data_dir}") as bar:
        for fname in data_files:
            data_path = data_dir / fname
            arr = np.load(data_path)

            id_guess = _extract_id_from_name(fname)
            if id_guess is None:
                id_guess = abs(hash(fname)) % (10**9)

            target_path = targets_dir / f"0{id_guess}.npy"
    
            if target_path.exists():
                tgt = np.load(target_path)
            else:
                tgt = np.zeros((scale_h, scale_w))

            images.append(arr)
            targets.append(tgt)
            info_ids.append(int(id_guess))

            bar()

    return images, targets, info_ids

class SignalDataset(Dataset):
    """
    DAS signal dataset:
    - max-abs normalization on input
    - resize input/target to (scale_h, scale_w)
    - optional simple augmentation (Gaussian noise + random mask)
    """
    def __init__(
        self,
        signals: List[np.ndarray],
        targets: List[np.ndarray],
        info: List[int],
        data_augmentation: bool = False,
        scale_h: int = DEFAULT_HEIGHT,
        scale_w: int = DEFAULT_WIDTH,
    ) -> None:
        assert len(signals) == len(targets) == len(info), "Input lists must have the same length."
        self.signals = signals
        self.targets = targets
        self.info = info
        self.data_augmentation = data_augmentation
        self.scale_h, self.scale_w = int(scale_h), int(scale_w)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        image_np = self.signals[idx]
        target_np = self.targets[idx]
        info_id = int(self.info[idx])

        image_t, target_t = self.transform(image_np, target_np, self.data_augmentation)
        return image_t, target_t, torch.tensor(info_id, dtype=torch.int64)

    def input_transform(self, image: np.ndarray) -> Tensor:
        image = _safe_normalize(image)
        pil = Image.fromarray((image * 255.0).astype(np.uint8))
        pil = pil.resize((self.scale_w, self.scale_h))
        tensor = F.to_tensor(pil)
        return tensor

    def target_transform(self, target: np.ndarray) -> Tensor:
        pil = Image.fromarray(target.astype(np.uint8))
        pil = pil.resize((self.scale_w, self.scale_h), resample=Image.NEAREST)
        arr = np.array(pil, dtype=TARGET_DTYPE)
        return torch.from_numpy(arr) 

    def transform(self, image: np.ndarray, target: np.ndarray, do_aug: bool = False) -> Tuple[Tensor, Tensor]:
        img_t = self.input_transform(image)
        tgt_t = self.target_transform(target)

        if do_aug and random.random() < 0.5:
            noise = torch.randn_like(img_t) * 0.05
            mask = (torch.rand_like(img_t) > 0.5).float()
            img_t = torch.clamp(img_t + noise * mask, 0.0, 1.0)

        return img_t, tgt_t

    def test_transform(self, image: np.ndarray) -> Tensor:
        return self.input_transform(image)

def _compute_class_weights(targets: List[np.ndarray], num_classes: Optional[int] = None) -> np.ndarray:
    """
    Compute inverse-frequency class weights with absent-class handling.
    Returns weights normalized to sum to 1.
    """
    flattened = np.concatenate([t.ravel() for t in targets]).astype(np.int64, copy=False)
    if num_classes is None:
        num_classes = int(flattened.max()) + 1 if flattened.size > 0 else 1

    counts = np.bincount(flattened, minlength=num_classes).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_freq = np.where(counts > 0, 1.0 / counts, 0.0)
    if inv_freq.sum() == 0:
        weights = np.ones(num_classes, dtype=np.float64) / num_classes
    else:
        weights = inv_freq / inv_freq.sum()
    return weights.astype(np.float32)

def get_dataset(
    train_data_dir: str,
    val_data_dir: str,
    test_data_dir: str,
    targets_dir: str,
    scale_h: int = DEFAULT_HEIGHT,
    scale_w: int = DEFAULT_WIDTH,
    num_classes: Optional[int] = 3,
) -> Tuple[SignalDataset, np.ndarray, SignalDataset, SignalDataset]:
    """
    Build train/val/test datasets and class weights.
    All paths must be directories.
    """
    # Load sets
    tr_imgs, tr_tgts, tr_ids = load_and_preprocess_data(train_data_dir, targets_dir, scale_h, scale_w)
    va_imgs, va_tgts, va_ids = load_and_preprocess_data(val_data_dir, targets_dir, scale_h, scale_w)
    te_imgs, te_tgts, te_ids = load_and_preprocess_data(test_data_dir, targets_dir, scale_h, scale_w)

    # Class weights
    class_weights = _compute_class_weights(tr_tgts, num_classes=num_classes)

    # Datasets
    train_dataset = SignalDataset(tr_imgs, tr_tgts, tr_ids, data_augmentation=True,  scale_h=scale_h, scale_w=scale_w)
    val_dataset   = SignalDataset(va_imgs, va_tgts, va_ids, data_augmentation=False, scale_h=scale_h, scale_w=scale_w)
    test_dataset  = SignalDataset(te_imgs, te_tgts, te_ids, data_augmentation=False, scale_h=scale_h, scale_w=scale_w)

    return train_dataset, class_weights, val_dataset, test_dataset


def get_events(test_data_file: str, mode: str = "array") -> Tuple[Optional[Tensor], Optional[np.ndarray]]:
    """
    Load a single sample for inference:

      - mode='array': `test_data_file` is a .npy file on disk
      - otherwise   : try reading with `read_febus` (h5)

    Returns (tensor_input[B=1,C,H,W], original_array) or (None, None) if it fails.
    """
    ds = SignalDataset([], [], [], data_augmentation=False)  

    if mode == "array":
        if not os.path.isfile(test_data_file):
            print('File not found')
            return None, None
        arr = np.load(test_data_file)
        signal = ds.test_transform(arr).unsqueeze(0)  # [1,C,H,W]
        return signal, arr

    arr = read_febus(test_data_file)
    if isinstance(arr, np.ndarray):
        signal = ds.test_transform(arr).unsqueeze(0)
        return signal, arr

    return None, None
