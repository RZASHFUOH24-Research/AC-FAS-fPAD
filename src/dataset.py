"""
Dataset utilities for AC-FAS.

Loads plain folders of real/live and spoof face images (sub-folders are
included). Labels follow the convention used throughout this codebase:
1 = live/real, 0 = spoof.

Spatial augmentations (crop, flip, jitter, rotation) and ToTensor are done
per-sample here. WFRF (frequency-domain reweighting) and ImageNet
normalisation are deliberately done at the *batch* level, in train.py /
evaluate.py, since WFRF's mode is selected once per mini-batch (see
wfrf.py) and normalisation must happen after WFRF.
"""

import os

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_SIZE = 224  # matches TinyViT-5M's expected input resolution ('tiny_vit_5m_224')
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """ImageNet normalisation, applied to a batch after WFRF."""
    return (batch - IMAGENET_MEAN.to(batch.device)) / IMAGENET_STD.to(batch.device)


class FaceImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = Image.fromarray(cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE)))
        if self.transform:
            image_rgb = self.transform(image_rgb)
        return image_rgb, self.labels[idx]


def _list_images(folder):
    """All images under `folder`, including sub-folders, in a fixed (sorted) order."""
    out = []
    for root, _, files in os.walk(folder):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(VALID_EXTENSIONS)]
    return sorted(out)


def _collect(real_dir, spoof_dirs):
    real_paths = _list_images(real_dir)
    spoof_paths = []
    for d in spoof_dirs:
        if os.path.exists(d):
            spoof_paths.extend(_list_images(d))
    paths = real_paths + spoof_paths
    labels = [1] * len(real_paths) + [0] * len(spoof_paths)
    print(f"{real_dir}: real {len(real_paths)} | spoof {len(spoof_paths)}")
    return paths, labels


def build_train_transform():
    # Spatial augmentations + ToTensor only. No Normalize here (see module docstring).
    return T.Compose(
        [
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
        ]
    )


def build_eval_transform():
    return T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()])


def build_train_loader(real_dir, spoof_dirs, batch_size, shuffle=True, num_workers=0):
    """`real_dir`: folder of live images. `spoof_dirs`: list of one or more
    folders of spoof images (empty list => strict one-class training).

    drop_last=True avoids a size-1 final batch, which the BatchNorm layer of
    the projection head cannot process in training mode."""
    paths, labels = _collect(real_dir, spoof_dirs)
    dataset = FaceImageDataset(paths, labels, build_train_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_eval_loader(real_dir, spoof_dirs, batch_size, num_workers=0):
    paths, labels = _collect(real_dir, spoof_dirs)
    dataset = FaceImageDataset(paths, labels, build_eval_transform())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
