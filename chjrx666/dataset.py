"""
Unified dataset loaders for MVTec AD and Omni-AD-30.

Both datasets share the same directory layout::

    <root>/<category>/train/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/*.png   (or *_mask.png)

``TrainDataset`` loads normal (``good``) images only.
``TestDataset`` loads all test images with ground-truth masks.

Usage::

    from dataset import TrainDataset, TestDataset, DATASET_ROOTS

    train_ds = TrainDataset(DATASET_ROOTS["omniad"], "air_conditioner_filter")
    test_ds  = TestDataset(DATASET_ROOTS["omniad"], "air_conditioner_filter")
"""

import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# =============================================================================
# Pre-configured dataset roots
# =============================================================================

DATASET_ROOTS: Dict[str, str] = {
    "mvtad": os.path.join(
        os.path.dirname(__file__), "..", "data", "mvtec_anomaly_detection"
    ),
    "omniad": os.path.join(
        os.path.dirname(__file__), "..", "data",
        "Omni-AD-30-release", "Omni-AD-30-release",
    ),
}

DATASET_CATEGORIES: Dict[str, List[str]] = {
    "mvtad": [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ],
    "omniad": [
        "air_conditioner_filter", "battery_piece", "battery_tab2",
        "button1", "capacitor1", "capacitor3", "ceramic_wafer",
        "character_string", "chip1", "cloth3", "coffee_cup_lid1",
        "dumplings2", "ice_cream_stick", "infusion_bottle_bottom5",
        "infusion_pipe_interface2", "iron_lattice", "led",
        "lunch_box_lid", "nameplate7", "remote_control",
        "resistance13", "screw_thread", "silicon_piece1",
        "solar_panel", "spindle_top", "wafer2", "webbing",
        "work_piece12", "work_piece14", "work_piece7",
    ],
}


# =============================================================================
# Image transforms  (same as Dinomaly get_data_transforms, dataset.py:19-31)
# =============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _is_image(fname: str) -> bool:
    return fname.lower().endswith(IMG_EXTENSIONS)


# =============================================================================
# TrainDataset  —  normal images only
# =============================================================================

class TrainDataset(Dataset):
    """
    Loads every normal (``good``) image from::

        <root>/<category>/train/good/

    Transforms: Resize → ToTensor → Normalize (ImageNet stats).
    Matches Dinomaly pipeline but keeps full 448×448 (no CenterCrop to 392)
    so that ``num_tokens == patch_count == 1024`` for LinearAttention.
    """

    def __init__(self, root: str, category: str, img_size: int = 448):
        self.root = root
        self.category = category
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        good_dir = os.path.join(root, category, "train", "good")
        if not os.path.isdir(good_dir):
            raise FileNotFoundError(
                f"Training directory not found: {good_dir}\n"
                f"Expected layout: <root>/<category>/train/good/"
            )

        self.paths = sorted([
            os.path.join(good_dir, f)
            for f in os.listdir(good_dir)
            if _is_image(f)
        ])
        if not self.paths:
            raise RuntimeError(f"No images found in {good_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)                         # (3, H, W)


# =============================================================================
# MultiCategoryTrainDataset  —  all categories, one model
# =============================================================================

class MultiCategoryTrainDataset(Dataset):
    """
    Loads normal (``good``) images from **all** categories::

        <root>/<category>/train/good/*.png   for every category in class_list

    Used for multi-category joint training — a single model learns to
    reconstruct normal patterns across diverse object types.

    Returns ``(image_tensor, category_name)``.
    """

    def __init__(self, root: str, categories: list, img_size: int = 448):
        self.root = root
        self.img_size = img_size
        self.categories = categories

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.paths: list = []
        self.cat_labels: list = []       # category index per image (for stats)

        for ci, cat in enumerate(categories):
            good_dir = os.path.join(root, cat, "train", "good")
            if not os.path.isdir(good_dir):
                print(f"  [WARN] skipping {cat}: {good_dir} not found", flush=True)
                continue
            files = sorted([
                os.path.join(good_dir, f)
                for f in os.listdir(good_dir)
                if _is_image(f)
            ])
            if not files:
                print(f"  [WARN] skipping {cat}: no images in {good_dir}", flush=True)
                continue
            self.paths.extend(files)
            self.cat_labels.extend([ci] * len(files))
            print(f"  {cat}: {len(files)} images", flush=True)

        if not self.paths:
            raise RuntimeError(f"No training images found under {root}")

        print(f"  Total: {len(self.paths)} images from {len(categories)} categories",
              flush=True)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)                         # (3, H, W)


# =============================================================================
# CachedFeatureDataset  —  pre-computed encoder features (skip image → encoder)
# =============================================================================

class CachedFeatureDataset(Dataset):
    """
    Loads pre-computed frozen-encoder features instead of raw images.

    Each cache file is a single **fp16** tensor of shape ``(L, N, C)`` where::

        L = number of target layers (8)
        N = patch tokens (1024 @ 448px, patch_size=14 → 32×32 grid)
        C = embed dim (768 for ViT-B)

    The DataLoader collates per-image tensors to ``(B, L, N, C)``.

    Layout expected::

        <cache_root>/<category>/<img_stem>.pt

    Used by ``train.py --use-cache`` — avoids re-running the frozen encoder
    every epoch (encoder forward is the dominant per-step cost).

    See ``precompute_features.py`` for the cache generator.
    """

    def __init__(self, cache_root: str, categories: list, num_layers: int = 8):
        self.cache_root = cache_root
        self.num_layers = num_layers
        self.paths: list = []

        for cat in categories:
            cat_dir = os.path.join(cache_root, cat)
            if not os.path.isdir(cat_dir):
                continue
            files = sorted([
                os.path.join(cat_dir, f)
                for f in os.listdir(cat_dir)
                if f.endswith(".pt")
            ])
            self.paths.extend(files)

        if not self.paths:
            raise RuntimeError(
                f"No cached feature files (*.pt) found under {cache_root}. "
                f"Run precompute_features.py first."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        feats = torch.load(self.paths[idx], map_location="cpu", weights_only=True)
        return feats  # (L, N, C)  fp16


# =============================================================================
# TestDataset  —  all test images + ground-truth masks
# =============================================================================

class TestDataset(Dataset):
    """
    MVTec / Omni-AD test set for a single category.

    Directory layout expected::

        <root>/<category>/test/<defect_type>/*.png
        <root>/<category>/ground_truth/<defect_type>/*.png   (or *_mask.png)

    ``good`` images → label=0, all-zero mask.
    Other types     → label=1, mask from ground_truth/<type>/<fname> (or *_mask).
    """

    def __init__(self, root: str, category: str, img_size: int = 448):
        self.root = root
        self.category = category
        self.img_size = img_size

        # ---- image transform ------------------------------------------------
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # ---- mask transform  (nearest-neighbour, no normalisation) ----------
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # ---- collect paths & labels -----------------------------------------
        test_dir = os.path.join(root, category, "test")
        gt_dir = os.path.join(root, category, "ground_truth")

        self.paths: List[str] = []
        self.labels: List[int] = []          # 0 = normal,  1 = anomaly
        self.mask_paths: List[Optional[str]] = []
        self.defect_types: List[str] = []

        for dtype in sorted(os.listdir(test_dir)):
            dpath = os.path.join(test_dir, dtype)
            if not os.path.isdir(dpath):
                continue
            is_good = (dtype == "good")
            for fname in sorted(os.listdir(dpath)):
                if not _is_image(fname):
                    continue
                img_path = os.path.join(dpath, fname)
                self.paths.append(img_path)
                self.labels.append(0 if is_good else 1)
                self.defect_types.append(dtype)

                if is_good:
                    self.mask_paths.append(None)
                else:
                    base, _ext = os.path.splitext(fname)
                    # try MVTec-style _mask suffix first, then raw filename
                    mask_path = os.path.join(gt_dir, dtype, base + "_mask.png")
                    if not os.path.exists(mask_path):
                        mask_path = os.path.join(gt_dir, dtype, fname)
                    self.mask_paths.append(
                        mask_path if os.path.exists(mask_path) else None
                    )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        img_tensor = self.img_transform(img)                         # (3, H, W)

        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            mask_tensor = self.mask_transform(mask)                  # (1, H, W)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros(1, self.img_size, self.img_size)

        return {
            "image": img_tensor,          # (3, H, W)
            "mask": mask_tensor,          # (1, H, W)
            "label": self.labels[idx],    # int
            "path": self.paths[idx],
            "defect_type": self.defect_types[idx],
        }


# =============================================================================
# Shape self-check
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("dataset.py  self-check")
    print("=" * 60)

    B, H, W = 4, 448, 448

    for ds_name in ["mvtad", "omniad"]:
        root = DATASET_ROOTS[ds_name]
        categories = DATASET_CATEGORIES[ds_name]
        print(f"\n--- {ds_name}: {len(categories)} categories, root={root} ---")
        print(f"  exists: {os.path.isdir(root)}")
        if os.path.isdir(root):
            available = [c for c in categories
                         if os.path.isdir(os.path.join(root, c))]
            print(f"  available: {len(available)}/{len(categories)}  "
                  f"({', '.join(available[:5])}...)")
            if available:
                c = available[0]
                # TrainDataset
                train_ds = TrainDataset(root, c)
                img = train_ds[0]
                print(f"  TrainDataset[{c}]: {len(train_ds)} images, "
                      f"shape={tuple(img.shape)}  (C, H, W)")
                assert img.shape == (3, H, W)

                # TestDataset
                test_ds = TestDataset(root, c)
                sample = test_ds[0]
                print(f"  TestDataset [{c}]: {len(test_ds)} images")
                print(f"    image: {tuple(sample['image'].shape)}  (C, H, W)")
                print(f"    mask:  {tuple(sample['mask'].shape)}  (1, H, W)")
                print(f"    label: {sample['label']}, defect: {sample['defect_type']}")
                assert sample["image"].shape == (3, H, W)
                assert sample["mask"].shape == (1, H, W)
        else:
            print("  (directory not found — skipping)")

    print("\n[OK] dataset.py self-check passed.")
