"""
Pre-compute frozen DINOv2 encoder features for all training images.

Runs ``DinoEncoder`` once per image (eval + no_grad + AMP fp16), stacks the
per-layer features ``en_list`` (List of (1, N, C)) into a single fp16 tensor
``(L, N, C)`` and saves it to ``<cache_dir>/<category>/<img_stem>.pt``.

Already-cached images are skipped, so re-runs are cheap and resume-safe.

Disk footprint (fp16, ViT-B/14 @ 448px):
    per image  = 8 layers × 1024 tokens × 768 dim × 2 bytes ≈ 12.0 MB
    Omni-AD-30 = 2252 images ≈ 26.4 GB
    MVTec AD   = 3629 images ≈ 45.6 GB

Usage:
    python precompute_features.py --dataset omniad --category all
    python precompute_features.py --dataset mvtad  --category all
"""

import argparse
import os
import sys

# immediate startup signal
print("=" * 55, flush=True)
print("  DinoAnomaly  feature pre-computation", flush=True)
print("=" * 55, flush=True)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
from PIL import Image
from torchvision import transforms

# dataset before model — model → encoder.py inserts Dinomaly-master into
# sys.path[0], which would shadow chjrx/dataset.py.
from dataset import (
    DATASET_ROOTS,
    DATASET_CATEGORIES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    _is_image,
)
from encoder import DinoEncoder


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pre-compute frozen encoder features for cached training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default="mvtad",
                   choices=["mvtad", "omniad"],
                   help="Dataset name: 'mvtad' or 'omniad' (default: mvtad).")
    p.add_argument("--data-path", type=str, default=None,
                   help="Root directory override.  If not set, uses the "
                        "pre-configured path for --dataset.")
    p.add_argument("--category", type=str, default="all",
                   help="Dataset category (e.g. 'chip1') or 'all' for every "
                        "category (default: 'all').")
    p.add_argument("--img-size", type=int, default=448,
                   help="Square input size (default: 448).  Must match training.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Encoder inference batch size (default: 8).")
    p.add_argument("--cache-dir", type=str, default="./cache",
                   help="Cache output directory (default: ./cache).")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device: 'cuda' or 'cpu' (default: cuda).")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable AMP fp16 autocast (use fp32 compute; still "
                        "stores fp16).")
    return p.parse_args(argv)


def collect_work(root, categories, cache_dir):
    """Return list of (img_path, cache_path) for images not yet cached."""
    work = []
    n_skip = 0
    for cat in categories:
        good_dir = os.path.join(root, cat, "train", "good")
        if not os.path.isdir(good_dir):
            print(f"  [WARN] skipping {cat}: {good_dir} not found", flush=True)
            continue
        cat_cache_dir = os.path.join(cache_dir, cat)
        os.makedirs(cat_cache_dir, exist_ok=True)
        for f in sorted(os.listdir(good_dir)):
            if not _is_image(f):
                continue
            stem = os.path.splitext(f)[0]
            cache_path = os.path.join(cat_cache_dir, stem + ".pt")
            if os.path.exists(cache_path):
                n_skip += 1
                continue
            work.append((os.path.join(good_dir, f), cache_path))
    return work, n_skip


def main(argv=None):
    args = parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  AMP: {'OFF' if args.no_amp else 'fp16'}", flush=True)

    data_root = args.data_path or DATASET_ROOTS.get(args.dataset)
    if data_root is None:
        raise ValueError(f"Unknown dataset '{args.dataset}'; provide --data-path.")

    categories = DATASET_CATEGORIES.get(args.dataset, [])
    if args.category != "all":
        categories = [args.category]

    # ---- collect pending work (skip already-cached) --------------------------
    work, n_skip = collect_work(data_root, categories, args.cache_dir)
    total_imgs = len(work) + n_skip
    print(f"Dataset: {args.dataset}  |  categories: {len(categories)}", flush=True)
    print(f"Images: {total_imgs} total  |  {n_skip} already cached  |  "
          f"{len(work)} to process", flush=True)
    if not work:
        print("Nothing to do — cache is complete.", flush=True)
        return

    # ---- image transform (match TrainDataset) --------------------------------
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ---- frozen encoder ------------------------------------------------------
    print("\nBuilding DinoEncoder (dinov2reg_vit_base_14)...", flush=True)
    encoder = DinoEncoder(
        encoder_name="dinov2reg_vit_base_14",
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        remove_class_token=True,
    )
    encoder.to(device)
    encoder.eval()

    # ---- batch process -------------------------------------------------------
    n = len(work)
    n_done = 0
    use_amp = (device.type == "cuda") and not args.no_amp
    print(f"Pre-computing {n} images (batch_size={args.batch_size})...\n",
          flush=True)

    for i in range(0, n, args.batch_size):
        batch = work[i:i + args.batch_size]

        imgs = []
        for img_path, _cache_path in batch:
            img = Image.open(img_path).convert("RGB")
            imgs.append(transform(img))
        imgs = torch.stack(imgs).to(device)                      # (B, 3, H, W)

        if use_amp:
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                en_list, _final = encoder(imgs)                  # List[(B, N, C)]
        else:
            with torch.no_grad():
                en_list, _final = encoder(imgs)                  # List[(B, N, C)]

        # stack (L, B, N, C) → permute → (B, L, N, C) → fp16 on CPU
        stacked = torch.stack(en_list, dim=0).permute(1, 0, 2, 3).cpu().half()

        for j, (_img_path, cache_path) in enumerate(batch):
            # .clone() gives the slice its OWN compact storage; saving a view
            # directly would serialize the whole batch's backing storage
            # (8x file size → disk blow-up).
            torch.save(stacked[j].clone(), cache_path)           # (L, N, C) fp16

        n_done += len(batch)
        if (i // args.batch_size) % 10 == 0 or n_done == n:
            print(f"  [{n_done}/{n}] cached", flush=True)
        else:
            print(".", end="", flush=True)

    print(f"\nDone. {n_done} new cache files written to {args.cache_dir}",
          flush=True)


if __name__ == "__main__":
    main()
