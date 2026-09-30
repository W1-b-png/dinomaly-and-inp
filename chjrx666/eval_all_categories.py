"""
Evaluate a trained DinoAnomaly checkpoint across ALL Omni-AD-30 categories
and report per-category + mean image/pixel AUROC.

Reuses the exact scoring pipeline from ``eval.py`` (cosine-similarity heatmap,
top-K% image score, Gaussian smoothing).

Usage::

    python eval_all_categories.py --checkpoint checkpoints/<run>/final_model.pt
"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# dataset MUST come before model (see eval.py / train.py note)
from dataset import TestDataset, DATASET_ROOTS, DATASET_CATEGORIES
from model import build_dino_anomaly
from eval import evaluate


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate checkpoint on all Omni-AD-30 categories",
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pt / .pth).")
    p.add_argument("--dataset", type=str, default="omniad",
                   choices=["mvtad", "omniad"])
    p.add_argument("--data-path", type=str, default=None,
                   help="Root directory override (default: pre-configured).")
    p.add_argument("--top-percent", type=float, default=1.0)
    p.add_argument("--gaussian-sigma", type=float, default=4.0)
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load checkpoint -----------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    build_config = ckpt.get("config", {}).get("build_config", None)
    if build_config is None:
        print("  [WARN] no config.build_config — using defaults.")
        model = build_dino_anomaly()
    else:
        model = build_dino_anomaly(**build_config)
        print(f"  Model: INP fusion={build_config.get('use_inp_fusion')}, "
              f"λ_cohere={build_config.get('lambda_cohere')}, "
              f"num_inp={build_config.get('num_inp')}, "
              f"decoder_layers={build_config.get('num_decoder_layers')}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"FATAL: {len(unexpected)} unexpected keys: "
                           f"{unexpected[:20]}")
    non_enc_missing = [k for k in missing if not k.startswith("encoder.")]
    if non_enc_missing:
        raise RuntimeError(f"FATAL: {len(non_enc_missing)} non-encoder missing "
                           f"keys: {non_enc_missing[:20]}")
    print(f"  [OK] State dict loaded "
          f"({len(missing)} encoder.* missing — expected, frozen encoder).")
    model.to(device)
    model.eval()

    # ---- dataset root & categories -------------------------------------------
    data_root = args.data_path or DATASET_ROOTS.get(args.dataset)
    if data_root is None:
        raise ValueError(f"Unknown dataset '{args.dataset}'")
    categories = DATASET_CATEGORIES[args.dataset]
    print(f"Dataset: {args.dataset}  |  root: {data_root}  "
          f"|  {len(categories)} categories\n")

    # ---- evaluate each category ----------------------------------------------
    rows: List[Dict[str, float]] = []
    for cat in categories:
        dataset = TestDataset(root=data_root, category=cat,
                              img_size=args.img_size)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        n_good = sum(1 for l in dataset.labels if l == 0)
        n_bad = len(dataset.labels) - n_good
        print(f"[{cat}] {len(dataset)} imgs (good={n_good}, anomaly={n_bad})")

        res = evaluate(
            model=model, dataloader=dataloader, device=device,
            top_percent=args.top_percent, gaussian_sigma=args.gaussian_sigma,
            img_size=args.img_size,
        )
        rows.append({"category": cat, **res})
        print(f"        → img-AUROC={res['image_auroc']:.4f}  "
              f"pix-AUROC={res['pixel_auroc']:.4f}\n")

    # ---- summary ---------------------------------------------------------------
    img_aurocs = np.array([r["image_auroc"] for r in rows])
    pix_aurocs = np.array([r["pixel_auroc"] for r in rows])

    print("=" * 66)
    print(f"{'Category':<32}{'img-AUROC':>10}{'pix-AUROC':>12}")
    print("-" * 66)
    for r in rows:
        print(f"{r['category']:<32}{r['image_auroc']:>10.4f}"
              f"{r['pixel_auroc']:>12.4f}")
    print("-" * 66)
    print(f"{'MEAN':<32}{img_aurocs.mean():>10.4f}{pix_aurocs.mean():>12.4f}")
    print(f"{'MEDIAN':<32}{np.median(img_aurocs):>10.4f}"
          f"{np.median(pix_aurocs):>12.4f}")
    print(f"{'STD':<32}{img_aurocs.std():>10.4f}{pix_aurocs.std():>12.4f}")
    print("=" * 66)

    return rows


if __name__ == "__main__":
    main()
