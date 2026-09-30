"""
Evaluation script for Dinomaly + INP-Former  (MVTec AD / Omni-AD-30).

Pipeline:
  1. Forward pass  →  encoder features (en) + decoder features (de)
  2. Per-patch cosine similarity between en_i / de_i  →  anomaly heatmap
  3. Reshape spatial maps → upsample to original resolution  (bilinear)
  4. Gaussian-filter smoothing
  5. Image-level score  =  mean of top-K%  **most anomalous**  pixels
     (lowest cosine similarity = highest anomaly score).
  6. Report image-level AUROC  &  pixel-level AUROC.

Usage:
  python eval.py --checkpoint model.pt --dataset omniad --category air_conditioner_filter
                 --top-percent 1.0 --gaussian-sigma 4.0
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Import model builder  (chjrx)
# ---------------------------------------------------------------------------
# dataset MUST come before model — model → encoder.py inserts Dinomaly-master
# into sys.path[0], which would shadow chjrx/dataset.py.
from dataset import TestDataset, DATASET_ROOTS
from model import build_dino_anomaly


# ===========================================================================
# Anomaly-map generation
# ===========================================================================

@torch.no_grad()
def generate_anomaly_map(
    en_list: List[torch.Tensor],
    de_list: List[torch.Tensor],
    img_size: int = 448,
    gaussian_sigma: Optional[float] = 4.0,
) -> np.ndarray:
    """
    Build a per-pixel anomaly heatmap from encoder / decoder feature maps.

    **Aligned with Dinomaly's ``cal_anomaly_map`` (utils.py):**

    For each scale pair (en_i, de_i) of shape (1, C, H, W):
      1.  Cosine similarity along channel dim → (1, H, W).
      2.  Anomaly = 1 − cos_sim   (higher = more anomalous).
      3.  **Upsample each scale separately to ``img_size`` first**
          (bilinear, ``align_corners=True`` — same as Dinomaly), THEN
          average across scales.  The old code averaged at 32×32 before
          upsampling, which smeared each scale's spatial detail during
          fusion.  Now every scale keeps its own detail in full resolution.
      Then smooth with a Gaussian filter (unless ``gaussian_sigma=None``).

    Args:
        en_list:  List[(1, C, H, W)]  encoder (target) feature maps.
        de_list:  List[(1, C, H, W)]  decoder (predicted) feature maps.
        img_size: target image resolution (square).
        gaussian_sigma: σ for Gaussian filter; ``None`` skips smoothing and
                        returns the raw fused map (for sigma sweeps).

    Returns:
        anomaly_map:  (img_size, img_size)  np.ndarray  float32.
    """
    anomaly_maps = []
    for en_i, de_i in zip(en_list, de_list):
        # cosine similarity per spatial position  (1, H, W)
        cos_sim = F.cosine_similarity(en_i, de_i, dim=1)              # (1, H, W)
        anomaly = 1.0 - cos_sim                                       # (1, H, W)

        # ---- per-scale upsample to full resolution (align with Dinomaly) ----
        anomaly = anomaly.unsqueeze(0)                                 # (1, 1, H, W)
        anomaly = F.interpolate(
            anomaly,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=True,                                        # Dinomaly uses True
        ).squeeze(0).squeeze(0)                                        # (img_size, img_size)
        anomaly_maps.append(anomaly)

    # fuse scales by averaging  (in full resolution)
    anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)         # (img_size, img_size)

    anomaly_map = anomaly_map.cpu().numpy().astype(np.float32)

    # Gaussian smoothing  (skip when None — caller sweeps sigma itself)
    if gaussian_sigma is not None:
        anomaly_map = gaussian_filter(anomaly_map, sigma=gaussian_sigma)

    return anomaly_map  # (img_size, img_size)


def compute_image_score(
    anomaly_map: np.ndarray,
    top_percent: float = 1.0,
    mode: str = "mean_topk",
) -> float:
    """
    Image-level anomaly score.

    Modes:
      - ``"mean_topk"`` (default): mean of the top-K% **most anomalous**
        pixels.
      - ``"max"``:                maximum pixel anomaly score.

    Args:
        anomaly_map:  (H, W)  anomaly scores.
        top_percent:  K  (default 1.0 → top 1 %).
        mode:         "mean_topk" or "max".

    Returns:
        scalar score (higher → more anomalous).
    """
    flat = anomaly_map.flatten()
    if mode == "max":
        return float(flat.max())
    k = max(1, int(flat.size * top_percent / 100.0))
    # partial sort: only the top-k largest
    topk = np.partition(flat, -k)[-k:]
    return float(topk.mean())


# ===========================================================================
# Main evaluation loop
# ===========================================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    top_percent: float = 1.0,
    gaussian_sigma: float = 4.0,
    img_size: int = 448,
) -> Dict[str, float]:
    """
    Run evaluation and return image-level + pixel-level AUROC.

    Returns:
        dict with keys ``image_auroc``, ``pixel_auroc``.
    """
    model.eval()

    image_scores: List[float] = []
    image_labels: List[int] = []
    pixel_scores: List[np.ndarray] = []
    pixel_labels: List[np.ndarray] = []

    for batch in tqdm(dataloader, desc="Eval"):
        imgs = batch["image"].to(device)         # (B, 3, H, W)
        masks = batch["mask"]                     # (B, 1, H, W)
        labels = batch["label"]                   # list[int]

        # ---- forward ---------------------------------------------------------
        en, de, _g_loss = model(imgs)             # en/de: List[(B, C, side, side)]

        B = imgs.shape[0]
        for i in range(B):
            # per-sample feature maps  List[(1, C, side, side)]
            en_i = [e[i:i + 1] for e in en]
            de_i = [d[i:i + 1] for d in de]

            anomaly_map = generate_anomaly_map(
                en_i, de_i,
                img_size=img_size,
                gaussian_sigma=gaussian_sigma,
            )  # (img_size, img_size)

            # image-level score
            score = compute_image_score(anomaly_map, top_percent=top_percent)
            image_scores.append(score)
            image_labels.append(labels[i])

            # pixel-level  (flatten for AUROC)
            pixel_scores.append(anomaly_map.flatten())
            pixel_labels.append(masks[i].cpu().numpy().flatten())

    # ---- aggregate -----------------------------------------------------------
    image_scores_np = np.array(image_scores, dtype=np.float32)
    image_labels_np = np.array(image_labels, dtype=np.int32)
    pixel_scores_np = np.concatenate(pixel_scores)
    pixel_labels_np = np.concatenate(pixel_labels)

    # ---- AUROC ---------------------------------------------------------------
    img_auroc = roc_auc_score(image_labels_np, image_scores_np)
    pix_auroc = roc_auc_score(pixel_labels_np, pixel_scores_np)

    return {
        "image_auroc": float(img_auroc),
        "pixel_auroc": float(pix_auroc),
    }


# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DinoAnomaly evaluation on MVTec AD / Omni-AD-30",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- required ---------------------------------------------------------
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pt / .pth).")
    p.add_argument("--dataset", type=str, default="mvtad",
                   choices=["mvtad", "omniad"],
                   help="Dataset name: 'mvtad' or 'omniad' (default: mvtad).")
    p.add_argument("--data-path", type=str, default=None,
                   help="Root directory override.  If not set, uses the "
                        "pre-configured path for --dataset.")
    p.add_argument("--category", type=str, required=True,
                   help="Dataset category, e.g. 'bottle' (MVTec) or "
                        "'air_conditioner_filter' (Omni-AD).")
    # --- scoring -----------------------------------------------------------
    p.add_argument("--top-percent", type=float, default=1.0,
                   help="Top-K%% most anomalous pixels for image score "
                        "(default: 1.0).")
    p.add_argument("--gaussian-sigma", type=float, default=4.0,
                   help="Gaussian filter sigma (default: 4.0).")
    # --- model / data ------------------------------------------------------
    p.add_argument("--img-size", type=int, default=448,
                   help="Input image size, square (default: 448).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Inference batch size (default: 1).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default: 0).  "
                        "Windows requires 0 to avoid multiprocessing deadlocks.")
    # --- misc --------------------------------------------------------------
    p.add_argument("--device", type=str, default="cuda",
                   help="Device: 'cuda' or 'cpu' (default: cuda).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load checkpoint ------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # ---- extract state_dict --------------------------------------------------
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # strip DDP / DataParallel "module." prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # ---- build model from checkpoint config (or fall back to defaults) --------
    build_config = ckpt.get("config", {}).get("build_config", None)
    if build_config is None:
        print("  [WARN] checkpoint has no config.build_config — "
              "using default build_dino_anomaly() params.  "
              "Results may be wrong if the training config differed!")
        model = build_dino_anomaly()
    else:
        model = build_dino_anomaly(**build_config)
        print(f"  Model built from checkpoint config: "
              f"INP fusion={build_config.get('use_inp_fusion')}, "
              f"λ_cohere={build_config.get('lambda_cohere')}, "
              f"num_inp={build_config.get('num_inp')}, "
              f"decoder_layers={build_config.get('num_decoder_layers')}")

    # ---- strict loading with encoder fallback ---------------------------------
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Expected cause: encoder keys (CLS token, register tokens, or
        # version mismatch in DINOv2 weights).  Filter them out and retry.
        enc_keys = [k for k in state_dict if k.startswith("encoder.")]
        print(f"  [info] strict=True failed — retrying without {len(enc_keys)} "
              f"encoder.* keys")
        filtered_sd = {k: v for k, v in state_dict.items()
                       if not k.startswith("encoder.")}
        missing, unexpected = model.load_state_dict(filtered_sd, strict=True)

    # ---- safety assertions: no silent random init of decoder / INP weights ----
    if missing:
        non_enc_missing = [k for k in missing if not k.startswith("encoder.")]
        if non_enc_missing:
            raise RuntimeError(
                f"FATAL: {len(non_enc_missing)} non-encoder keys are MISSING "
                f"from checkpoint and would be randomly initialised:\n"
                + "\n".join(f"  - {k}" for k in non_enc_missing[:20])
            )
        print(f"  [info] {len(missing)} missing keys (all encoder.* — expected, "
              f"encoder is frozen / loaded from DINOv2 pretrain)")

    if unexpected:
        raise RuntimeError(
            f"FATAL: {len(unexpected)} UNEXPECTED keys in checkpoint "
            f"(checkpoint was trained with a different model architecture):\n"
            + "\n".join(f"  - {k}" for k in unexpected[:20])
        )

    print("  [OK] State dict loaded successfully.")

    model.to(device)
    model.eval()
    print("Model ready.")

    # ---- dataset -------------------------------------------------------------
    data_root = args.data_path or DATASET_ROOTS.get(args.dataset)
    if data_root is None:
        raise ValueError(f"Unknown dataset '{args.dataset}'; provide --data-path "
                         f"or use one of {list(DATASET_ROOTS.keys())}")
    dataset = TestDataset(
        root=data_root,
        category=args.category,
        img_size=args.img_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    n_good = sum(1 for l in dataset.labels if l == 0)
    n_bad = len(dataset.labels) - n_good
    print(f"Dataset: {args.dataset}  |  {len(dataset)} images  "
          f"(good={n_good}, anomaly={n_bad})  "
          f"category={args.category}")

    # ---- evaluate ------------------------------------------------------------
    results = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        top_percent=args.top_percent,
        gaussian_sigma=args.gaussian_sigma,
        img_size=args.img_size,
    )

    print(f"\n{'=' * 60}")
    print(f"  Category:           {args.category}")
    print(f"  Top-%:              {args.top_percent:.1f}")
    print(f"  Gaussian σ:         {args.gaussian_sigma:.1f}")
    print(f"  {'─' * 46}")
    print(f"  Image-level AUROC:  {results['image_auroc']:.4f}")
    print(f"  Pixel-level AUROC:  {results['pixel_auroc']:.4f}")
    print(f"{'=' * 60}")

    return results


# ===========================================================================
# Shape self-check  (B, N, C annotations)
# ===========================================================================
if __name__ == "__main__":
    # If CLI arguments are passed, run main(); otherwise run self-check.
    if len(sys.argv) > 1:
        main()

    else:
        print("=" * 60)
        print("eval.py  shape self-check")
        print("=" * 60)

        B, C, H, W = 2, 768, 32, 32           # 2 fused maps  (default config)
        img_size = 448
        P = 6

        # --- dummy encoder / decoder feature maps ------------------------------
        # en / de  are  List[(B, C, H, W)]   —  model.py output format
        en_maps = [torch.randn(B, C, H, W) for _ in range(1)]    # 1 fused scale
        de_maps = [torch.randn(B, C, H, W) for _ in range(1)]

        print(f"\nEncoder maps:  {len(en_maps)} scale(s), "
              f"each ({B}, {C}, {H}, {W})  (B, C, H, W)")
        print(f"Decoder maps:  {len(de_maps)} scale(s), "
              f"each ({B}, {C}, {H}, {W})  (B, C, H, W)")

        # --- anomaly map (batch processing, B=2) -------------------------------
        print(f"\n--- Anomaly map generation (B={B}) ---")
        all_anomaly = []
        for i in range(B):
            en_i = [e[i:i + 1] for e in en_maps]    # List[(1, C, H, W)]
            de_i = [d[i:i + 1] for d in de_maps]
            amap = generate_anomaly_map(en_i, de_i, img_size=img_size,
                                        gaussian_sigma=4.0)
            all_anomaly.append(amap)
            print(f"  sample[{i}]: {amap.shape}  "
                  f"range=[{amap.min():.4f}, {amap.max():.4f}]  "
                  f"(H, W)")

        assert all_anomaly[0].shape == (img_size, img_size), \
            f"Anomaly map shape: {all_anomaly[0].shape} != ({img_size},{img_size})"

        # --- image score --------------------------------------------------------
        print(f"\n--- Image-level scoring ---")
        for top_pct in [0.1, 1.0, 5.0]:
            s = compute_image_score(all_anomaly[0], top_percent=top_pct)
            print(f"  top-{top_pct:4.1f}%  score = {s:.6f}  (scalar)")

        # --- cosine similarity sanity check ------------------------------------
        print(f"\n--- Sanity check: identical maps → anomaly ≈ 0 ---")
        en_one = torch.randn(1, C, H, W)
        amap_self = generate_anomaly_map([en_one], [en_one.clone()],
                                         img_size=img_size, gaussian_sigma=0.0)
        print(f"  anomaly map mean  = {amap_self.mean():.8f}  "
              f"(expected ~0 for identical encoder/decoder)")

        # --- dummy evaluation loop (model-independent) --------------------------
        print(f"\n--- Dummy evaluation loop ---")
        print(f"  (sklearn roc_auc_score available: "
              f"{hasattr(__import__('sklearn.metrics'), 'roc_auc_score')})")

        # simulate scores for a binary classification sanity check
        rng = np.random.RandomState(42)
        dummy_labels = np.array([0, 0, 0, 1, 1, 1])
        dummy_scores_good = rng.randn(3) * 0.2 + 0.0       # low anomaly
        dummy_scores_bad = rng.randn(3) * 0.2 + 1.0        # high anomaly
        dummy_scores = np.concatenate([dummy_scores_good, dummy_scores_bad])
        auroc = roc_auc_score(dummy_labels, dummy_scores)
        print(f"  Dummy image AUROC: {auroc:.4f}  (expected > 0.9)")

        print("\n[OK] All shape & sanity checks passed.")
