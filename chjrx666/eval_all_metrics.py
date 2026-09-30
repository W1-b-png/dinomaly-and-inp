"""
Evaluate a trained DinoAnomaly checkpoint across all Omni-AD-30 categories
with 7 metrics (image-level & pixel-level):
    AUROC · Average Precision (AP) · Oracle F1 (F1-max) · AUPRO (pixel)

Reuses the exact scoring pipeline from ``eval.py`` (cosine-similarity heatmap,
top-K% image score, Gaussian smoothing).

Usage::

    python eval_all_metrics.py --checkpoint checkpoints/<run>/final_model.pt
"""

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage.measure import label
from tqdm import tqdm

# dataset MUST come before model (see eval.py / train.py note)
from dataset import TestDataset, DATASET_ROOTS, DATASET_CATEGORIES
from model import build_dino_anomaly
from eval import generate_anomaly_map, compute_image_score


# ===========================================================================
# Metrics
# ===========================================================================

def oracle_f1(scores, labels) -> float:
    """
    Oracle F1 (F1-max): the highest F1 achievable by *any* threshold on the
    ground-truth labels.  Also known as the optimal / max F1.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    order = np.argsort(scores)[::-1]           # descending scores
    sorted_labels = labels[order]

    n_pos = int(sorted_labels.sum())
    n_total = len(sorted_labels)
    if n_pos == 0:
        return 0.0
    if n_pos == n_total:
        return 1.0

    best_f1 = 0.0
    tp = 0
    fp = 0
    for lbl in sorted_labels:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        denom = precision + recall
        if denom > 0.0:
            f1 = 2.0 * precision * recall / denom
            if f1 > best_f1:
                best_f1 = f1
    return float(best_f1)


def _score_metrics(scores, labels) -> Dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "oracle_f1": oracle_f1(scores, labels),
    }


def compute_aupro(
    amaps: List[np.ndarray],
    masks: List[np.ndarray],
    num_th: int = 200,
    fpr_max: float = 0.3,
) -> float:
    """
    AUPRO — Area Under the Per-Region Overlap curve (MVTec standard).

    For every anomaly map / GT-mask pair:
      * label connected components of the GT mask (per-region),
      * for each threshold: region overlap = |region ∩ pred| / |region|,
      * PRO(th) = mean overlap over all regions (across all images),
      * AUPRO    = ∫ PRO dFPR over FPR ∈ [0, fpr_max]  (trapezoid).

    Args:
        amaps:  list of (H, W) float32 anomaly maps  (anomalous images only).
        masks:  list of (H, W) bool/int GT masks   (1 = defect), same order.
        num_th: number of thresholds.
        fpr_max: max false-positive rate for the PRO integral.

    Returns:
        aupro: float in [0, 1].
    """
    assert len(amaps) == len(masks) and len(amaps) > 0
    min_score = min(a.min() for a in amaps)
    max_score = max(a.max() for a in amaps)
    if max_score - min_score < 1e-12:
        return 0.0
    ths = np.linspace(min_score, max_score, num_th)

    total_pixels = sum(m.size for m in masks)

    pros: List[float] = []
    fprs: List[float] = []
    for th in ths:
        region_overlaps: List[float] = []
        fp = 0                                   # false-positive pixels
        for amap, mask in zip(amaps, masks):
            binary = amap >= th
            # connected components of the GT defect mask
            comps = label((mask > 0).astype(np.uint8), connectivity=2)
            for comp_id in range(1, comps.max() + 1):
                region = comps == comp_id
                overlap = float((region & binary).sum()) / float(region.sum())
                region_overlaps.append(overlap)
            fp += int((((mask > 0) == False) & binary).sum())   # noqa: E712
        pros.append(float(np.mean(region_overlaps)) if region_overlaps else 0.0)
        fprs.append(fp / total_pixels)

    # 按 FPR 升序排序（原始顺序 th 从 min→max, FPR 从高→低）
    order = np.argsort(fprs)
    fprs_s = np.asarray(fprs, dtype=np.float64)[order]
    pros_s = np.asarray(pros, dtype=np.float64)[order]

    # 只在 FPR ∈ [0, fpr_max] 内积分
    keep = fprs_s <= fpr_max
    fprs_keep = np.clip(fprs_s[keep], 0.0, fpr_max)
    pros_keep = pros_s[keep]
    if fprs_keep.size == 0:
        return 0.0

    # 端点补齐, 使积分区间覆盖 [0, fpr_max]
    if fprs_keep[0] > 0.0:
        fprs_keep = np.concatenate(([0.0], fprs_keep))
        pros_keep = np.concatenate(([pros_keep[0]], pros_keep))
    if fprs_keep[-1] < fpr_max:
        fprs_keep = np.concatenate((fprs_keep, [fpr_max]))
        pros_keep = np.concatenate((pros_keep, [pros_keep[-1]]))

    # trapezoid rule: AUPRO = (1/fpr_max) · ∫₀^{fpr_max} PRO dFPR
    aupro = 0.0
    for i in range(len(fprs_keep) - 1):
        aupro += 0.5 * (pros_keep[i] + pros_keep[i + 1]) * (fprs_keep[i + 1] - fprs_keep[i])
    aupro /= fpr_max
    return float(aupro)


# ===========================================================================
# Full evaluation loop  (collects all scores, returns 6 metrics)
# ===========================================================================

@torch.no_grad()
def evaluate_full(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    top_percent: float = 1.0,
    gaussian_sigma: float = 4.0,
    img_size: int = 448,
) -> Dict[str, float]:
    model.eval()

    image_scores: List[float] = []
    image_labels: List[int] = []
    pixel_scores: List[np.ndarray] = []
    pixel_labels: List[np.ndarray] = []
    aupro_amaps: List[np.ndarray] = []          # per-image anomaly maps (anomalous)
    aupro_masks: List[np.ndarray] = []          # per-image GT masks (anomalous)

    for batch in tqdm(dataloader, desc="Eval"):
        imgs = batch["image"].to(device)         # (B, 3, H, W)
        masks = batch["mask"]                     # (B, 1, H, W)
        labels = batch["label"]                   # list[int]

        en, de, _g_loss = model(imgs)             # en/de: List[(B, C, side, side)]

        B = imgs.shape[0]
        for i in range(B):
            en_i = [e[i:i + 1] for e in en]
            de_i = [d[i:i + 1] for d in de]

            anomaly_map = generate_anomaly_map(
                en_i, de_i,
                img_size=img_size,
                gaussian_sigma=gaussian_sigma,
            )

            image_scores.append(compute_image_score(anomaly_map, top_percent=top_percent))
            image_labels.append(labels[i])

            pixel_scores.append(anomaly_map.flatten())
            pixel_labels.append(masks[i].cpu().numpy().flatten())

            # keep per-image map+mask for AUPRO (anomalous images only)
            if labels[i] == 1:
                aupro_amaps.append(anomaly_map.astype(np.float32))
                aupro_masks.append(masks[i].cpu().numpy().squeeze().astype(np.uint8))

    img_s = np.array(image_scores, dtype=np.float32)
    img_l = np.array(image_labels, dtype=np.int32)
    pix_s = np.concatenate(pixel_scores)
    pix_l = np.concatenate(pixel_labels).astype(np.int32)

    img_m = _score_metrics(img_s, img_l)
    pix_m = _score_metrics(pix_s, pix_l)

    aupro = compute_aupro(aupro_amaps, aupro_masks) if aupro_amaps else 0.0

    return {
        "image_auroc":  img_m["auroc"],
        "image_ap":     img_m["ap"],
        "image_f1max":  img_m["oracle_f1"],
        "pixel_auroc":  pix_m["auroc"],
        "pixel_ap":     pix_m["ap"],
        "pixel_f1max":  pix_m["oracle_f1"],
        "pixel_aupro":  aupro,
    }


# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="7-metric evaluation on all categories")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="omniad", choices=["mvtad", "omniad"])
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--top-percent", type=float, default=1.0)
    p.add_argument("--gaussian-sigma", type=float, default=4.0)
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-csv", type=str, default=None,
                   help="Path to save per-category results as CSV (optional).")
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

        res = evaluate_full(
            model=model, dataloader=dataloader, device=device,
            top_percent=args.top_percent, gaussian_sigma=args.gaussian_sigma,
            img_size=args.img_size,
        )
        rows.append({"category": cat, **res})
        print(f"        → img AUROC={res['image_auroc']:.4f} AP={res['image_ap']:.4f} "
              f"F1={res['image_f1max']:.4f} | "
              f"pix AUROC={res['pixel_auroc']:.4f} AP={res['pixel_ap']:.4f} "
              f"F1={res['pixel_f1max']:.4f} AUPRO={res['pixel_aupro']:.4f}\n")

        # free per-category memory
        del dataloader, dataset
        torch.cuda.empty_cache()

    # ---- summary ---------------------------------------------------------------
    def summarize(key: str):
        vals = np.array([r[key] for r in rows])
        return vals.mean(), np.median(vals), vals.std()

    hdr = ["image_auroc", "image_ap", "image_f1max",
           "pixel_auroc", "pixel_ap", "pixel_f1max", "pixel_aupro"]
    print("=" * 120)
    print(f"{'Category':<30}" + "".join(f"{h:>13}" for h in hdr))
    print("-" * 120)
    for r in rows:
        print(f"{r['category']:<30}" + "".join(f"{r[h]:>13.4f}" for h in hdr))
    print("-" * 120)
    for label, fn in [("MEAN", np.mean), ("MEDIAN", np.median), ("STD", np.std)]:
        line = f"{label:<30}" + "".join(
            f"{fn(np.array([r[h] for r in rows])):>13.4f}" for h in hdr)
        print(line)
    print("=" * 120)

    # ---- save CSV -------------------------------------------------------------
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["category", *hdr])
            w.writeheader()
            for r in rows:
                w.writerow(r)
            w.writerow({"category": "MEAN",
                        **{h: float(np.mean([r[h] for r in rows])) for h in hdr}})
            w.writerow({"category": "MEDIAN",
                        **{h: float(np.median([r[h] for r in rows])) for h in hdr}})
        print(f"\nSaved per-category results → {args.out_csv}")

    return rows


if __name__ == "__main__":
    main()
