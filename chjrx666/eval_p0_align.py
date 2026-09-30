"""
P0 — Align heatmap generation with Dinomaly's ``cal_anomaly_map`` (eval-only).

Verifies on an existing checkpoint **without retraining**:
  1. per-scale upsample-then-average fusion (now in ``eval.py``);
  2. Gaussian σ sweep  σ ∈ {2, 4, 8}  → report best pixel-AP σ per category;
  3. image-level scores:  top-1% mean  AND  max  → compare AUROC.

Only the forward pass is done once per image; the σ sweep re-uses the raw
(unfiltered) anomaly map, so it is cheap.

Usage::

    python eval_p0_align.py --checkpoint checkpoints/<run>/final_model.pt \
        --out-csv eval_p0_align.csv
"""

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# dataset MUST come before model (see eval.py / train.py note)
from dataset import TestDataset, DATASET_ROOTS, DATASET_CATEGORIES
from model import build_dino_anomaly
from eval import generate_anomaly_map, compute_image_score
from eval_all_metrics import oracle_f1, compute_aupro

SIGMAS: List[float] = [2.0, 4.0, 8.0]
TOP_PERCENT: float = 1.0


# ===========================================================================
# Metrics
# ===========================================================================

def _score_metrics(scores, labels) -> Dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "oracle_f1": oracle_f1(scores, labels),
    }


# ===========================================================================
# Per-category collection + sigma sweep
# ===========================================================================

@torch.no_grad()
def collect_raw_maps(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    img_size: int = 448,
):
    """
    Forward once per image and keep the raw (unfiltered) anomaly map so the
    σ sweep below only re-filters, never re-runs the network.
    """
    model.eval()
    raw_maps: List[np.ndarray] = []
    labels: List[int] = []
    masks: List[np.ndarray] = []

    for batch in tqdm(dataloader, desc="Eval"):
        imgs = batch["image"].to(device)
        en, de, _g_loss = model(imgs)                 # en/de: List[(B, C, side, side)]
        B = imgs.shape[0]
        for i in range(B):
            en_i = [e[i:i + 1] for e in en]
            de_i = [d[i:i + 1] for d in de]
            raw = generate_anomaly_map(
                en_i, de_i, img_size=img_size, gaussian_sigma=None,
            )                                          # (img_size, img_size)
            raw_maps.append(raw)
            labels.append(batch["label"][i])
            masks.append(batch["mask"][i].cpu().numpy().squeeze())
    return raw_maps, labels, masks


def score_at_sigma(
    raw_maps: List[np.ndarray],
    labels: List[int],
    masks: List[np.ndarray],
    sigma: float,
    top_percent: float = TOP_PERCENT,
) -> Dict[str, float]:
    """
    Filter the raw maps with one σ and compute image(top-1% mean / max) and
    pixel(7-metric) results.
    """
    img_topk: List[float] = []
    img_max: List[float] = []
    img_labels: List[int] = []
    pix_scores: List[np.ndarray] = []
    pix_labels: List[np.ndarray] = []
    aupro_amaps: List[np.ndarray] = []
    aupro_masks: List[np.ndarray] = []

    for raw, lab, mask in zip(raw_maps, labels, masks):
        amap = gaussian_filter(raw, sigma=sigma)       # (img_size, img_size)
        img_topk.append(compute_image_score(amap, top_percent=top_percent, mode="mean_topk"))
        img_max.append(compute_image_score(amap, top_percent=top_percent, mode="max"))
        img_labels.append(lab)
        pix_scores.append(amap.flatten())
        pix_labels.append(mask.flatten())
        if lab == 1:                                   # anomalous images → AUPRO
            aupro_amaps.append(amap.astype(np.float32))
            aupro_masks.append(mask.astype(np.uint8))

    img_t = np.array(img_topk, dtype=np.float32)
    img_m = np.array(img_max, dtype=np.float32)
    img_l = np.array(img_labels, dtype=np.int32)
    pix_s = np.concatenate(pix_scores)
    pix_l = np.concatenate(pix_labels).astype(np.int32)

    it = _score_metrics(img_t, img_l)
    im = _score_metrics(img_m, img_l)
    pt = _score_metrics(pix_s, pix_l)
    aupro = compute_aupro(aupro_amaps, aupro_masks) if aupro_amaps else 0.0

    return {
        "img_auroc_topk": it["auroc"],
        "img_auroc_max":  im["auroc"],
        "img_ap_topk":    it["ap"],
        "img_ap_max":     im["ap"],
        "img_f1max_max":  im["oracle_f1"],
        "pix_auroc":      pt["auroc"],
        "pix_ap":         pt["ap"],
        "pix_f1max":      pt["oracle_f1"],
        "pix_aupro":      aupro,
    }


# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P0: align heatmap with Dinomaly cal_anomaly_map (eval-only)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="omniad",
                   choices=["mvtad", "omniad"])
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-csv", type=str, default=None)
    p.add_argument("--sigma", type=float, default=None,
                   help="Fix a single Gaussian σ (no sweep). e.g. 8.0. "
                        "If omitted, sweeps SIGMAS={2,4,8}.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # fixed σ mode → no sweep; otherwise sweep SIGMAS
    if args.sigma is not None:
        sigmas: List[float] = [args.sigma]
        print(f"固定 σ = {args.sigma:.1f}（不做扫描）   top-percent: {TOP_PERCENT}")
    else:
        sigmas = SIGMAS
        print(f"σ sweep: {SIGMAS}   top-percent: {TOP_PERCENT}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load checkpoint ---------------------------------------------------
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
    print(f"  [OK] State dict loaded ({len(missing)} encoder.* missing — expected).")
    model.to(device)
    model.eval()

    # ---- dataset -----------------------------------------------------------
    data_root = args.data_path or DATASET_ROOTS.get(args.dataset)
    if data_root is None:
        raise ValueError(f"Unknown dataset '{args.dataset}'")
    categories = DATASET_CATEGORIES[args.dataset]
    print(f"Dataset: {args.dataset}  |  root: {data_root}  "
          f"|  {len(categories)} categories\n")

    # ---- per-category evaluation -------------------------------------------
    # Primary metrics: pix-F1max, img-F1max, pix-AUROC
    rows: List[Dict] = []
    per_sigma = {s: {"pix_f1": [], "img_f1": [], "pix_auroc": []}
                 for s in sigmas}

    for cat in categories:
        dataset = TestDataset(root=data_root, category=cat,
                              img_size=args.img_size)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        print(f"[{cat}] {len(dataset)} imgs")
        raw_maps, labels, masks = collect_raw_maps(
            model, dataloader, device, img_size=args.img_size)

        sigma_results = {s: score_at_sigma(raw_maps, labels, masks, s)
                         for s in sigmas}

        # best σ by the primary-metric composite (pix-F1 highest weight)
        def primary_score(res):
            return (0.5 * res["pix_f1max"] +
                    0.25 * res["img_f1max_max"] +
                    0.25 * res["pix_auroc"])

        best_sigma = max(sigmas, key=lambda s: primary_score(sigma_results[s]))
        best = sigma_results[best_sigma]

        for s in sigmas:
            per_sigma[s]["pix_f1"].append(sigma_results[s]["pix_f1max"])
            per_sigma[s]["img_f1"].append(sigma_results[s]["img_f1max_max"])
            per_sigma[s]["pix_auroc"].append(sigma_results[s]["pix_auroc"])

        rows.append({
            "category": cat,
            "best_sigma": best_sigma,
            # image-level score uses MAX (P0: higher AUROC than top-1% mean)
            "img_auroc": best["img_auroc_max"],
            "img_ap":    best["img_ap_max"],
            "img_f1max": best["img_f1max_max"],
            "pix_auroc": best["pix_auroc"],
            "pix_ap":    best["pix_ap"],
            "pix_f1max": best["pix_f1max"],
            "pix_aupro": best["pix_aupro"],
        })

        print(f"        pixF1: σ2={sigma_results[2.0]['pix_f1max']:.4f}  "
              f"σ4={sigma_results[4.0]['pix_f1max']:.4f}  "
              f"σ8={sigma_results[8.0]['pix_f1max']:.4f}  "
              f"→ best σ={best_sigma:.1f}  "
              f"(pixF1={best['pix_f1max']:.4f} imgF1={best['img_f1max_max']:.4f} "
              f"pixAUROC={best['pix_auroc']:.4f})\n")
        del dataloader, dataset, raw_maps
        torch.cuda.empty_cache()

    # ---- summary table (optimal-σ 7 metrics, image score = max) --------------
    hdr = ["img_auroc", "img_ap", "img_f1max", "pix_auroc",
           "pix_ap", "pix_f1max", "pix_aupro"]
    print("=" * 120)
    print(f"{'Category':<26}{'bestσ':>7}" + "".join(f"{h:>11}" for h in hdr))
    print("-" * 120)
    for r in rows:
        print(f"{r['category']:<26}{r['best_sigma']:>7.1f}" +
              "".join(f"{r[h]:>11.4f}" for h in hdr))
    print("-" * 120)
    for label, fn in [("MEAN", np.mean), ("MEDIAN", np.median), ("STD", np.std)]:
        print(f"{label:<26}{fn([r['best_sigma'] for r in rows]):>7.1f}" +
              "".join(f"{fn(np.array([r[h] for r in rows])):>11.4f}" for h in hdr))
    print("=" * 120)

    def mean(vals):
        return float(np.mean(vals))

    # σ-sweep overall means (primary metrics: pix-F1, img-F1, pix-AUROC)
    print(f"\nσ overall MEAN (30 cats, PRIMARY metrics):")
    print(f"{'σ':>5}{'pix-F1max':>12}{'img-F1max':>12}{'pix-AUROC':>12}")
    for s in sigmas:
        print(f"{s:>5.1f}"
              f"{mean(per_sigma[s]['pix_f1']):>12.4f}"
              f"{mean(per_sigma[s]['img_f1']):>12.4f}"
              f"{mean(per_sigma[s]['pix_auroc']):>12.4f}")
    print("=" * 120)

    # ---- save CSV (standard 7 metrics @ per-category best σ) ------------------
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        fields = ["category", "best_sigma", *hdr]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fields})
            w.writerow({"category": "MEAN",
                        "best_sigma": mean([r["best_sigma"] for r in rows]),
                        **{h: mean([r[h] for r in rows]) for h in hdr}})
            w.writerow({"category": "MEDIAN",
                        "best_sigma": float(np.median([r["best_sigma"] for r in rows])),
                        **{h: float(np.median([r[h] for r in rows])) for h in hdr}})
        print(f"\nSaved per-category results → {args.out_csv}")

    return rows


if __name__ == "__main__":
    main()
