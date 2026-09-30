"""
Training script for DinoAnomaly on MVTec AD  (single category).

Two-phase training (Dinomaly convention):

  Phase 1 (epochs 1 – warmup_simple_epochs):
      ``reconstruction_loss_simple`` — no gradient masking.
      Lets the model settle before introducing hard-sample selection.

  Phase 2 (remaining epochs):
      ``reconstruction_loss`` with hard-sample gradient masking
      (Dinomaly ``global_cosine_hm_percent``, p=0.9, factor=0.1).

LR schedule: linear warmup → cosine decay  (matches Dinomaly
``WarmCosineScheduler`` in ``utils.py:775-792``).

Usage::

    python train.py --data-path /data/mvtec --category bottle
                    --epochs 200 --batch-size 16 --lr 2e-3
                    --use-inp-fusion --lambda-cohere 0.2

    # Ablation: no INP fusion
    python train.py --data-path /data/mvtec --category bottle --no-inp-fusion
"""

import argparse
import csv
import math
import os
import sys
from datetime import datetime
from typing import Dict

# ── immediate startup signal (before any heavy imports) ──────────────────
print("=" * 55, flush=True)
print("  DinoAnomaly Training", flush=True)
print("=" * 55, flush=True)

# Force unbuffered stdout for VS Code / non-TTY terminals.
# Use the lower-level approach (fd reopen) which is more robust than
# .reconfigure() across Python 3.7–3.12.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass  # ignore if already closed / not supported

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── CUDA performance flags ────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True        # 输入尺寸固定时自动选最快卷积算法
torch.backends.cuda.matmul.allow_tf32 = True # TF32 矩阵乘（Ampere+，约 2× 吞吐）
torch.backends.cudnn.allow_tf32 = True       # TF32 卷积

# chjrx imports  —  dataset/losses MUST come before model because
# model → encoder.py inserts Dinomaly-master into sys.path[0], which
# would shadow chjrx/dataset.py with Dinomaly-master/dataset.py.
from dataset import TrainDataset, MultiCategoryTrainDataset, CachedFeatureDataset, DATASET_ROOTS, DATASET_CATEGORIES
from losses import reconstruction_loss, reconstruction_loss_simple
from model import build_dino_anomaly


# =============================================================================
# LR Scheduler  —  linear warmup + cosine decay
# =============================================================================

class WarmupCosineScheduler:
    """
    Linear warmup from 0 → ``base_lr`` over ``warmup_steps``, then cosine
    decay to ``min_lr``.

    Matches Dinomaly ``WarmCosineScheduler`` (utils.py:775-792).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        min_lr: float,
        total_steps: int,
        warmup_steps: int,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        if self._step <= self.warmup_steps:
            lr = self.base_lr * (self._step / max(1, self.warmup_steps))
        else:
            progress = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_last_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# =============================================================================
# Training helpers
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    epoch: int,
    log_interval: int,
    csv_writer,
    use_hard_mask: bool,
    recon_p: float,
    recon_factor: float,
    use_cache: bool = False,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Run one training epoch; return average losses."""

    model.train()
    # fp16 需要 GradScaler 来防止梯度下溢；use_amp=False 时二者均为空操作。
    scaler = torch.amp.GradScaler(enabled=use_amp)
    total_loss = total_recon = total_cohere = 0.0
    n_batches = len(dataloader)

    for step, batch in enumerate(dataloader):
        if use_cache:
            # batch: (B, L, N, C) fp16 cached features — skip frozen encoder
            feats = batch.to(device, dtype=torch.float32)   # (B, L, N, C)
            en_list = [feats[:, i] for i in range(feats.shape[1])]  # List[(B, N, C)]
        else:
            feats = batch.to(device)                        # (B, 3, H, W)

        # ---- forward + loss (AMP autocast) ---------------------------------
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
            if use_cache:
                en, de, g_loss = model.forward_from_features(en_list)
            else:
                en, de, g_loss = model(feats)

            # ---- reconstruction loss ---------------------------------------
            if use_hard_mask:
                loss_recon = reconstruction_loss(en, de, p=recon_p, factor=recon_factor)
            else:
                loss_recon = reconstruction_loss_simple(en, de)

            # ---- total loss ------------------------------------------------
            loss = loss_recon + model.lambda_cohere * g_loss

        # ---- backward + step (fp16: clip 前必须先 unscale_) ---------------
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=0.1,
        )
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()

        # ---- accumulate ----------------------------------------------------
        total_loss   += loss.item()
        total_recon  += loss_recon.item()
        total_cohere += g_loss.item()

        # ---- console + CSV -------------------------------------------------
        global_step = (epoch - 1) * n_batches + step + 1
        if (step + 1) % log_interval == 0 or step == 0:
            print(
                f"  Epoch {epoch:3d} | step {step+1:4d}/{n_batches} | "
                f"lr {lr:.2e} | "
                f"loss {loss.item():.4f} | "
                f"recon {loss_recon.item():.4f} | "
                f"cohere {g_loss.item():.4f}",
                flush=True,
            )
        else:
            # progress dot between log lines (so user knows it's alive)
            print(".", end="", flush=True)
        csv_writer.writerow({
            "epoch": epoch,
            "step": global_step,
            "lr": f"{lr:.6e}",
            "loss_total": f"{loss.item():.6f}",
            "loss_recon": f"{loss_recon.item():.6f}",
            "loss_cohere": f"{g_loss.item():.6f}",
        })

    return {
        "loss_total":  total_loss  / n_batches,
        "loss_recon":  total_recon  / n_batches,
        "loss_cohere": total_cohere / n_batches,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    config: dict,
    save_optimizer: bool = True,
):
    """Save model, (optionally optimizer), and training config.

    ``save_optimizer=False`` writes a much smaller file (model weights only,
    no AdamW momentum/variance state) — used for ``best.pt``, which only needs
    to be loadable for evaluation / inference, not for resuming training.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    if save_optimizer:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)


# =============================================================================
# CLI + main
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train DinoAnomaly on MVTec AD / Omni-AD-30 (single category)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- data --------------------------------------------------------------
    p.add_argument("--dataset", type=str, default="omniad",
                   choices=["omniad"],
                   help="Dataset name: 'omniad' (only Omni-AD-30 is connected).  "
                        "Uses the same layout as MVTec AD.")
    p.add_argument("--data-path", type=str, default=None,
                   help="Root directory override.  If not set, uses the "
                        "pre-configured path for --dataset.")
    p.add_argument("--category", type=str, default="all",
                   help="Dataset category (e.g. 'chip1') or 'all' for "
                        "multi-category joint training (default: 'all').")
    p.add_argument("--img-size", type=int, default=448,
                   help="Square input size (default: 448).  Must be divisible "
                        "by patch_size=14;  num_tokens = (img_size/14)^2.")
    # ---- training hyper-params ---------------------------------------------
    p.add_argument("--epochs", type=int, default=200,
                   help="Total epochs (default: 200).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size (default: 8).")
    p.add_argument("--lr", type=float, default=None,
                   help="Peak learning rate.  Default: auto-scaled as "
                        "2e-3 × (batch_size / 16) — Dinomaly's 2e-3 is tuned "
                        "for bs=16.  Pass an explicit value to override.")
    p.add_argument("--min-lr", type=float, default=2e-4,
                   help="Minimum LR after cosine decay (default: 2e-4, "
                        "matches Dinomaly).")
    p.add_argument("--warmup-epochs", type=float, default=1.0,
                   help="LR linear-warmup duration in epochs (default: 1.0).")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay (default: 1e-4, matches Dinomaly).")
    p.add_argument("--warmup-simple-epochs", type=int, default=5,
                   help="Epochs with simple recon loss before switching to "
                        "hard-sample masking (default: 5).")
    # ---- loss --------------------------------------------------------------
    p.add_argument("--recon-p", type=float, default=0.9,
                   help="Fraction of easy positions to mask (default: 0.9, "
                        "matches Dinomaly).")
    p.add_argument("--recon-factor", type=float, default=0.1,
                   help="Gradient scale for masked positions (default: 0.1, "
                        "matches Dinomaly).")
    # ---- model -------------------------------------------------------------
    p.add_argument("--use-inp-fusion", action="store_true", default=True,
                   help="Enable INP cosine-sim fusion (default: ON).")
    p.add_argument("--no-inp-fusion", dest="use_inp_fusion",
                   action="store_false",
                   help="Disable INP fusion for ablation.")
    p.add_argument("--lambda-cohere", type=float, default=0.2,
                   help="Weight for INP coherence loss (default: 0.2).")
    # ---- misc --------------------------------------------------------------
    p.add_argument("--log-interval", type=int, default=10,
                   help="Print loss every N steps (default: 10).")
    p.add_argument("--output-dir", type=str, default="./checkpoints",
                   help="Checkpoint output directory (default: ./checkpoints).")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save a full checkpoint (with optimizer, for resume) "
                        "every N epochs (default: 10).  Set 0 to disable "
                        "periodic full checkpoints.")
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader workers (default: 8).  "
                        "Linux recommends 4~8, no more than the CPU core count; "
                        "Windows requires 0 — spawn-based multiprocessing "
                        "deadlocks when workers re-import the heavy DINOv2 module.")
    p.add_argument("--use-cache", action="store_true",
                   help="Train from pre-computed encoder features (CachedFeatureDataset) "
                        "instead of raw images — skips the frozen encoder forward. "
                        "Requires running precompute_features.py first.")
    p.add_argument("--cache-dir", type=str, default="./cache",
                   help="Cache directory for pre-computed features (default: ./cache).")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device: 'cuda' or 'cpu' (default: cuda).")
    p.add_argument("--amp", dest="use_amp", action="store_true", default=True,
                   help="Enable AMP mixed precision (fp16 autocast + GradScaler). "
                        "Default ON for CUDA — reduces VRAM and speeds up matmuls. "
                        "Use --no-amp to disable.")
    p.add_argument("--no-amp", dest="use_amp", action="store_false",
                   help="Disable AMP (run in pure fp32).")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed (default: 1, matches Dinomaly).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ---- LR auto-scaling: Dinomaly's 2e-3 is tuned for bs=16 --------------
    if args.lr is None:
        args.lr = 2e-3 * (args.batch_size / 16)
        print(f"LR auto-scaled to {args.lr:.2e} "
              f"(= 2e-3 × {args.batch_size}/16)", flush=True)

    # ---- reproducibility ---------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seed: {args.seed}", flush=True)

    # ---- dataset -----------------------------------------------------------
    data_root = args.data_path or DATASET_ROOTS.get(args.dataset)
    if data_root is None:
        raise ValueError(f"Unknown dataset '{args.dataset}'; provide --data-path "
                         f"or use one of {list(DATASET_ROOTS.keys())}")
    categories = DATASET_CATEGORIES.get(args.dataset, [])
    multi_category = (args.category == "all")

    if args.use_cache:
        cat_list = categories if multi_category else [args.category]
        print(f"Cached-feature training: {len(cat_list)} classes "
              f"(cache_dir={args.cache_dir})", flush=True)
        dataset = CachedFeatureDataset(
            cache_root=args.cache_dir,
            categories=cat_list,
        )
    elif multi_category:
        print(f"Multi-category training: {len(categories)} classes", flush=True)
        dataset = MultiCategoryTrainDataset(
            root=data_root,
            categories=categories,
            img_size=args.img_size,
        )
    else:
        dataset = TrainDataset(
            root=data_root,
            category=args.category,
            img_size=args.img_size,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,                                     # Dinomaly convention
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    steps_per_epoch = len(dataloader)
    if args.use_cache:
        src_label = f"cached features @ {args.cache_dir}"
    elif multi_category:
        src_label = "all categories"
    else:
        src_label = f"{data_root}/{args.category}/train/good/"
    print(f"Dataset: {args.dataset}  |  {len(dataset)} normal images  ({src_label})", flush=True)
    print(f"Batches / epoch: {steps_per_epoch}  (batch_size={args.batch_size})", flush=True)

    # ---- auto-tune log interval for small datasets -------------------------
    log_interval = args.log_interval
    if steps_per_epoch < log_interval * 3:
        log_interval = max(1, steps_per_epoch // 3)
        print(f"  (auto log_interval={log_interval} — small dataset)", flush=True)

    # ---- model -------------------------------------------------------------
    num_inp = 6   # same for single & multi-category (prototypes shared across classes)
    print(f"\nBuilding model (num_inp={num_inp})...", flush=True)
    build_config = {
        "encoder_name": "dinov2reg_vit_base_14",
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
        "fuse_layer_encoder": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "fuse_layer_decoder": [[0, 1, 2, 3, 4, 5, 6, 7]],
        "embed_dim": 768,
        "num_heads": 12,
        "num_inp": num_inp,
        "num_inp_blocks": 1,
        "num_decoder_layers": 8,
        "num_decoder_tokens": 1024,
        "bottleneck_hidden_ratio": 4.0,
        "bottleneck_drop": 0.2,
        "inp_mlp_ratio": 4.0,
        "inp_drop": 0.0,
        "inp_attn_drop": 0.0,
        "decoder_mlp_ratio": 4.0,
        "decoder_drop": 0.0,
        "decoder_attn_drop": 0.0,
        "lambda_cohere": args.lambda_cohere,
        "use_inp_fusion": args.use_inp_fusion,
    }
    model = build_dino_anomaly(**build_config)
    model.to(device)
    print(f"Model: DinoAnomaly  "
          f"(INP fusion={'ON' if args.use_inp_fusion else 'OFF'}, "
          f"λ_cohere={args.lambda_cohere})", flush=True)

    # ---- optimizer (trainable parameters ONLY) -----------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_trainable:,} trainable  /  {n_total:,} total  "
          f"({100 * n_trainable / n_total:.1f}%)", flush=True)

    # AdamW with Dinomaly hyper-params (dinomaly_mvtec_uni.py:165-166).
    # Note: Dinomaly uses ``StableAdamW`` (custom variant with gradient RMS
    # clipping); standard AdamW is a close approximation.
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        amsgrad=True,
        eps=1e-8,
        fused=True,          # CUDA 逐参数融合更新，快 10-20%，数值等价
    )

    # ---- LR scheduler ------------------------------------------------------
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=args.lr,
        min_lr=args.min_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    print(f"LR schedule: warmup {warmup_steps} steps  →  "
          f"cosine decay  {args.lr:.1e} → {args.min_lr:.1e}  "
          f"over {total_steps} steps", flush=True)

    # ---- AMP (mixed precision) ---------------------------------------------
    use_amp = args.use_amp and device.type == "cuda"
    print(f"AMP: {'ON (fp16 autocast + GradScaler)' if use_amp else 'OFF (fp32)'}",
          flush=True)

    # ---- output directory --------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fusion_tag = "inp" if args.use_inp_fusion else "noinp"
    cache_tag = "cache" if args.use_cache else "img"
    mode_tag = f"multi{len(categories)}" if multi_category else args.category
    run_name = (
        f"{mode_tag}_{cache_tag}_{fusion_tag}_lcoh{args.lambda_cohere}"
        f"_b{args.batch_size}_ep{args.epochs}_{timestamp}"
    )
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}", flush=True)

    # ---- CSV logger --------------------------------------------------------
    csv_path = os.path.join(out_dir, "train_log.csv")
    csv_fh = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=[
        "epoch", "step", "lr", "loss_total", "loss_recon", "loss_cohere",
    ])
    csv_writer.writeheader()

    # ---- config snapshot ---------------------------------------------------
    config = {k: v for k, v in vars(args).items()}
    config["total_steps"] = total_steps
    config["warmup_steps"] = warmup_steps
    config["steps_per_epoch"] = steps_per_epoch
    config["build_config"] = build_config             # ← 供 eval.py 重建模型

    print(f"\n{'='*65}", flush=True)
    print(f"  Phase 1  (epochs     1 – {args.warmup_simple_epochs}):  "
          f"reconstruction_loss_simple", flush=True)
    print(f"  Phase 2  (epochs {args.warmup_simple_epochs+1} – {args.epochs}):  "
          f"reconstruction_loss  p={args.recon_p}  factor={args.recon_factor}", flush=True)
    print(f"  Gradient clipping: max_norm = 0.1", flush=True)
    print(f"{'='*65}\n", flush=True)

    # ---- train =============================================================
    print(f"Starting training ({args.epochs} epochs, {steps_per_epoch} steps/epoch).  "
          f"First step compiles CUDA kernels — may take 10-30 s.\n", flush=True)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        use_hard_mask = (epoch > args.warmup_simple_epochs)
        phase_tag = "HARD" if use_hard_mask else "SIMPLE"

        print(f"─── Epoch {epoch}/{args.epochs}  [{phase_tag}] "
              f"lr={scheduler.get_last_lr():.2e} ───", flush=True)

        avg = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            csv_writer=csv_writer,
            use_hard_mask=use_hard_mask,
            recon_p=args.recon_p,
            recon_factor=args.recon_factor,
            use_cache=args.use_cache,
            use_amp=use_amp,
        )

        print(f"  >>> Epoch {epoch:3d}  AVG  "
              f"loss={avg['loss_total']:.4f}  "
              f"recon={avg['loss_recon']:.4f}  "
              f"cohere={avg['loss_cohere']:.4f}\n", flush=True)

        # ---- best model (every epoch, on new-low loss) ---------------------
        # Weights only (no optimizer) — half the size, for eval/inference.
        if avg["loss_total"] < best_loss:
            best_loss = avg["loss_total"]
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(out_dir, "best.pt"),
                            config, save_optimizer=False)
            print(f"  >>> New best loss {best_loss:.4f} → saved best.pt",
                  flush=True)

        # ---- periodic full checkpoint (for resume) -------------------------
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch,
                            os.path.join(out_dir, f"epoch_{epoch:03d}.pt"),
                            config, save_optimizer=True)

    # ---- final model -------------------------------------------------------
    final_path = os.path.join(out_dir, "final_model.pt")
    save_checkpoint(model, optimizer, args.epochs, final_path, config)
    csv_fh.close()

    print(f"Training complete.  Final model : {final_path}", flush=True)
    print(f"                    Training log: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
