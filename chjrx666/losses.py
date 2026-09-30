"""
Loss functions for Dinomaly + INP-Former training.

  inp_coherence_loss      — INP Coherence Loss  (from INP-Former paper)
  reconstruction_loss     — multi-scale cosine-distance reconstruction loss
                            (from Dinomaly paper, ``global_cosine_hm_percent``)

All functions operate on **patch-token-level** features (B, N, C) or
**spatial** feature maps (B, C, H, W).
"""

from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Gradient-modification helpers  (from Dinomaly utils.py)
# ===========================================================================

def _modify_grad_v2(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Scale gradient element-wise by ``factor`` (used in hook)."""
    factor = factor.to(x.dtype).expand_as(x)   # cast to grad dtype (fp16-safe under AMP)
    x *= factor
    return x


# ===========================================================================
# INP Coherence Loss  (INP-Former paper)
# ===========================================================================

def inp_coherence_loss(
    patch_tokens: torch.Tensor,
    inp_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    INP Coherence Loss — encourages every patch token to be close to at least
    one INP prototype.

    Ported exactly from ``INP_Former.gather_loss()``
    (INP-Former-main/models/uad.py:37).

    For each patch token q ∈ patch_tokens, compute the cosine distance to
    every INP prototype k ∈ inp_tokens, take the **minimum** (closest INP),
    and average across all patches:

        L_cohere = mean_n [ min_p ( 1 - cos_sim(q_n, k_p) ) ]

    Args:
        patch_tokens:  (B, N, C)  feature tokens to be "covered" by INPs.
        inp_tokens:    (B, P, C)  INP prototype tokens.

    Returns:
        loss:          scalar — mean of per-patch minimum cosine distances.
        distribution:  (B, N, P)  full pairwise cosine-distance matrix.
        cluster_index: (B, N)     index of closest INP per patch (for analysis).
    """
    # pairwise cosine distance  (B, N, P)
    #   cosine_similarity(x, y, dim=C) with unsqueeze for broadcasting:
    #   patch_tokens  (B, N, 1, C)  vs  inp_tokens  (B, 1, P, C)
    #   →  (B, N, P)  pairwise cos-sim, then 1 - cos_sim
    distribution = 1.0 - F.cosine_similarity(
        patch_tokens.unsqueeze(2),    # (B, N, 1, C)
        inp_tokens.unsqueeze(1),     # (B, 1, P, C)
        dim=-1,
    )                                # (B, N, P)

    # for each patch, find the closest INP
    distance, cluster_index = torch.min(distribution, dim=2)
    #   distance:      (B, N)  — cosine distance to nearest INP
    #   cluster_index: (B, N)  — index of that INP

    loss = distance.mean()

    return loss, distribution, cluster_index


# ===========================================================================
# Reconstruction Loss  (Dinomaly paper, ``global_cosine_hm_percent``)
# ===========================================================================

def reconstruction_loss(
    en_list: List[torch.Tensor],
    de_list: List[torch.Tensor],
    p: float = 0.9,
    factor: float = 0.0,
) -> torch.Tensor:
    """
    Multi-scale cosine-distance reconstruction loss with hard-sample gradient
    masking.

    Ported from Dinomaly ``utils.global_cosine_hm_percent`` (line 70-89).

    For each scale pair (en_i, de_i):
      1. Compute per-position cosine distance  (B, 1, H, W).
      2. Find the top (1-p) percentile threshold.
      3. Backward hook::

             hard positions  (point_dist >= thresh)  →  gradient × 1.0
             easy positions  (point_dist <  thresh)  →  gradient × factor

         i.e. only the hardest (1-p)% of positions receive full gradients;
         the remaining p% are scaled by ``factor`` (default 0.0 = masked).

    This matches the original Dinomaly ``modify_grad`` semantics:
    ``x[inds] *= factor`` where ``inds`` = easy positions, leaving hard
    positions untouched.

    Args:
        en_list:  List[(B, C, H, W)]  encoder (target) feature maps.
        de_list:  List[(B, C, H, W)]  decoder (predicted) feature maps.
        p:        fraction of positions whose gradients are scaled by
                  ``factor`` (0<p<1).  p=0.9 means only top 10% hardest
                  positions get full gradients.
        factor:   gradient scale for masked (easy) positions
                  (0.0 = zero gradient).

    Returns:
        loss:  scalar  (averaged across scales).
    """
    cos_loss = F.cosine_similarity                                   # alias
    loss = 0.0

    for en_i, de_i in zip(en_list, de_list):
        a_ = en_i.detach()                                           # target (no grad)
        b_ = de_i                                                    # prediction

        # per-position cosine distance  (B, 1, H, W)  for broadcast with grad
        with torch.no_grad():
            point_dist = 1.0 - cos_loss(a_, b_).unsqueeze(1)

        # hard-sample threshold: top (1-p) proportion of largest distances
        thresh = torch.topk(
            point_dist.reshape(-1),
            k=int(point_dist.numel() * (1.0 - p)),
        )[0][-1]

        # cosine loss (mean over all positions)
        loss += torch.mean(
            1.0 - cos_loss(
                a_.reshape(a_.shape[0], -1),
                b_.reshape(b_.shape[0], -1),
            )
        )

        # backward hook: hard=1, easy=factor (matches original modify_grad)
        #   hard: point_dist >= thresh  →  1.0 + (0.0)*factor = 1.0
        #   easy: point_dist <  thresh  →  0.0 + (1.0)*factor = factor
        factor_map = (
            (point_dist >= thresh).float()
            + (point_dist < thresh).float() * factor
        )
        b_.register_hook(
            partial(_modify_grad_v2, factor=factor_map)
        )

    loss = loss / len(en_list)
    return loss


def reconstruction_loss_simple(
    en_list: List[torch.Tensor],
    de_list: List[torch.Tensor],
) -> torch.Tensor:
    """
    Simplified cosine-distance reconstruction loss (no hard-sample masking).

    Useful for initial debugging before enabling ``reconstruction_loss``.
    """
    loss = 0.0
    for en_i, de_i in zip(en_list, de_list):
        loss += torch.mean(
            1.0 - F.cosine_similarity(
                en_i.reshape(en_i.shape[0], -1),
                de_i.reshape(de_i.shape[0], -1),
            )
        )
    return loss / len(en_list)


# ===========================================================================
# Combined training loss
# ===========================================================================

def total_loss(
    en_list: List[torch.Tensor],
    de_list: List[torch.Tensor],
    inp_tokens: torch.Tensor,
    patch_tokens: torch.Tensor,
    lambda_cohere: float = 0.2,        # INP-Former default
    recon_p: float = 0.9,
    recon_factor: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined training loss = reconstruction + λ * coherence.

    Args:
        en_list, de_list:  feature maps for reconstruction (List of B,C,H,W).
        inp_tokens:        (B, P, C)  INP prototypes.
        patch_tokens:      (B, N, C)  patch features from encoder final layer.
        lambda_cohere:     weight for INP coherence loss.
        recon_p, recon_factor:  parameters for ``reconstruction_loss``.

    Returns:
        total:  scalar loss.
        info:   dict with individual loss components for logging.
    """
    loss_recon = reconstruction_loss(en_list, de_list, p=recon_p, factor=recon_factor)
    loss_cohere, _, _ = inp_coherence_loss(patch_tokens, inp_tokens)
    total = loss_recon + lambda_cohere * loss_cohere
    info = {
        "loss_total": total.item(),
        "loss_recon": loss_recon.item(),
        "loss_cohere": loss_cohere.item(),
    }
    return total, info


# ===========================================================================
# Shape self-check
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("losses.py  shape self-check")
    print("=" * 60)

    B, N, C = 2, 1024, 768
    P = 6
    H = W = 32   # 448 / 14

    # --- INP Coherence Loss -------------------------------------------------
    print("\n--- inp_coherence_loss ---")
    dummy_patch = torch.randn(B, N, C)
    dummy_inp = torch.randn(B, P, C)

    loss_cohere, dist, idx = inp_coherence_loss(dummy_patch, dummy_inp)

    print(f"  patch_tokens  : ({B}, {N}, {C})")
    print(f"  inp_tokens    : ({B}, {P}, {C})")
    print(f"  loss          : {loss_cohere.item():.4f}  (scalar)")
    print(f"  distribution  : {tuple(dist.shape)}  (B, N, P)")
    print(f"  cluster_index : {tuple(idx.shape)}  (B, N)")
    assert loss_cohere.ndim == 0, "Loss must be scalar"
    assert dist.shape == (B, N, P)
    assert idx.shape == (B, N)

    # Verify: distance is in [0, 2] (cosine distance range)
    assert 0.0 <= dist.min().item() <= 2.0
    assert 0.0 <= dist.max().item() <= 2.0
    print("  [OK] Distance range valid [0, 2].")

    # --- Reconstruction Loss (simple) ---------------------------------------
    print("\n--- reconstruction_loss_simple ---")
    en_maps = [torch.randn(B, C, H, W) for _ in range(2)]  # 2 fused scales
    de_maps = [torch.randn(B, C, H, W) for _ in range(2)]

    loss_recon = reconstruction_loss_simple(en_maps, de_maps)
    print(f"  en[0], de[0]  : ({B}, {C}, {H}, {W})")
    print(f"  loss          : {loss_recon.item():.4f}  (scalar)")
    assert loss_recon.ndim == 0

    # --- Reconstruction Loss (with hard-sample masking) ----------------------
    print("\n--- reconstruction_loss (with hard-sample masking) ---")
    de_params = [torch.nn.Parameter(d.clone()) for d in de_maps]
    loss_recon2 = reconstruction_loss(en_maps, de_params, p=0.9, factor=0.0)
    print(f"  loss          : {loss_recon2.item():.4f}  (scalar)")
    assert loss_recon2.ndim == 0

    # Gradient check for reconstruction_loss — verifies:
    #   (a) gradient is NOT all-zero  (catches the factor=0.0 bug)
    #   (b) hardest 10% positions → non-zero grad;  easiest 90% → zero grad
    print("  backward...")
    loss_recon2.backward()
    for i, dp in enumerate(de_params):
        assert dp.grad is not None, f"de_params[{i}] should receive gradients!"
        g_sum = dp.grad.abs().sum().item()
        assert g_sum > 0.0, (
            f"de_params[{i}] gradient is ALL ZEROS (sum={g_sum})! "
            f"Possible cause: factor_map was all zeros in the backward hook."
        )
    print(f"  de_params[0] grad abs-sum = {de_params[0].grad.abs().sum().item():.4f}  (> 0 OK)")

    # --- Verify mask direction: hard (top-10%) ≠ 0, easy (bottom-90%) = 0 ----
    # Re-build with p=0.9, factor=0.0 and a deterministic de input so we can
    # check per-position gradient masks.
    print("\n--- Verify hard/easy gradient mask direction ---")
    torch.manual_seed(42)
    en_test = [torch.randn(1, C, H, W)]
    de_test = [torch.nn.Parameter(torch.randn(1, C, H, W))]

    # Re-run forward so we can inspect the hook factor_map
    cos = F.cosine_similarity
    with torch.no_grad():
        pd_test = 1.0 - cos(en_test[0].detach(), de_test[0]).unsqueeze(1)  # (1,1,H,W)
    thresh_test = torch.topk(
        pd_test.reshape(-1), k=int(pd_test.numel() * (1.0 - 0.9))
    )[0][-1]

    mask_hard = (pd_test >= thresh_test)   # top 10%  — should get grad
    mask_easy = (pd_test <  thresh_test)   # bottom 90% — should be zero

    n_hard = mask_hard.sum().item()
    n_easy = mask_easy.sum().item()
    print(f"  hard positions (>= thresh): {n_hard}  ({100*n_hard/pd_test.numel():.1f}%)")
    print(f"  easy positions (<  thresh): {n_easy}  ({100*n_easy/pd_test.numel():.1f}%)")

    # Forward + backward with known thresholds
    loss_test = reconstruction_loss(en_test, de_test, p=0.9, factor=0.0)
    de_test[0].retain_grad()
    loss_test.backward()

    g = de_test[0].grad                          # (1, C, H, W)
    assert g is not None
    g_abs_per_pos = g.abs().sum(dim=1)             # (1, H, W)  aggregate over channels

    # mask_hard/easy are (1, 1, H, W) → squeeze to (H, W) for boolean indexing
    grad_hard_sum = g_abs_per_pos[:, mask_hard[0, 0]].sum().item()
    grad_easy_sum = g_abs_per_pos[:, mask_easy[0, 0]].sum().item()

    print(f"  grad abs-sum @ hard positions: {grad_hard_sum:.6f}")
    print(f"  grad abs-sum @ easy positions: {grad_easy_sum:.6f}")

    assert grad_hard_sum > 0.0, (
        f"HARD positions should have NON-ZERO gradient (got {grad_hard_sum})! "
        f"Mask direction is WRONG — check factor_map in reconstruction_loss."
    )
    assert grad_easy_sum == 0.0, (
        f"EASY positions should have ZERO gradient (got {grad_easy_sum})! "
        f"Mask direction is WRONG — check factor_map in reconstruction_loss."
    )
    print("  [OK] Mask direction verified: hard≠0, easy=0")

    print("  [OK] Gradients flow to de_params through reconstruction_loss.")

    # --- Combined loss ------------------------------------------------------
    print("\n--- total_loss ---")
    de_params2 = [torch.nn.Parameter(d.clone()) for d in de_maps]
    total, info = total_loss(en_maps, de_params2, dummy_inp, dummy_patch)
    print(f"  loss_total   : {info['loss_total']:.4f}")
    print(f"  loss_recon   : {info['loss_recon']:.4f}")
    print(f"  loss_cohere  : {info['loss_cohere']:.4f}")
    assert total.ndim == 0

    # --- Gradient check for coherence loss -----------------------------------
    print("\n--- Gradient check: inp_coherence_loss ---")
    inp_param = nn.Parameter(dummy_inp.clone())
    patch_param = nn.Parameter(dummy_patch.clone())
    loss_g, _, _ = inp_coherence_loss(patch_param, inp_param)
    loss_g.backward()
    assert inp_param.grad is not None, "INP tokens should receive gradients!"
    assert patch_param.grad is not None, "Patch tokens should receive gradients!"
    print("  [OK] Gradients flow to both INP and patch tokens.")

    print("\n[OK] All shape checks passed.")
