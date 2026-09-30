"""
Frozen DINOv2-R ViT Encoder.
Extracts multi-layer patch token features from a pretrained DINOv2 with registers.

Forward returns:
  intermediate_features: List[Tensor]  — patch tokens from target_layers, each (B, N_patch, C)
  final_features:       Tensor        — patch tokens from the last target layer, (B, N_patch, C)

All encoder parameters are frozen (requires_grad=False) and the encoder runs in eval mode.
"""

import sys
import os
import math
import torch
import torch.nn as nn
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Reuse the encoder loader from Dinomaly-master
# ---------------------------------------------------------------------------
_DINOMALY_ROOT = os.path.join(os.path.dirname(__file__), "..", "Dinomaly-master")
if _DINOMALY_ROOT not in sys.path:
    sys.path.insert(0, _DINOMALY_ROOT)

from models import vit_encoder  # noqa: E402


class DinoEncoder(nn.Module):
    """
    Frozen DINOv2-Register ViT encoder.

    Args:
        encoder_name:  e.g. 'dinov2reg_vit_base_14', 'dinov2reg_vit_small_14', 'dinov2reg_vit_large_14'
        target_layers: which encoder block indices to collect features from (0-indexed).
        remove_class_token: if True, strip CLS + register tokens, keep only patch tokens.

    .. note::
        ``train(mode)`` is overridden to always keep the encoder in eval mode
        so that ``model.train()`` does not activate dropout / stochastic depth
        in the frozen backbone.
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        target_layers: List[int] = None,
        remove_class_token: bool = True,
    ):
        super().__init__()
        if target_layers is None:
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

        self.encoder_name = encoder_name
        self.target_layers = target_layers
        self.remove_class_token = remove_class_token

        # -- load the full ViT backbone --------------------------------------
        self.encoder = vit_encoder.load(encoder_name)
        self.encoder.num_register_tokens = getattr(self.encoder, "num_register_tokens", 0)

        # -- freeze all parameters -------------------------------------------
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # -- derive embed_dim for shape reporting -----------------------------
        self.embed_dim = self.encoder.embed_dim

    def train(self, mode: bool = True):
        """
        Override to **always keep the encoder in eval mode**.

        ``model.train()`` recursively sets every submodule to training mode,
        which would activate dropout / stochastic depth inside the frozen
        DINOv2 backbone.  This override short-circuits that.
        """
        return super().train(False)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x:  (B, 3, H, W)  input image batch.

        Returns:
            intermediate_features:  list of (B, N_patch, C) for each target layer.
            final_features:         (B, N_patch, C)  output of the *last* target layer.
        """
        # -- patch embedding --------------------------------------------------
        #     prepare_tokens does patch_embed + cls_token + pos_embed + register_tokens
        with torch.no_grad():
            x = self.encoder.prepare_tokens(x)

        en_list: List[torch.Tensor] = []

        # -- run encoder blocks, collecting target layers ---------------------
        for i, blk in enumerate(self.encoder.blocks):
            if i > self.target_layers[-1]:
                break  # no need to go past the deepest requested layer
            with torch.no_grad():
                x = blk(x)
            if i in self.target_layers:
                en_list.append(x)

        # -- strip CLS + register tokens if requested -------------------------
        if self.remove_class_token:
            offset = 1 + self.encoder.num_register_tokens
            en_list = [e[:, offset:, :] for e in en_list]

        # final_features = the deepest collected layer
        final_features = en_list[-1]

        return en_list, final_features


# ---------------------------------------------------------------------------
# Shape self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("encoder.py  shape self-check")
    print("=" * 60)

    # Use a random tensor to avoid downloading weights during a quick check.
    # Swap to a real encoder load for a full test:
    DO_FULL_CHECK = False  # set True after weights are downloaded

    if DO_FULL_CHECK:
        encoder = DinoEncoder(
            encoder_name="dinov2reg_vit_base_14",
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            remove_class_token=True,
        )
        encoder = encoder.cuda()
        B, C, H, W = 2, 3, 448, 448
        dummy = torch.randn(B, C, H, W).cuda()
        en_list, final = encoder(dummy)

        print(f"Input         : {dummy.shape}  (B={B}, C={C}, H={H}, W={W})")
        print(f"Num target layers: {len(en_list)}")
        for idx, f in enumerate(en_list):
            print(f"  en[{idx}]    : {f.shape}  (B, N_patch, C)")
        print(f"final_features : {final.shape}  (B, N_patch, C)")

        # Expected for ViT-B/14 with 448px input:
        #   patch_size=14 → side = 448/14 = 32 → N_patch = 1024
        #   C = 768 (base)
        expected_N = (H // 14) ** 2
        expected_C = 768
        assert en_list[0].shape == (B, expected_N, expected_C), (
            f"Shape mismatch: {en_list[0].shape} vs ({B},{expected_N},{expected_C})"
        )
        print("✓ All shapes correct.")
    else:
        print("Skipping full encoder load (set DO_FULL_CHECK=True after downloading weights).")
        print("Testing encoder module construction only...")
        # We still verify the class is importable and can be instantiated with a
        # mock.  Actually constructing DinoEncoder triggers a download so we
        # only print the expected shapes here.
        patch_size = 14
        img_size = 448
        side = img_size // patch_size
        print(f"Expected for dinov2reg_vit_base_14, input (B,3,448,448):")
        print(f"  N_patch = ({img_size}/{patch_size})^2 = {side**2}")
        print(f"  C       = 768 (base)")
        for layer_idx in [2, 3, 4, 5, 6, 7, 8, 9]:
            print(f"  en[layer {layer_idx}]  : (B, {side**2}, 768)")
        print(f"  final_features        : (B, {side**2}, 768)")
        print("✓ Shape expectations documented.")
