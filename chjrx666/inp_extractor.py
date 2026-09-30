"""
INP Extractor — ports the INP (Intrinsic Normal Prototype) extraction module
from INP-Former  (``models/vision_transformer.py``, ``INP_Former_Multi_Class.py``).

Architecture (exact INP-Former paper implementation):

    Learnable prototype tokens  (P, C)
           ↓  unsqueeze + repeat → (B, P, C)
    ┌─────────────────────────────────────────┐
    │  Aggregation_Block  (× num_blocks)       │
    │    LN → Aggregation_Attention → residual │
    │    LN → Mlp                  → residual │
    │                                          │
    │  Q  ←  prototype tokens  (B, P, C)       │
    │  K,V ←  patch tokens      (B, N, C)      │
    └─────────────────────────────────────────┘
           ↓
    INP tokens  (B, P, C)  — ready for decoder cross-attention

Input:   patch_tokens  (B, N, C)  — encoder final-layer features
Output:  inp_tokens    (B, P, C)  — P intrinsic normal prototypes

Reference:
    INP-Former  CVPR 2025
    https://github.com/.../INP-Former-main
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Minimal building blocks  (self-contained, no external deps)
# ===========================================================================

class DropPath(nn.Module):
    """DropPath (Stochastic Depth) — matches INP-Former ``DropPath``."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class Mlp(nn.Module):
    """Standard 2-layer MLP with GELU + Dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ===========================================================================
# Aggregation_Attention — Q from prototypes, K/V from patch tokens
# ===========================================================================

class AggregationAttention(nn.Module):
    """
    Cross-attention where **prototype tokens attend to patch tokens**.

    Q  ←  prototype tokens  (B, P, C)
    K, V ←  patch tokens     (B, N, C)

    Uses standard scaled dot-product attention with softmax.
    Ported exactly from INP-Former ``Aggregation_Attention``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)        # projects prototype tokens
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)   # projects patch tokens → K, V
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        prototype: torch.Tensor,   # (B, P, C)  — Q
        patch: torch.Tensor,       # (B, N, C)  — K, V
    ) -> torch.Tensor:
        """
        Args:
            prototype:  (B, P, C)  learnable prototype tokens.
            patch:      (B, N, C)  encoder patch features.

        Returns:
            (B, P, C)  prototype tokens updated by attending to patch features.
        """
        B, P, C = prototype.shape      # P = num prototypes
        N = patch.shape[1]             # N = num patch tokens

        # Q from prototype         (B, num_heads, P, head_dim)
        q = (
            self.q(prototype)
            .reshape(B, P, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # K, V from patch          each (B, num_heads, N, head_dim)
        kv = (
            self.kv(patch)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        # scaled dot-product attention  (B, num_heads, P, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # aggregate values          (B, num_heads, P, head_dim) → (B, P, C)
        x = (attn @ v).transpose(1, 2).reshape(B, P, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ===========================================================================
# Aggregation_Block — one transformer layer of prototype aggregation
# ===========================================================================

class AggregationBlock(nn.Module):
    """
    Single aggregation block:

        prototype = prototype + DropPath( AggregationAttention( LN(prototype), LN(patch) ) )
        prototype = prototype + DropPath( Mlp( LN(prototype) ) )

    Ported exactly from INP-Former ``Aggregation_Block``
    (models/vision_transformer.py:130-146).

    .. note::
        **Variable-name mapping** (our → original)::

            prototype  →  x      (learnable prototypes, used as Q)
            patch      →  y      (encoder patch features, used as K/V)

        ``self.norm1`` is intentionally shared across both inputs
        (``norm1(prototype)`` ≈ ``norm1(x)``, ``norm1(patch)`` ≈ ``norm1(y)``).
        This matches the original design — the same LayerNorm instance is
        called twice, once on each tensor.

    已与 INP-Former 源码核对一致（核对人/日期: ________）。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_val: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AggregationAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # DropPath — matched to original; defaults to Identity when drop_path_val=0
        self.drop_path = nn.Identity() if drop_path_val <= 0.0 else DropPath(drop_path_val)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        prototype: torch.Tensor,   # (B, P, C)  — original var name: x
        patch: torch.Tensor,       # (B, N, C)  — original var name: y
    ) -> torch.Tensor:
        """
        Args:
            prototype:  (B, P, C)  current prototype tokens (= original x).
            patch:      (B, N, C)  encoder patch features (= original y).

        Returns:
            (B, P, C)  refined prototype tokens.
        """
        # cross-attention: prototype attends to patch features
        # norm1 applied to both inputs (same LN instance) — matches original
        prototype = prototype + self.drop_path(
            self.attn(
                self.norm1(prototype),   # original: self.norm1(x)
                self.norm1(patch),       # original: self.norm1(y)
            )
        )
        # FFN
        prototype = prototype + self.drop_path(self.mlp(self.norm2(prototype)))
        return prototype


# ===========================================================================
# INPExtractor — full INP extraction module
# ===========================================================================

class INPExtractor(nn.Module):
    """
    INP Extractor — extracts P intrinsic normal prototypes from patch features.

    Matches the INP-Former paper implementation:
      1.  P learnable prototype tokens  (P, C)  initialised randomly.
      2.  1 × AggregationBlock  (default)  refines prototypes by cross-attending
          to encoder patch features.
      3.  Returns  (B, P, C)  INP tokens.

    Args:
        embed_dim:   feature dimension C.
        num_heads:   number of attention heads.
        num_inp:     P — number of INP prototypes (default 6, from INP-Former).
        num_blocks:  number of AggregationBlocks (default 1, from INP-Former).
        mlp_ratio:   hidden expansion ratio for the FFN.
        qkv_bias:    whether Q/K/V projections use bias.
        drop:        projection dropout.
        attn_drop:   attention dropout.
        init_std:    std for truncated normal init of prototype tokens.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_inp: int = 6,
        num_blocks: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,           # INP-Former uses bias=True
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_std: float = 1.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_inp = num_inp
        self.num_blocks = num_blocks

        # -- learnable prototype tokens  (P, C) -------------------------------
        # Initialised with randn (std=1.0), matching INP-Former.
        self.prototype_token = nn.Parameter(
            torch.randn(num_inp, embed_dim) * init_std
        )

        # -- aggregation blocks -----------------------------------------------
        self.blocks = nn.ModuleList([
            AggregationBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
            )
            for _ in range(num_blocks)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        """Truncated normal init for all Linear layers (INP-Former convention)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens:  (B, N, C)  encoder final-layer patch features.

        Returns:
            inp_tokens:    (B, P, C)  extracted INP prototypes.
        """
        B = patch_tokens.shape[0]                                   # batch size

        # expand learnable prototypes to batch  (P, C) → (B, P, C)
        inp = self.prototype_token.unsqueeze(0).repeat(B, 1, 1)    # (B, P, C)

        # refine through aggregation blocks
        for blk in self.blocks:
            inp = blk(inp, patch_tokens)                            # Q=inp, K/V=patch

        return inp  # (B, P, C)


# ===========================================================================
# Shape self-check
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("inp_extractor.py  shape self-check")
    print("=" * 60)

    B, N, C = 2, 1024, 768    # ViT-B/14 @ 448px
    P = 6                      # num INP prototypes

    dummy_patch = torch.randn(B, N, C)   # encoder final-layer features

    # -- build extractor -----------------------------------------------------
    extractor = INPExtractor(
        embed_dim=C,
        num_heads=12,
        num_inp=P,
        num_blocks=1,
        mlp_ratio=4.0,
        qkv_bias=True,
    )

    inp = extractor(dummy_patch)

    print(f"Patch tokens (input)  : ({B}, {N}, {C})")
    print(f"INP tokens  (output)  : {tuple(inp.shape)}")
    print(f"  B={B}, P={P}, C={C}")
    assert inp.shape == (B, P, C), f"Shape mismatch: {inp.shape} != ({B},{P},{C})"

    # -- gradient check ------------------------------------------------------
    print(f"\n--- Gradient check ---")
    extractor.train()
    patch_param = nn.Parameter(dummy_patch.clone())
    inp_out = extractor(patch_param)
    loss = inp_out.sum()
    loss.backward()
    has_grad_proto = extractor.prototype_token.grad is not None
    has_grad_patch = patch_param.grad is not None
    print(f"  prototype.grad exists : {has_grad_proto}  (expected: True)")
    print(f"  patch.grad exists     : {has_grad_patch}  (expected: True)")
    assert has_grad_proto, "Prototype tokens should receive gradients!"
    assert has_grad_patch, "Patch tokens should receive gradients through extractor!"
    print("  [OK] Gradients flow correctly.")

    # -- multiple blocks test ------------------------------------------------
    print(f"\n--- Multi-block test (num_blocks=2) ---")
    extractor2 = INPExtractor(embed_dim=C, num_heads=12, num_inp=P, num_blocks=2)
    inp2 = extractor2(dummy_patch)
    print(f"  INP shape (2 blocks): {tuple(inp2.shape)}")
    assert inp2.shape == (B, P, C)

    # -- different P test ----------------------------------------------------
    print(f"\n--- P=10 test ---")
    extractor3 = INPExtractor(embed_dim=C, num_heads=12, num_inp=10)
    inp3 = extractor3(dummy_patch)
    print(f"  INP shape (P=10)    : {tuple(inp3.shape)}")
    assert inp3.shape == (B, 10, C)

    print("\n[OK] All shape checks passed.")
