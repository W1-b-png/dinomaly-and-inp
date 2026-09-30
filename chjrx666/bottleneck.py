"""
MLP Bottleneck — directly ported from Dinomaly's ``bMlp``.

A single hidden-layer MLP with GELU activation and Dropout applied *before*
each linear projection (the Dinomaly "b" variant).  Used to compress / perturb
the fused encoder features before reconstruction.

Forward:
    x:  (B, N, C)  →  (B, N, C)   (same shape)
"""

import torch
import torch.nn as nn


class bMlp(nn.Module):
    """
    Bottleneck MLP — Dropout *before* fc1 and fc2 (Dinomaly convention).

    Args:
        in_features:   input / output channel dim (C).
        hidden_ratio:  hidden = in_features * hidden_ratio.
        drop:          dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = int(in_features * hidden_ratio)
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  (B, N, C)  input features.

        Returns:
            (B, N, C)  output features (same shape).
        """
        x = self.drop(x)         # Dinomaly: dropout BEFORE fc1
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)         # dropout BEFORE fc2
        x = self.fc2(x)
        x = self.drop(x)         # final dropout
        return x


class MLPBottleneck(nn.Module):
    """
    Thin wrapper around ``bMlp`` for a single-layer bottleneck.
    Keeps a ``ModuleList`` so the interface mirrors the original Dinomaly code
    (which iterates over a ModuleList in the forward pass).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            bMlp(in_features=embed_dim, hidden_ratio=hidden_ratio, drop=drop)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  (B, N, C)  fused encoder features.

        Returns:
            (B, N, C)  bottleneck output.
        """
        for blk in self.layers:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Shape self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("bottleneck.py  shape self-check")
    print("=" * 60)

    B, N, C = 2, 1024, 768  # ViT-B/14 @ 448px
    dummy = torch.randn(B, N, C)

    bottleneck = MLPBottleneck(embed_dim=C, hidden_ratio=4.0, drop=0.0)
    out = bottleneck(dummy)

    print(f"Input  : {dummy.shape}  (B={B}, N={N}, C={C})")
    print(f"Output : {out.shape}  (B={B}, N={N}, C={C})")
    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape} != {dummy.shape}"
    print("[OK] Shape check passed.")
