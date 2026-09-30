"""
Dinomaly Decoder with INP-guided cross-attention.

Two decoder flavours are provided:

  Block             -- original Dinomaly single-attention block (kept for
                      backward compatibility with the baseline).
  DecoderBlock      -- dual-attention block:
                      1) LinearAttention  (Q=x, K=learnable, V=x)
                         -> O(N*D^2) einsum-based, returns kv (DxD)
                      2) Prototype_Attention (Q=x, K=inp, V=inp) -- INP cross-attn
                      3) Mlp (FFN)
                      INP is passed as a forward argument and **shared across
                      all decoder layers** (not updated per layer).

  DinomalyDecoder   -- stack of DecoderBlocks; forward(x, inp) -> List[(B,N,C)]

Forward:
    x:  (B, N, C),  inp: (B, P, C)  ->  List[(B, N, C)]   (one per layer)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ===========================================================================
# Shared building blocks  (from Dinomaly-master/models/vision_transformer.py)
# ===========================================================================

class Mlp(nn.Module):
    """Standard 2-layer MLP with GELU + Dropout (post-activation dropout)."""

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


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """DropPath (Stochastic Depth)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class DropKey(nn.Module):
    """DropKey regularisation -- sets random attention positions to -1e12."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        if self.training:
            m_r = torch.ones_like(attn) * self.p
            attn = attn + torch.bernoulli(m_r) * -1e12
        return attn


# ===========================================================================
# LinearAttention -- Q=x, K=learnable tokens, V=x  (elu kernel, no softmax)
# ===========================================================================

class LinearAttention(nn.Module):
    """
    Linearised softmax-free attention with **learnable key tokens**.

        Q  <-  x  (patch tokens)          (B, N, C)
        K  <-  self.k_token               (num_tokens, C)  -- learnable, shared
        V  <-  self.v(x)                  (B, N, C)        -- projected from x

    Uses  elu(q)+1  and  elu(k)+1  kernels so that attention weights are
    all-positive but not sharpened by a softmax (key Dinomaly principle).
    Row-wise normalisation replaces softmax.

    **Einsum-based linear attention** -- matches Dinomaly ``LinearAttention2``
    (models/vision_transformer.py:261-272).  Three einsum calls, O(N . D^2),
    no explicit NxN matrix.  Mathematically identical to the explicit form::

        attn = (q @ k^T) / row_sum(q @ k^T)    # (B, heads, N, N)
        x = attn @ v

    but computed as::

        kv = einsum('…sd,…se->…de',  k, v)            # (B, heads, D, D)
        z  = 1 / einsum('…sd,…d->…s', q, k.sum(-2))   # (B, heads, N)
        x  = einsum('…de,…sd,…s->…se', kv, q, z)

    **num_tokens must equal the input patch count N** so that k and v share
    the same sequence length (required by the einsum reduction over N).
    A runtime assertion enforces this.

    Returns:
        x:  (B, N, C)            attended output.
        kv: (B, num_heads, D, D)  key-value outer product (for inspection).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_tokens: int = 1024,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Q and V are projected from x;  K is a learnable parameter bank
        self.q = nn.Linear(dim, dim, bias=qkv_bias)            # Q <- x
        self.v = nn.Linear(dim, dim, bias=qkv_bias)            # V <- x
        self.k_token = nn.Parameter(torch.randn(num_tokens, dim))   # learnable K

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:  (B, N, C)  patch token features.

        Returns:
            x:  (B, N, C)            attended output.
            kv: (B, num_heads, D, D)  key-value outer-product (O(N*D^2)).
                The DxD matrix encodes the key-value interaction summed
                over all N positions; D = head_dim = C // num_heads.

        Raises:
            AssertionError if N != num_tokens.
        """
        B, N, C_b = x.shape
        num_tokens = self.num_tokens
        head_dim = C_b // self.num_heads

        # ---- runtime guard: num_tokens must equal patch count N --------------
        assert N == num_tokens, (
            f"LinearAttention: num_tokens ({num_tokens}) must equal "
            f"the input patch count N ({N})."
        )

        # ---- Q from x  (B, num_heads, N, head_dim) --------------------------
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )

        # ---- K from learnable tokens  (B, num_heads, N, head_dim) -----------
        k = (
            self.k_token
            .unsqueeze(0)                            # (1, N, C)
            .expand(B, -1, -1)                       # (B, N, C)
            .reshape(B, N, self.num_heads, head_dim)
            .permute(0, 2, 1, 3)                     # (B, num_heads, N, head_dim)
        )

        # ---- V from x  (B, num_heads, N, head_dim) ---------------------------
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, head_dim)
            .permute(0, 2, 1, 3)                     # (B, num_heads, N, head_dim)
        )

        # ---- elu + 1 kernel  (Dinomaly linear attention) ---------------------
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # ---- Einsum-based linear attention  O(N*D^2), no NxN matrix ----------
        # Matches Dinomaly LinearAttention2 (models/vision_transformer.py:269-271).
        # Proof of equivalence (explicit -> einsum):
        #   attn = (q @ k^T) / Σ_l(q_i . k_l)    (B, heads, N, N)
        #   x_i = Σ_j attn_{ij} . v_j
        #       = Σ_j (q_i . k_j) . v_j / Σ_l (q_i . k_l)
        #       = q_i @ [Σ_j k_j ⊗ v_j] / Σ_l (q_i . k_l)   <-  einsum form
        #
        # Computed in fp32: the kv outer-product and the row-sum z accumulate
        # over N=1024 tokens.  bf16 has fp32's exponent range but only a 7-bit
        # mantissa, so a 1024-term sum in bf16 loses ~3 digits of precision
        # (and fp16 would overflow its ~65504 range → inf/NaN).  Also,
        # ``torch.einsum`` is on the autocast op list, so even fp32 inputs get
        # downcast back to bf16 inside autocast — must disable autocast
        # explicitly for this block.  These are small D×D / N×D ops, so the
        # extra cost is negligible.  Result cast back to the activation dtype.

        with torch.amp.autocast(device_type='cuda', enabled=False):
            kv = torch.einsum('...sd,...se->...de', k.float(), v.float())  # (B, heads, D, D)
            z = 1.0 / (torch.einsum('...sd,...d->...s',
                                    q.float(), k.float().sum(dim=-2)) + 1e-8)
            #                                                      ^ (B, heads, N)
            x = torch.einsum('...de,...sd,...s->...se',
                             kv, q.float(), z)                 # (B, heads, N, D)
        x = x.to(v.dtype)                                       # back to the activation dtype

        x = self.attn_drop(x)
        x = x.transpose(1, 2).reshape(B, N, C_b)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


# ===========================================================================
# Prototype_Attention -- cross-attention with INP tokens
#   Ported from INP-Former  models/vision_transformer.py
# ===========================================================================

class PrototypeAttention(nn.Module):
    """
    INP-guided cross-attention.

    Q  <-  x   (patch tokens)        (B, N, C)
    K, V  <-  inp  (INP tokens)      (B, P, C)

    Q and K are L2-normalised; attention uses a **learnable per-head scale**
    followed by **ReLU activation (no softmax)**.  This forces the attention to
    select the most similar INP per query token while allowing the magnitude
    of the response to vary.

    Ported from INP-Former ``Prototype_Attention``
    (models/vision_transformer.py:70-100).

    .. note::
        **已核对原版 (line 95):**  ``attn = F.relu(attn)`` -- ReLU 为原始设计，
        并非 softmax。此项已逐行比对确认。

        **Variable-name mapping** (our -> original)::

            x    ->  x               (patch tokens, used as Q)
            inp  ->  prototype_token (INP tokens, used as K/V)

        **learn_scale init:**  ``torch.ones(num_heads, 1, 1)``
        (matches original line 77).

    已与 INP-Former 源码核对一致（核对人/日期: ________）。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # learnable per-head scale -- init matches original (torch.ones)
        self.learn_scale = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)           # Q from x
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)      # K, V from inp (prototype_token)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        inp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, N, C)  query tokens (= original x).
            inp:  (B, P, C)  INP prototype tokens (= original prototype_token).

        Returns:
            x:     (B, N, C)  attended output.
            attn:  (B, num_heads, N, P)  cross-attention weights.
        """
        B, N, C = x.shape         # N = patch tokens, C = embed_dim
        P = inp.shape[1]          # P = number of INP prototypes

        # Q from x                 (B, num_heads, N,   head_dim)
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # K, V from inp            (B, num_heads, P,   head_dim) each
        kv = (
            self.kv(inp)
            .reshape(B, P, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]        # each (B, num_heads, P, head_dim)

        # L2 normalise Q and K  (INP-Former convention, original line 92-93)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # attention: (B, num_heads, N, P)
        attn = (q @ k.transpose(-2, -1)) * self.learn_scale
        # 已核对原版: F.relu(attn) -- ReLU gate, 非 softmax (original line 95)
        attn = F.relu(attn)
        attn = self.attn_drop(attn)

        # aggregate V                (B, num_heads, N, head_dim) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


# ===========================================================================
# Block -- original Dinomaly single-attention block  (kept for reference)
# ===========================================================================

class Block(nn.Module):
    """
    Original Dinomaly Transformer block:
        LN -> Attention -> residual -> LN -> MLP -> residual.
    Attention type is injected via ``attn``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_val: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn=LinearAttention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path_val) if drop_path_val > 0.0 else nn.Identity()
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
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:               (B, N, C)  input.
            return_attention: if True, also return attention weights.

        Returns:
            x  or  (x, attn)
        """
        y, attn = self.attn(self.norm1(x))

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn
        return x


# ===========================================================================
# DecoderBlock -- dual-attention decoder block  (core change for INP-Former)
# ===========================================================================

class DecoderBlock(nn.Module):
    """
    Dual-attention decoder block:

        x = x + DropPath( LinearAttention   ( LN1(x)        ) )   # Q=x, K=learnable, V=x
        x = x + DropPath( PrototypeAttention( LN2(x), inp   ) )   # INP cross-attn
        x = x + DropPath( Mlp               ( LN3(x)        ) )   # FFN

    The first attention uses **learnable key tokens** with Q and V projected
    from x. ``num_tokens`` must equal the input patch count N.
    INP tokens in the second attention are **read-only** -- they flow through
    K and V but are never updated.  All decoder layers share the same INP.

    Args:
        dim:         embedding dimension C.
        num_heads:   number of attention heads.
        num_tokens:  number of learnable K tokens for LinearAttention.
                     Must equal the input patch count N (default 1024,
                     corresponding to 448x448 input, patch_size=14,
                     32x32 grid).
        mlp_ratio:   FFN hidden-expansion ratio.
        qkv_bias:    whether Q/K/V linear projections use bias.
        qk_scale:    explicit attention scale (default: head_dim ** -0.5).
        drop:        projection dropout rate.
        attn_drop:   attention dropout rate.
        drop_path_val: DropPath stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_tokens: int = 1024,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_val: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        # -- sub-layer 1: LinearAttention (Q=x, K=learnable, V=x) ------------
        self.norm1 = norm_layer(dim)
        self.attn1 = LinearAttention(
            dim,
            num_heads=num_heads,
            num_tokens=num_tokens,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # -- sub-layer 2: PrototypeAttention (INP cross-attention) ------------
        self.norm2 = norm_layer(dim)
        self.attn2 = PrototypeAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # -- sub-layer 3: FFN ------------------------------------------------
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # -- DropPath --------------------------------------------------------
        self.drop_path = DropPath(drop_path_val) if drop_path_val > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        inp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, N, C)  patch token features from previous layer.
            inp:  (B, P, C)  INP prototype tokens (**shared, read-only**).

        Returns:
            x:    (B, N, C)  updated features.
        """
        # ---- 1) Q=x, K=learnable, V projected from x  (LinearAttention) -----
        y1, _ = self.attn1(self.norm1(x))
        #        y1: (B, N, C)
        x = x + self.drop_path(y1)

        # ---- 2) INP cross-attention (PrototypeAttention) -------------------
        y2, _ = self.attn2(self.norm2(x), inp)
        #        y2: (B, N, C)   inp: (B, P, C) -- shared, not updated
        x = x + self.drop_path(y2)

        # ---- 3) FFN --------------------------------------------------------
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x


# ===========================================================================
# DinomalyDecoder -- stack of DecoderBlocks with shared INP
# ===========================================================================

class DinomalyDecoder(nn.Module):
    """
    Dinomaly decoder with INP-guided cross-attention.

    A stack of ``DecoderBlock`` modules.  INP prototype tokens are passed
    via ``forward()`` and shared across **all** decoder layers (read-only).

    Args:
        embed_dim:     feature dimension C.
        num_heads:     number of attention heads.
        num_layers:    how many decoder blocks (default 8).
        num_tokens:    number of learnable K tokens for LinearAttention.
                       Must equal the input patch count N (default 1024,
                       corresponding to 448x448 input, patch_size=14,
                       32x32 grid).
        mlp_ratio:     hidden-expansion ratio for the MLP inside each block.
        qkv_bias:      whether QKV projection uses bias.
        drop:          projection dropout.
        attn_drop:     attention dropout.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 8,
        num_tokens: int = 1024,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_tokens=num_tokens,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        inp: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            x:          (B, N, C)  bottleneck output.
            inp:        (B, P, C)  INP prototype tokens -- shared across
                                   **all** decoder layers, not updated per layer.

        Returns:
            de_list:  List[(B, N, C)] -- one output per decoder layer
                      (layer 0 ... L-1, in forward order; the caller
                       typically reverses for loss computation).
        """
        # Gradient checkpointing needs a grad-tracking input to reconstruct
        # activations in backward; if x arrived without a grad requirement
        # (e.g. a detached bottleneck output), force it on.
        if self.training and not x.requires_grad:
            x = x.requires_grad_(True)

        de_list: List[torch.Tensor] = []
        for blk in self.blocks:
            if self.training:
                # Recompute each block's activations in backward instead of
                # storing them -- trades compute for memory. use_reentrant=False
                # is the standard, thread-safe path.
                x = checkpoint(blk, x, inp, use_reentrant=False)
            else:
                x = blk(x, inp)                                        # INP shared
            de_list.append(x)
        return de_list


# ===========================================================================
# Shape self-check
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("decoder.py  shape self-check  (dual-attention DecoderBlock)")
    print("=" * 60)

    B, N, C = 2, 1024, 768    # ViT-B/14 @ 448px: 32x32 patches, 768-dim
    P = 6                      # number of INP prototypes
    num_tokens = 1024          # must equal N (input patch count)

    dummy_x = torch.randn(B, N, C)       # bottleneck features
    dummy_inp = torch.randn(B, P, C)     # INP tokens (shared across layers)

    # -- K token + V projection shape check ------------------------------------
    print(f"\n--- K token + V projection check ---")
    attn_demo = LinearAttention(dim=C, num_heads=12, num_tokens=num_tokens)
    # k_token: learnable Parameter
    p = attn_demo.k_token
    is_param = isinstance(p, nn.Parameter)
    print(f"  k_token : shape={tuple(p.shape)}  "
          f"is_Parameter={is_param}  requires_grad={p.requires_grad}")
    assert is_param, "k_token must be nn.Parameter!"
    assert p.shape == (num_tokens, C), f"k_token shape: {p.shape}"
    # v: nn.Linear projection (not a Parameter token)
    v_is_linear = isinstance(attn_demo.v, nn.Linear)
    print(f"  v       : nn.Linear  (in={attn_demo.v.in_features}, "
          f"out={attn_demo.v.out_features}, bias={attn_demo.v.bias is not None})")
    assert v_is_linear, "v must be nn.Linear (projected from x)!"
    print("  [OK] K is learnable Parameter, V is nn.Linear projection.")

    # -- single DecoderBlock test --------------------------------------------
    blk = DecoderBlock(dim=C, num_heads=12, num_tokens=num_tokens)
    out_1 = blk(dummy_x, dummy_inp)

    print(f"\n--- Single DecoderBlock ---")
    print(f"  x    : ({B}, {N}, {C})")
    print(f"  inp  : ({B}, {P}, {C})")
    print(f"  out  : {tuple(out_1.shape)}")
    assert out_1.shape == dummy_x.shape, \
        f"DecoderBlock shape mismatch: {out_1.shape} != {dummy_x.shape}"
    print("  [OK] Single block shape correct.")

    # -- verify attention shapes: einsum returns kv (DxD), ProtoAttn stays (NxP) -----
    head_dim = C // 12            # 768 / 12 = 64
    print(f"\n--- Attention shape check: kv (DxD), attn2 (NxP) ---")
    blk.eval()
    with torch.no_grad():
        _, kv1 = blk.attn1(blk.norm1(dummy_x))
        _, attn2 = blk.attn2(blk.norm2(out_1), dummy_inp)
    print(f"  kv1 (LinearAttn)       : {tuple(kv1.shape)}  "
          f"(B={B}, heads=12, D={head_dim}, D={head_dim})  <- einsum O(N*D^2)")
    print(f"  attn2 (ProtoAttn)      : {tuple(attn2.shape)}  "
          f"(B={B}, heads=12, N={N}, P={P})")
    assert kv1.shape == (B, 12, head_dim, head_dim), \
        f"kv1 shape: {kv1.shape} != ({B}, 12, {head_dim}, {head_dim})"
    assert attn2.shape == (B, 12, N, P), \
        f"attn2 shape: {attn2.shape} != ({B}, 12, {N}, {P})"
    print("  [OK] kv1 (DxD) via einsum, attn2 (NxP).")

    # -- full decoder stack test ---------------------------------------------
    print(f"\n--- DinomalyDecoder ({8} layers) ---")
    decoder = DinomalyDecoder(embed_dim=C, num_heads=12, num_layers=8,
                              num_tokens=num_tokens)

    de_list = decoder(dummy_x, dummy_inp)

    print(f"  x          : ({B}, {N}, {C})")
    print(f"  inp (shared): ({B}, {P}, {C})")
    print(f"  num layers : {len(de_list)}  (num_tokens={num_tokens})")
    for idx, d in enumerate(de_list):
        print(f"    de[{idx}] : {tuple(d.shape)}  (B, N, C)")
        assert d.shape == dummy_x.shape, \
            f"Layer {idx} shape mismatch: {d.shape}"

    de_rev = de_list[::-1]
    print(f"\n  Reversed list (for loss matching):")
    print(f"    de_rev[0] (deepest)  : {tuple(de_rev[0].shape)}")
    print(f"    de_rev[-1] (shallowest): {tuple(de_rev[-1].shape)}")

    # -- verify K token + V projection gradients flow --------------------------
    print(f"\n--- Gradient check: k_token / v (Linear) receive grad ---")
    decoder.train()
    # Collect k_token (Parameter) and v (Linear) params from all blocks
    kv_params: List[Tuple[str, nn.Parameter]] = []
    for i, blk in enumerate(decoder.blocks):
        kv_params.append((f"block[{i}].attn1.k_token", blk.attn1.k_token))
        kv_params.append((f"block[{i}].attn1.v.weight", blk.attn1.v.weight))
    inp_param = nn.Parameter(dummy_inp.clone())
    de_out = decoder(dummy_x, inp_param)
    loss = de_out[-1].sum()
    loss.backward()
    for name, p in kv_params:
        has_grad = p.grad is not None
        g_sum = p.grad.abs().sum().item() if p.grad is not None else 0.0
        print(f"  {name}: grad={'YES' if has_grad else 'NO'}  "
              f"(abs-sum={g_sum:.4f})")
        assert has_grad, f"{name} should receive gradients!"
        assert g_sum > 0.0, f"{name} gradient is all zeros!"
    has_grad_inp = inp_param.grad is not None
    print(f"  inp (cross-attn K/V)  : grad={'YES' if has_grad_inp else 'NO'}")
    print(f"  loss value            : {loss.item():.2f}")
    assert has_grad_inp, "INP should receive gradients through cross-attention!"
    print("  [OK] Gradients flow to k_token, v.weight and INP correctly.")

    # -- verify unique k_token per block (not shared) -------------------------
    print(f"\n--- Independence check: each block has its own k_token ---")
    for i in range(len(decoder.blocks) - 1):
        same = (decoder.blocks[i].attn1.k_token is decoder.blocks[i+1].attn1.k_token)
        assert not same, f"block[{i}] and block[{i+1}] share the same k_token object!"
    print("  [OK] Each block owns independent k_token.")

    print("\n[OK] All shape checks passed.")
