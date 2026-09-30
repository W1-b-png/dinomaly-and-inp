"""
Dinomaly + INP-Former hybrid model.

  DinoAnomaly  — full pipeline with INP Extractor + dual-attention decoder

DinoAnomaly forward:
    1. Frozen DINOv2 encoder  →  en_list (List of B,N,C), final (B,N,C)
    2. Fuse target layers     →  (B,N,C)
    3. INP Extractor          →  inp (B,P,C)   +  coherence loss  g_loss
    4. [opt] INP Fusion:  cosine-sim(patches, INP) → softmax → weighted sum
              → Linear projection → add back to fused features   (B,N,C)
    5. MLP Bottleneck         →  (B,N,C)
    6. Decoder (x, inp)       →  de_list (List of B,N,C)
    7. Fuse + reshape         →  en, de  (List of B,C,H,W)
    8. Return  en, de, g_loss

Training loss = reconstruction_loss(en, de) + lambda_cohere * g_loss
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import DinoEncoder
from bottleneck import MLPBottleneck
from decoder import DinomalyDecoder
from inp_extractor import INPExtractor
from losses import inp_coherence_loss


# ===========================================================================
# DinoAnomaly — full Dinomaly + INP-Former hybrid
# ===========================================================================

class DinoAnomaly(nn.Module):
    """
    Full Dinomaly + INP-Former hybrid pipeline.

    Forward::

        x  →  Frozen Encoder  →  en_list, final
                                 │
              ┌──────────────────┘
              ▼
         fuse(en_list)  →  (B, N, C)  ──→  INP Extractor  →  inp (B, P, C)
              │                           │                    │
              │                           │  g_loss =          │
              │                           │  coherence(patch,  │
              │                           │  inp)              │
              │                           ▼                    │
              │                    ┌── INP Fusion ──┐          │
              │                    │  cos-sim +     │          │
              │                    │  softmax +     │          │
              │                    │  weighted sum  │          │
              │                    │  + Linear proj │          │
              │                    └───────┬────────┘          │
              │                            │  (optional)       │
              │              ┌─────────────┘                   │
              │              ▼                                 │
              │         x_fused + inp_aggregated               │
              │              │                                 │
              ▼              ▼                                 │
         Bottleneck  →  (B, N, C)                              │
              │                                                │
              ▼                                                │
         Decoder(x, inp)  →  de_list                           │
              │                                                │
              ▼                                                │
         fuse + reshape  →  en, de  (B,C,H,W)                  │
              └────────────────────────────────────────────────┘
                             return  en, de, g_loss

    Training::

        en, de, g_loss = model(img)
        L = reconstruction_loss(en, de) + lambda_cohere * g_loss

    Args:
        encoder:          frozen DINOv2 encoder.
        inp_extractor:    INP extractor module.
        bottleneck:       MLP bottleneck.
        decoder:          dual-attention decoder (requires INP in forward).
        target_layers:    encoder block indices to collect.
        fuse_layer_encoder / fuse_layer_decoder: grouping indices for fusion.
        remove_class_token: whether CLS/reg tokens are stripped upstream.
        lambda_cohere:    weight for INP coherence loss (configurable).
        use_inp_fusion:   if True (default), aggregate INP tokens back into
                          the feature stream before bottleneck via cosine-sim
                          attention + Linear projection.  Set False for ablation.
    """

    def __init__(
        self,
        encoder: DinoEncoder,
        inp_extractor: INPExtractor,
        bottleneck: MLPBottleneck,
        decoder: DinomalyDecoder,
        target_layers: List[int] = None,
        fuse_layer_encoder: List[List[int]] = None,
        fuse_layer_decoder: List[List[int]] = None,
        remove_class_token: bool = True,
        lambda_cohere: float = 0.2,
        use_inp_fusion: bool = True,
    ):
        super().__init__()
        # --- defaults --------------------------------------------------------
        if target_layers is None:
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        if fuse_layer_encoder is None:
            fuse_layer_encoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
        if fuse_layer_decoder is None:
            fuse_layer_decoder = [[0, 1, 2, 3, 4, 5, 6, 7]]

        # --- submodules ------------------------------------------------------
        self.encoder = encoder
        self.inp_extractor = inp_extractor
        self.bottleneck = bottleneck
        self.decoder = decoder

        # INP fusion projection  (independent Linear, no bias — for ablation)
        embed_dim = decoder.embed_dim if hasattr(decoder, 'embed_dim') else 768
        self.inp_fusion_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # --- config ----------------------------------------------------------
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.lambda_cohere = lambda_cohere
        self.use_inp_fusion = use_inp_fusion

        if self.encoder is not None:
            enc = self.encoder.encoder
            self.num_register_tokens = getattr(enc, "num_register_tokens", 0)
        else:
            self.num_register_tokens = 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x:  (B, 3, H, W)  input image batch.

        Returns:
            en / de / g_loss  — see ``forward_from_features``.
        """
        # ---- 1. Frozen encoder ----------------------------------------------
        #        en_list:  List of (B, N_patch, C)  per target layer
        #        final:    (B, N_patch, C)           deepest target layer
        en_list, _final = self.encoder(x)                               # all (B, N, C)
        return self.forward_from_features(en_list)

    def forward_from_features(
        self, en_list: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Full pipeline from pre-computed encoder features (skips the frozen
        encoder forward — used by ``train.py --use-cache``).

        Args:
            en_list:  List[(B, N_patch, C)]  per target layer — exactly what
                      ``DinoEncoder.forward`` returns (CLS/reg tokens already
                      stripped when ``remove_class_token=True``).

        Returns:
            en:      List[(B, C, side, side)]  fused encoder feature maps
                     (reconstruction targets).
            de:      List[(B, C, side, side)]  fused decoder feature maps
                     (reconstruction predictions).
            g_loss:  scalar  — INP coherence loss (for adding to total loss).
        """
        side = int(math.sqrt(en_list[0].shape[1]))                      # patch grid side

        # ---- 2. Fuse target layers ------------------------------------------
        #        Mean-pool all collected layers → single representation.
        x_fused = self._fuse_feature(en_list)                            # (B, N, C)

        # ---- 3. INP Extractor -----------------------------------------------
        #        Cross-attention: learnable prototypes ← patch features.
        inp = self.inp_extractor(x_fused)                                # (B, P, C)

        # ---- 4. INP Coherence Loss ------------------------------------------
        #        Encourages every patch token to be close to at least one INP.
        g_loss, _, _ = inp_coherence_loss(x_fused, inp)                  # scalar

        # ---- 5. INP Fusion (optional — skip with use_inp_fusion=False) --------
        #        Cosine-sim attention over P INP prototypes, softmax-normalised,
        #        weighted sum → aggregate (B, P, C) → (B, N, C), then project
        #        through a learned Linear layer and add back to the main stream.
        if self.use_inp_fusion:
            # pairwise cosine similarity  (B, N, P)
            sim = F.cosine_similarity(
                x_fused.unsqueeze(2),       # (B, N, 1, C)
                inp.unsqueeze(1),           # (B, 1, P, C)
                dim=-1,
            )                                # (B, N, P)
            attn = F.softmax(sim, dim=-1)    # (B, N, P)  — row-normalised over P
            # weighted sum over INP tokens  (B, N, P) @ (B, P, C) → (B, N, C)
            inp_aggregated = attn @ inp
            # project and fuse
            x_fused = x_fused + self.inp_fusion_proj(inp_aggregated)

        # ---- 6. Bottleneck --------------------------------------------------
        x_fused = self.bottleneck(x_fused)                               # (B, N, C)

        # ---- 7. Decoder (with shared INP cross-attention) -------------------
        de_list = self.decoder(x_fused, inp)                               # List[(B, N, C)]
        de_list = de_list[::-1]                                           # deepest first

        # ---- 8. Fuse grouped layers → (B, C, H, W) --------------------------
        en = [
            self._fuse_feature([en_list[idx] for idx in idxs])
            for idxs in self.fuse_layer_encoder
        ]
        de = [
            self._fuse_feature([de_list[idx] for idx in idxs])
            for idxs in self.fuse_layer_decoder
        ]

        # CLS/register already stripped by DinoEncoder when remove_class_token=True
        if not self.remove_class_token:
            offset = 1 + self.num_register_tokens
            en = [e[:, offset:, :] for e in en]
            de = [d[:, offset:, :] for d in de]

        # reshape: (B, N, C) → (B, C, side, side)
        en = [
            e.permute(0, 2, 1).reshape(e.shape[0], -1, side, side).contiguous()
            for e in en
        ]
        de = [
            d.permute(0, 2, 1).reshape(d.shape[0], -1, side, side).contiguous()
            for d in de
        ]

        return en, de, g_loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fuse_feature(feat_list: List[torch.Tensor]) -> torch.Tensor:
        """Mean-pool a list of same-shape tensors along a new stack dim."""
        return torch.stack(feat_list, dim=1).mean(dim=1)

# ===========================================================================
# Convenience builders
# ===========================================================================

def build_dino_anomaly(
    encoder_name: str = "dinov2reg_vit_base_14",
    target_layers: List[int] = None,
    fuse_layer_encoder: List[List[int]] = None,
    fuse_layer_decoder: List[List[int]] = None,
    embed_dim: int = 768,
    num_heads: int = 12,
    num_inp: int = 6,
    num_inp_blocks: int = 1,
    num_decoder_layers: int = 8,
    num_decoder_tokens: int = 1024,
    bottleneck_hidden_ratio: float = 4.0,
    # Dinomaly ``Noisy Bottleneck`` — dropout=0.2 is the canonical paper
    # value;  see ``dinomaly_mvtec_uni.py`` line 137 et al. (6/7 scripts use
    # 0.2;  only Real-IAD unified uses 0.4).  The bMlp class applies dropout
    # *three times* per forward pass (pre-fc1, post-act, post-fc2).
    bottleneck_drop: float = 0.2,
    inp_mlp_ratio: float = 4.0,
    inp_drop: float = 0.0,
    inp_attn_drop: float = 0.0,
    decoder_mlp_ratio: float = 4.0,
    decoder_drop: float = 0.0,
    decoder_attn_drop: float = 0.0,
    lambda_cohere: float = 0.2,
    use_inp_fusion: bool = True,
) -> DinoAnomaly:
    """
    Build a complete DinoAnomaly model with sensible defaults.

    Args:
        encoder_name:      'dinov2reg_vit_base_14' / 'dinov2reg_vit_small_14'
                           / 'dinov2reg_vit_large_14'.
        target_layers:     encoder layer indices to collect (default [2..9]).
        fuse_layer_encoder / fuse_layer_decoder: grouping indices for fusion.
        embed_dim:         C (768 for base, 384 for small, 1024 for large).
        num_heads:         attention heads (12 for base, 6 for small, 16 for large).
        num_inp:           P — number of INP prototypes (default 6).
        num_inp_blocks:    number of AggregationBlocks in INP Extractor (default 1).
        num_decoder_layers:  number of DecoderBlocks (default 8).
        num_decoder_tokens:  number of learnable K tokens in each decoder block's
                             LinearAttention. Must equal input patch count N
                             (default 1024, for 448×448 input, patch_size=14,
                             32×32 grid).
        lambda_cohere:     weight for INP coherence loss in total loss.
        use_inp_fusion:   if True (default), fuse INP tokens back into the
                          feature stream before bottleneck (cosine-sim attention
                          + Linear projection).  Set False for ablation.
        (other args control dropout/ratio for bottleneck, INP, decoder.)

    Returns:
        DinoAnomaly  ready for training.
    """
    if target_layers is None:
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    if fuse_layer_encoder is None:
        fuse_layer_encoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
    if fuse_layer_decoder is None:
        fuse_layer_decoder = [[0, 1, 2, 3, 4, 5, 6, 7]]

    encoder = DinoEncoder(
        encoder_name=encoder_name,
        target_layers=target_layers,
        remove_class_token=True,
    )

    inp_extractor = INPExtractor(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_inp=num_inp,
        num_blocks=num_inp_blocks,
        mlp_ratio=inp_mlp_ratio,
        drop=inp_drop,
        attn_drop=inp_attn_drop,
    )

    bottleneck = MLPBottleneck(
        embed_dim=embed_dim,
        hidden_ratio=bottleneck_hidden_ratio,
        drop=bottleneck_drop,
    )

    decoder = DinomalyDecoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_decoder_layers,
        num_tokens=num_decoder_tokens,
        mlp_ratio=decoder_mlp_ratio,
        drop=decoder_drop,
        attn_drop=decoder_attn_drop,
    )

    model = DinoAnomaly(
        encoder=encoder,
        inp_extractor=inp_extractor,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        remove_class_token=True,
        lambda_cohere=lambda_cohere,
        use_inp_fusion=use_inp_fusion,
    )
    return model


# ===========================================================================
# Shape self-check
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("model.py  shape self-check  (DinoAnomaly full pipeline)")
    print("=" * 60)

    DO_FULL_CHECK = False  # set True after encoder weights are downloaded

    if DO_FULL_CHECK:
        model = build_dino_anomaly(
            encoder_name="dinov2reg_vit_base_14",
            embed_dim=768,
            num_heads=12,
            num_inp=6,
        )
        model = model.cuda()
        model.eval()

        B, C_img, H, W = 2, 3, 448, 448
        dummy = torch.randn(B, C_img, H, W).cuda()

        with torch.no_grad():
            en, de, g_loss = model(dummy)

        print(f"\nInput         : ({B}, {C_img}, {H}, {W})")
        print(f"Num en maps   : {len(en)}")
        for idx, e in enumerate(en):
            print(f"  en[{idx}]    : {tuple(e.shape)}")
        print(f"Num de maps   : {len(de)}")
        for idx, d in enumerate(de):
            print(f"  de[{idx}]    : {tuple(d.shape)}")
        print(f"g_loss        : {g_loss.item():.4f}")

        expected_side = H // 14
        expected_C = 768
        assert en[0].shape == (B, expected_C, expected_side, expected_side)
        assert de[0].shape == (B, expected_C, expected_side, expected_side)
        assert g_loss.ndim == 0
        print("\n[OK] Full pipeline shapes correct.")

    else:
        print("Skipping full encoder load (set DO_FULL_CHECK=True after weights).")
        print()
        print("--- Module-level shape trace (no encoder weights) ---")
        B, N, C = 2, 1024, 768    # ViT-B/14 @ 448px
        P = 6                      # INP prototypes
        num_tokens = 1024          # learnable K tokens (must equal N = 32×32)

        # Simulate encoder output
        en_list_sim = [torch.randn(B, N, C) for _ in range(8)]          # 8 target layers
        x_fused = torch.stack(en_list_sim, dim=1).mean(dim=1)            # (B, N, C)
        print(f"  encoder: {len(en_list_sim)} layers, each ({B}, {N}, {C})")
        print(f"    fused → {tuple(x_fused.shape)}  (B, N, C)")

        # INP Extractor
        from inp_extractor import INPExtractor
        inp_ext = INPExtractor(embed_dim=C, num_heads=12, num_inp=P)
        inp = inp_ext(x_fused)                                           # (B, P, C)
        print(f"  inp_extractor: ({B}, {N}, {C}) → ({B}, {P}, {C})")

        # Coherence loss
        from losses import inp_coherence_loss as coh_loss
        g_loss, dist, _ = coh_loss(x_fused, inp)
        print(f"  g_loss: {g_loss.item():.4f}  dist: {tuple(dist.shape)}  (B, N, P)")

        # --- INP Fusion trace --------------------------------------------------
        print(f"\n--- INP Fusion (cosine-sim + softmax + Linear proj) ---")

        # Simulate the fusion step
        sim = F.cosine_similarity(
            x_fused.unsqueeze(2), inp.unsqueeze(1), dim=-1,
        )  # (B, N, P)
        attn = F.softmax(sim, dim=-1)                              # (B, N, P)
        inp_agg = attn @ inp                                          # (B, N, C)
        print(f"  cosine_sim   : {tuple(sim.shape)}   (B, N={N}, P={P})")
        print(f"  softmax attn : {tuple(attn.shape)}  row-sum ≈ "
              f"{attn.sum(-1).mean().item():.4f}  (expected 1.0)")
        print(f"  aggregated   : {tuple(inp_agg.shape)}   (B, N, C)")

        # Linear projection
        proj = nn.Linear(C, C, bias=False)
        fused = x_fused + proj(inp_agg)                               # (B, N, C)
        print(f"  after proj+add: {tuple(fused.shape)}   (B, N, C)")
        assert fused.shape == x_fused.shape
        assert attn.shape == (B, N, P)

        # --- [OK] fusion trace complete ---
        print("  [OK] INP fusion shapes correct.")

        # Bottleneck
        from bottleneck import MLPBottleneck
        bn = MLPBottleneck(embed_dim=C)
        x_bn = bn(fused)                                                # (B, N, C)
        print(f"\n  bottleneck: {tuple(x_bn.shape)}")

        # Decoder
        from decoder import DinomalyDecoder
        dec = DinomalyDecoder(embed_dim=C, num_heads=12, num_layers=8)
        de_list = dec(x_bn, inp)                                          # List[(B,N,C)]
        print(f"  decoder [{len(de_list)} layers]: each ({B}, {N}, {C})")

        # Fuse + reshape
        de_rev = de_list[::-1]
        de_fused = torch.stack(de_rev, dim=1).mean(dim=1)
        side = int(math.sqrt(N))
        de_map = de_fused.permute(0, 2, 1).reshape(B, C, side, side)     # (B, C, 32, 32)
        en_map = x_fused.permute(0, 2, 1).reshape(B, C, side, side)      # (B, C, 32, 32)
        print(f"  en_map: {tuple(en_map.shape)}  (B, C, H, W)")
        print(f"  de_map: {tuple(de_map.shape)}  (B, C, H, W)")

        assert en_map.shape == (B, C, side, side)
        assert de_map.shape == (B, C, side, side)
        assert g_loss.ndim == 0

        # --- verify forward pipeline can run with training ---
        print("\n--- Training-mode trace (gradient check) ---")
        from inp_extractor import INPExtractor as _IE
        from bottleneck import MLPBottleneck as _BN
        from decoder import DinomalyDecoder as _DD
        from losses import reconstruction_loss, total_loss as _total_loss

        # ---- Test both fusion paths -----------------------------------------
        for use_fusion in [True, False]:
            label = "ON" if use_fusion else "OFF"
            print(f"\n  use_inp_fusion={label}:")
            model = DinoAnomaly(
                encoder=None,
                inp_extractor=_IE(embed_dim=C, num_heads=12, num_inp=P),
                bottleneck=_BN(embed_dim=C),
                decoder=_DD(embed_dim=C, num_heads=12, num_layers=8, num_tokens=num_tokens),
                lambda_cohere=0.2,
                use_inp_fusion=use_fusion,
            )

            class FakeEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, x):
                    Bx = x.shape[0]
                    layers = [torch.randn(Bx, N, C) for _ in range(8)]
                    return layers, layers[-1]
                def train(self, mode: bool = True):
                    # Mirror DinoEncoder — always stays in eval
                    return super().train(False)
            model.encoder = FakeEncoder()

            dummy_img = torch.randn(B, 3, 448, 448)
            en, de, g = model(dummy_img)

            x_fused2 = torch.stack([torch.randn(B, N, C) for _ in range(8)], dim=1).mean(dim=1)
            total, info = _total_loss(en, de, model.inp_extractor(x_fused2), x_fused2,
                                      lambda_cohere=model.lambda_cohere)
            total.backward()

            trainable_has_grad = []
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    trainable_has_grad.append(name)

            # Verify fusion_proj grad specifically
            fusion_grad = model.inp_fusion_proj.weight.grad is not None
            fusion_status = "has_grad" if fusion_grad else "no_grad (expected if OFF)"

            print(f"    Trainable params with grad: {len(trainable_has_grad)}")
            print(f"    inp_fusion_proj.weight    : {fusion_status}")
            print(f"    Loss: total={info['loss_total']:.4f}  "
                  f"recon={info['loss_recon']:.4f}  cohere={info['loss_cohere']:.4f}")

            if use_fusion:
                assert fusion_grad, \
                    "inp_fusion_proj should receive gradients when fusion is ON!"

        # --- verify encoder stays in eval after model.train() ------------------
        print("\n--- Encoder freeze check: model.train() must keep encoder in eval ---")
        model.train()
        assert model.encoder.training is False, (
            f"Encoder training={model.encoder.training}! "
            "model.train() should NOT switch encoder to training mode."
        )
        print(f"  model.train() called → encoder.training = {model.encoder.training}  (expected: False)")
        model.eval()
        print("  [OK] Encoder stays in eval mode regardless of model.train().")

        print("\n[OK] Full pipeline shape trace passed.")


# #############################################################################
# DEPRECATED — 不可运行，仅供结构参考
#
# DinomalyBaseline 是原版 Dinomaly (ViTill) 的骨架封装。它依赖旧版 decoder 的
# Block 接口（单路 self-attention，无 INP），而当前 decoder.py 已整体切换为
# DecoderBlock（双路 attn1/attn2 + INP 交叉注意力），两者接口不兼容。
#
# 参看原版 Dinomaly 结构时可用此类作为索引，但**不要尝试实例化或训练**。
# #############################################################################

class DinomalyBaseline(nn.Module):
    """
    **[DEPRECATED — 不可运行，仅供结构参考]**

    Original Dinomaly baseline — uses single-attention Blocks (no INP).

    The current ``decoder.py`` uses ``DecoderBlock`` with INP cross-attention
    and is **not** compatible with this baseline.  This class is kept solely
    as an architectural reference for the original Dinomaly (ViTill) design.
    """

    def __init__(
        self,
        encoder: DinoEncoder,
        bottleneck: MLPBottleneck,
        decoder: DinomalyDecoder,
        target_layers: List[int] = None,
        fuse_layer_encoder: List[List[int]] = None,
        fuse_layer_decoder: List[List[int]] = None,
        remove_class_token: bool = True,
        mask_neighbor_size: int = 0,
    ):
        super().__init__()
        if target_layers is None:
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        if fuse_layer_encoder is None:
            fuse_layer_encoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
        if fuse_layer_decoder is None:
            fuse_layer_decoder = [[0, 1, 2, 3, 4, 5, 6, 7]]

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.mask_neighbor_size = mask_neighbor_size

        enc = self.encoder.encoder
        self.num_register_tokens = getattr(enc, "num_register_tokens", 0)

    def forward(self, x: torch.Tensor):
        """Raises RuntimeError — this class is NOT runnable."""
        raise RuntimeError(
            "DinomalyBaseline is DEPRECATED and NOT runnable. "
            "It depends on the old Block-based decoder interface which "
            "has been replaced by DecoderBlock with INP cross-attention. "
            "Use DinoAnomaly for all training and inference."
        )

    @staticmethod
    def _fuse_feature(feat_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def _generate_mask(self, feature_size: int, device=None) -> torch.Tensor:
        h = w = feature_size
        hm = wm = self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for ih in range(h):
            for iw in range(w):
                h0, h1 = max(ih - hm // 2, 0), min(ih + hm // 2 + 1, h)
                w0, w1 = max(iw - wm // 2, 0), min(iw + wm // 2 + 1, w)
                mask[ih, iw, h0:h1, w0:w1] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        total = h * w + 1 + self.num_register_tokens
        mask_all = torch.ones(total, total, device=device)
        offset = 1 + self.num_register_tokens
        mask_all[offset:, offset:] = mask
        return mask_all
