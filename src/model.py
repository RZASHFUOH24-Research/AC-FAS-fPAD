"""
AC-FAS model: TinyViT-5M feature extractor + projection head + cross-layer
attention-map extraction used for the attention-consistency score C(x).

Paper references: Section III-C (Feature Extractor and Attention
Representation) and Section III-E (Manhattan-Norm L1 Compactness).
"""

import types

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Layers used for the cross-layer attention-consistency score C(x).
#
# C(x) is computed over ALL M = 10 attention blocks of TinyViT-5M
# (stage 0 is convolutional; stages 1-3 contain 2 + 6 + 2 blocks), as the
# mean KL divergence between the attention signatures of consecutive blocks
# (9 KL terms).
#
# For a 224x224 input the three transformer stages work on token grids of
# 28x28, 14x14 and 7x7. Stage 1 uses windowed attention (7x7 windows ->
# 16 windows of 49 tokens, folded into the batch dimension); in stages 2 and
# 3 the window covers the whole grid, so attention is effectively global.
#
# Every block's signature is therefore placed back on its full token grid and
# mapped to the common 7x7 grid by mass-preserving sum pooling, so that every
# KL term compares two distributions on the same simplex Delta^48.
# --------------------------------------------------------------------------
COMMON_GRID = 7  # common spatial grid of the pooled signatures (N = 49)

PROJECTED_EMBEDDING_DIM = 128  # p in the paper (Section III-E)
BACKBONE_EMBEDDING_DIM = 320  # d = 320, TinyViT-5M globally pooled feature dim


def all_attention_paths(backbone: nn.Module):
    """Paths of ALL attention blocks of TinyViT-5M: stages 1..3 (stage 0 is convolutional)."""
    paths = []
    for s in range(1, len(backbone.stages)):
        for b in range(len(backbone.stages[s].blocks)):
            paths.append(f"stages.{s}.blocks.{b}.attn")
    return paths


def _resolve_module(root: nn.Module, dotted_path: str) -> nn.Module:
    """Resolves a dotted path like 'stages.2.blocks.-1.attn' to a submodule,
    supporting negative indices for the last element of a list/Sequential."""
    module = root
    for part in dotted_path.split("."):
        if part.lstrip("-").isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


# --------------------------------------------------------------------------
# Attention-weight capture.
#
# timm's TinyViT `Attention.forward` may take a fused
# scaled-dot-product-attention path that never materialises the attention
# probabilities, so a plain forward hook cannot recover them. Each attention
# module's `forward` is therefore replaced with an equivalent explicit
# implementation (mirroring timm's non-fused branch) that additionally stores
# the post-softmax attention weights as `self.last_attn_weights`, of shape
# (B * num_windows, num_heads, N, N).
#
# This depends on TinyViT's internal attribute names (`norm`, `qkv`, `scale`,
# `num_heads`, `key_dim`, `val_dim`, `out_dim`, `proj`,
# `get_attention_biases`). If you upgrade timm and this breaks, run
# `import inspect, timm; from timm.models import tiny_vit;
# print(inspect.getsource(tiny_vit.Attention.forward))` and update
# `_patched_attention_forward` below to match.
# --------------------------------------------------------------------------
def _patched_attention_forward(self, x):
    attn_bias = self.get_attention_biases(x.device)
    B, N, _ = x.shape
    x = self.norm(x)
    qkv = self.qkv(x)
    q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn + attn_bias
    attn = attn.softmax(dim=-1)

    self.last_attn_weights = attn  # (B * n_windows, num_heads, N, N) -- captured for C(x)

    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, self.out_dim)
    x = self.proj(x)
    return x


class AttentionMapExtractor:
    """Patches the attention submodules (all blocks by default) to record their
    post-softmax attention-weight tensors on every forward pass, and exposes
    them via `get()`. Call `clear()` before each forward pass and `remove()`
    when done to restore the original forward methods."""

    REQUIRED = ("qkv", "norm", "proj", "get_attention_biases", "num_heads", "key_dim", "val_dim", "scale", "out_dim")

    def __init__(self, backbone: nn.Module, hook_paths=None):
        self.hook_paths = hook_paths if hook_paths is not None else all_attention_paths(backbone)
        self._modules = []
        self._original_forwards = []

        for path in self.hook_paths:
            attn_module = _resolve_module(backbone, path)
            missing = [r for r in self.REQUIRED if not hasattr(attn_module, r)]
            if missing:
                raise AttributeError(
                    f"Module at '{path}' is missing expected TinyViT-Attention attributes {missing}. "
                    f"Your installed timm version's TinyViT implementation may differ — inspect "
                    f"tiny_vit.Attention.forward's source and update _patched_attention_forward() "
                    f"in model.py accordingly."
                )
            self._original_forwards.append((attn_module, attn_module.forward))
            attn_module.last_attn_weights = None
            attn_module.forward = types.MethodType(_patched_attention_forward, attn_module)
            self._modules.append(attn_module)

    def clear(self):
        for m in self._modules:
            m.last_attn_weights = None

    def get(self):
        """Returns the captured attention maps in block order.

        NOT detached: during training the gradients of L_AC and L_AM must flow
        through the attention maps into the backbone."""
        maps = [m.last_attn_weights for m in self._modules]
        if any(a is None for a in maps):
            raise RuntimeError(
                "Attention maps were not captured. Call clear() before the forward pass and check "
                "AttentionMapExtractor against your installed timm version."
            )
        return maps

    def remove(self):
        for module, original_forward in self._original_forwards:
            module.forward = original_forward
        self._modules = []
        self._original_forwards = []


def attention_signature(attn_weights: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Computes the pooled attention signature of one block on the common grid.

    Args:
        attn_weights: (batch_size * n_windows, heads, N, N) softmax attention
            probabilities (n_windows = 1 for global attention).
        batch_size: number of images in the batch.

    Steps:
        1) mean over heads, then mean over query tokens -> a distribution over
           the N tokens of each window;
        2) windows are placed back on the full token grid and averaged (each
           window contributes one distribution);
        3) the grid is sum-pooled to COMMON_GRID x COMMON_GRID (mass-preserving)
           and renormalised against numerical drift.

    Returns:
        (batch_size, COMMON_GRID ** 2) tensor, each row a probability
        distribution on the simplex Delta^48.
    """
    bw, _, n, _ = attn_weights.shape
    n_win = bw // batch_size
    ws = int(round(n ** 0.5))  # window side
    nh = int(round(n_win ** 0.5))  # windows per side
    assert bw == batch_size * n_win and ws * ws == n and nh * nh == n_win, \
        f"Unexpected attention shape {tuple(attn_weights.shape)} for batch_size={batch_size}"

    s = attn_weights.mean(dim=1).mean(dim=1)  # (B * n_win, N)
    s = s.view(batch_size, nh, nh, ws, ws).permute(0, 1, 3, 2, 4).reshape(batch_size, 1, nh * ws, nh * ws)
    s = s / (nh * nh)  # windows -> one distribution over the full grid

    grid = nh * ws
    assert grid % COMMON_GRID == 0, f"Token grid {grid}x{grid} is not divisible by {COMMON_GRID}"
    k = grid // COMMON_GRID
    if k > 1:
        s = F.avg_pool2d(s, k) * (k * k)  # sum pooling
    s = s.flatten(1)
    return s / s.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class ProjectionHead(nn.Module):
    """phi(z) = LayerNorm(W2 * Dropout(ReLU(BN(W1*z + b1))) + b2) — Section III-E."""

    def __init__(self, in_dim: int = BACKBONE_EMBEDDING_DIM, hidden_dim: int = 512, out_dim: int = PROJECTED_EMBEDDING_DIM, dropout: float = 0.4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc1(z)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.layer_norm(x)


class ACFASModel(nn.Module):
    """TinyViT-5M backbone + projection head. Attention maps are captured
    separately via `AttentionMapExtractor` (attached to `self.backbone`)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("tiny_vit_5m_224", pretrained=pretrained, num_classes=0)
        self.head = ProjectionHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (B, 320)
        return self.head(features)  # (B, 128)
