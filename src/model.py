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
# Which layers to use for the cross-layer attention-consistency score C(x).
#
# The paper defines C(x) generically over a set of selected layers
# L = {l_1, ..., l_M}, without pinning down M or the exact layers in the
# main text, but its KL-divergence formula requires every selected layer's
# attention signature a^l to live in the SAME simplex Delta^{N-1} — i.e.
# every selected layer must have the same number of spatial tokens N.
#
# This matters because it is NOT true of arbitrary layer pairs in TinyViT-5M,
# for two independent reasons verified against the actual installed timm
# implementation:
#
# 1) Token count N differs by stage:
#      stage 0: no attention (conv/MBConv blocks)
#      stage 1: N = 49   (7x7, WINDOWED attention, 2 blocks)
#      stage 2: N = 196  (14x14, WINDOWED attention, 6 blocks)
#      stage 3: N = 49   (7x7, GLOBAL attention, 2 blocks)
#    An earlier draft of this pipeline hooked stage 2 and stage 3 directly
#    and reconciled the N=196 vs N=49 mismatch by truncating both
#    signatures to their first 49 entries before taking a divergence — this
#    is NOT a valid KL divergence between two distributions over the same
#    support, and does not match the paper's Delta^{N-1} formulation.
#
# 2) Windowed stages fold the (batch, num_windows) dims together before
#    calling attention: a captured attention tensor from a windowed layer
#    has shape (B * num_windows, heads, N, N), not (B, heads, N, N) — so
#    even a windowed layer with a *matching* N is not directly comparable,
#    batch-index for batch-index, to a non-windowed layer's output without
#    extra unwindowing logic.
#
# Both issues are avoided by selecting two blocks WITHIN the final stage
# (stage 3), which performs global (non-windowed) self-attention over the
# whole 7x7=49-token feature map — Fig. 1 explicitly ties the pipeline's
# selected layers to this global-attention stage. We use stage 3's first
# and last block, both with clean (B, heads, 49, 49) output and no window
# folding. (Note: the paper's Fig. 1 caption states this stage produces
# "14x14 = 196" tokens, which does not match what we measure here (49) —
# this looks like an artifact of the manuscript text rather than the
# actual architecture. If your own training run used a different layer
# pair — e.g. a different input resolution that changes the final grid to
# 14x14 — update ATTENTION_HOOK_PATHS to match your setup.)
#
# If you use more than 2 layers, every entry in ATTENTION_HOOK_PATHS must
# still resolve to matching N *and* matching (non-windowed) batch
# semantics — the rest of the pipeline (attention_signature,
# attention_consistency_score) generalises to any number of layers >= 2 as
# long as that holds.
# --------------------------------------------------------------------------
ATTENTION_HOOK_PATHS = [
    "stages.3.blocks.0.attn",  # first block of stage 3 (global attention, N=49)
    "stages.3.blocks.-1.attn",  # last block of stage 3 (global attention, N=49)
]

PROJECTED_EMBEDDING_DIM = 128  # p in the paper (Section III-E)
BACKBONE_EMBEDDING_DIM = 320  # d = 320, TinyViT-5M CLS/pooled feature dim


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
# timm's TinyViT `Attention.forward` computes the attention weights as a
# purely *local* variable (`attn`), and by default takes a fused
# scaled-dot-product-attention path (`F.scaled_dot_product_attention`) that
# never materialises an explicit (B, heads, N, N) attention-probability
# tensor at all — so a plain forward hook cannot recover it (verified
# against timm's tiny_vit.py source directly).
#
# We therefore monkey-patch each selected Attention module's `forward` with
# an equivalent implementation (mirroring timm's non-fused branch exactly)
# that additionally stashes the post-softmax attention weights on the
# module as `self.last_attn_weights`, of shape (B, num_heads, N, N).
#
# This depends on TinyViT's internal attribute names (`norm`, `qkv`, `scale`,
# `num_heads`, `key_dim`, `val_dim`, `out_dim`, `proj`,
# `get_attention_biases`, `attention_bias_idxs`) as of timm's tiny_vit
# implementation. If you upgrade timm and this breaks, run
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

    self.last_attn_weights = attn  # (B, num_heads, N, N) -- captured for C(x)

    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, self.out_dim)
    x = self.proj(x)
    return x


class AttentionMapExtractor:
    """Patches the selected attention submodules to record their post-softmax
    attention-weight tensors on every forward pass, and exposes them via
    `get()`. Call `clear()` before each forward pass and `remove()` when
    done to restore the original (fused-capable) forward methods."""

    def __init__(self, backbone: nn.Module, hook_paths=ATTENTION_HOOK_PATHS):
        self.hook_paths = hook_paths
        self._modules = []
        self._original_forwards = []

        for path in hook_paths:
            attn_module = _resolve_module(backbone, path)
            required = ("qkv", "norm", "proj", "get_attention_biases", "num_heads", "key_dim", "val_dim", "scale")
            missing = [r for r in required if not hasattr(attn_module, r)]
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

    def get(self, hook_paths=None):
        """Returns captured attention maps (detached) in hook order."""
        return [m.last_attn_weights.detach() if m.last_attn_weights is not None else None for m in self._modules]

    def remove(self):
        for module, original_forward in self._original_forwards:
            module.forward = original_forward
        self._modules = []
        self._original_forwards = []


def attention_signature(attn_weights: torch.Tensor) -> torch.Tensor:
    """Computes the per-layer attention signature a^l.

    Args:
        attn_weights: (B, heads, N, N) softmax attention probabilities.

    Returns:
        (B, N) tensor, each row a probability distribution over the N
        spatial tokens (lies on the simplex Delta^{N-1}).
    """
    head_avg = attn_weights.mean(dim=1)  # (B, N, N) -- average over heads
    signature = head_avg.mean(dim=1)  # (B, N) -- average over query tokens
    # Each row of head_avg already sums to 1 (softmax outputs); the mean of
    # valid distributions is itself a valid distribution, so no renormalisation
    # is strictly required, but we normalise defensively against numerical drift.
    signature = signature / signature.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return signature


class ProjectionHead(nn.Module):
    """phi(z) = LayerNorm(W2 * sigma(W1*z + b1) + b2) — Section III-E."""

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
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.layer_norm(x)


class ACFASModel(nn.Module):
    """TinyViT-5M backbone + projection head. Attention maps are captured
    separately via `AttentionMapExtractor` (attached to `self.backbone`)."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tiny_vit_5m_224", pretrained=True, num_classes=0)
        self.head = ProjectionHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (B, 320)
        return self.head(features)  # (B, 128)
