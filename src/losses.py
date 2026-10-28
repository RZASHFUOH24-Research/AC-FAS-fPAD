"""
AC-FAS losses — Section III-D/E/F of the paper.

- ManhattanCompactnessLoss: Eq. (manhattan_live), (manhattan_spoof), (lmh)
- attention_consistency_score C(x): Eq. (consistency)
- attention_consistency_loss_real L_AC^r: Eq. (lac_real)
- attention_margin_loss L_AM: Eq. (lam)
- total_objective: Eq. (total)

Hyperparameters: r = 2.0, m = 20.0, lambda_l = lambda_s = 1.0, alpha = 0.5,
beta = 0.3, lambda_m = 2.0, alpha_s = 0.6, beta_s = 0.4.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import normalize_batch
from model import attention_signature
from wfrf import wfrf_eval_batch


class ManhattanCompactnessLoss(nn.Module):
    """L1-norm hinge compactness loss around a *fixed* center c.

    Eq. (manhattan_live):  L_j^l = max(||z~_j^l - c||_1 - r, 0)
    Eq. (manhattan_spoof): L_j^s = max(m - ||z~_j^s - c||_1, 0)
    Eq. (lmh): L_MH = (lambda_l / N_l) * sum(L^l) + (lambda_s / N_s) * sum(L^s)

    The center `c` is computed once via `compute_warmup_center` (a warm-up
    pass over the real training samples) and then held fixed for the rest of
    training to prevent compactness collapse.
    """

    def __init__(self, center: torch.Tensor, r: float = 2.0, m: float = 20.0, lambda_l: float = 1.0, lambda_s: float = 1.0):
        super().__init__()
        self.register_buffer("center", center)
        self.r = r
        self.m = m
        self.lambda_l = lambda_l
        self.lambda_s = lambda_s

    def forward(self, projected: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask_live = labels == 1
        mask_spoof = labels == 0

        dist = torch.norm(projected - self.center, p=1, dim=1)  # (B,) L1 distance to center

        loss = projected.new_tensor(0.0)

        if mask_live.any():
            loss = loss + self.lambda_l * F.relu(dist[mask_live] - self.r).mean()

        if mask_spoof.any():
            loss = loss + self.lambda_s * F.relu(self.m - dist[mask_spoof]).mean()

        return loss


@torch.no_grad()
def compute_warmup_center(model, data_loader, device) -> torch.Tensor:
    """Warm-up pass over the real (live) training samples: evaluation mode,
    deterministic WFRF + ImageNet normalisation (the same input processing as
    training), no parameter updates. Returns the mean projected embedding,
    used as the fixed compactness center `c` (Section III-E)."""
    model.eval()
    total = None
    count = 0
    for images, labels in data_loader:
        real_mask = labels == 1
        if not real_mask.any():
            continue
        images = normalize_batch(wfrf_eval_batch(images[real_mask].to(device)))
        projected = model(images)
        batch_sum = projected.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        count += int(real_mask.sum())
    model.train()
    if count == 0:
        raise RuntimeError("Warm-up pass found no live/real samples to initialise the center.")
    return (total / count).detach()


def attention_consistency_score(attn_maps: List[torch.Tensor], batch_size: int) -> torch.Tensor:
    """Cross-layer attention consistency C(x) — Eq. (consistency).

    Args:
        attn_maps: list of M attention tensors (one per block, in block order),
            each (batch_size * n_windows, heads, N, N).
        batch_size: number of images in the batch.

    Returns:
        (batch_size,) tensor: mean KL divergence between the pooled attention
        signatures of consecutive blocks (M = 10 -> 9 KL terms).
    """
    signatures = [attention_signature(a, batch_size) for a in attn_maps]  # each (B, 49)
    M = len(signatures)
    assert M >= 2, "Need at least two attention blocks to compute cross-layer consistency."

    total = torch.zeros(batch_size, device=signatures[0].device)
    for m in range(M - 1):
        p = signatures[m].clamp_min(1e-8)
        q = signatures[m + 1].clamp_min(1e-8)
        # KL(p || q) = sum_i p_i * log(p_i / q_i), computed per-sample.
        total = total + (p * (p.log() - q.log())).sum(dim=-1)
    return total / (M - 1)


def attention_consistency_loss_real(c_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """L_AC^r = E_{x ~ D_r}[C(x)] — Eq. (lac_real). Only over live samples."""
    mask_live = labels == 1
    if not mask_live.any():
        return c_scores.new_tensor(0.0)
    return c_scores[mask_live].mean()


def attention_margin_loss(c_scores: torch.Tensor, labels: torch.Tensor, lambda_m: float = 2.0) -> torch.Tensor:
    """L_AM = E_{x^r, x^s}[max(0, lambda_m - C(x^s) + C(x^r))] — Eq. (lam).

    The expectation is taken over all live-spoof pairs in the mini-batch
    (pairwise hinge, then mean).
    """
    mask_live = labels == 1
    mask_spoof = labels == 0
    if not (mask_live.any() and mask_spoof.any()):
        return c_scores.new_tensor(0.0)

    gap = lambda_m - c_scores[mask_spoof].unsqueeze(0) + c_scores[mask_live].unsqueeze(1)  # (n_live, n_spoof)
    return F.relu(gap).mean()


def total_objective(mh_loss: torch.Tensor, lac_real: torch.Tensor, lam: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """L_total = L_MH + alpha * L_AC^r + beta * L_AM — Eq. (total).

    `alpha` follows the epoch-dependent ramp schedule (see train.py);
    `beta` is a fixed 0.3 throughout training.
    """
    return mh_loss + alpha * lac_real + beta * lam
