"""
AC-FAS losses — Section III-D/E/F of the paper.

- ManhattanCompactnessLoss: Eq. (manhattan_live), (manhattan_spoof), (lmh)
- attention_consistency_score C(x): Eq. (consistency)
- attention_consistency_loss_real L_AC^r: Eq. (lac_real)
- attention_margin_loss L_AM: Eq. (lam)
- total_objective: Eq. (total)

Paper-confirmed hyperparameters (do not change without re-checking the
manuscript): r = 2.0, m = 20.0, lambda_l = lambda_s = 1.0, alpha = 0.5,
beta = 0.3, lambda_m = 2.0, alpha_s = 0.6, beta_s = 0.4.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import attention_signature


class ManhattanCompactnessLoss(nn.Module):
    """L1-norm hinge compactness loss around a *fixed* center c.

    Eq. (manhattan_live):  L_j^l = max(||z~_j^l - c||_1 - r, 0)
    Eq. (manhattan_spoof): L_j^s = max(m - ||z~_j^s - c||_1, 0)
    Eq. (lmh): L_MH = (lambda_l / N_l) * sum(L^l) + (lambda_s / N_s) * sum(L^s)

    The center `c` must be computed once via `compute_warmup_center` (a
    warm-up pass over all real training samples) and then held fixed for
    the rest of training (paper: "initialised ... in a warm-up pass and
    fixed throughout training to prevent compactness collapse").
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

        loss = torch.tensor(0.0, device=projected.device)

        if mask_live.any():
            live_term = F.relu(dist[mask_live] - self.r)
            loss = loss + self.lambda_l * live_term.sum() / mask_live.sum()

        if mask_spoof.any():
            spoof_term = F.relu(self.m - dist[mask_spoof])
            loss = loss + self.lambda_s * spoof_term.sum() / mask_spoof.sum()

        return loss


@torch.no_grad()
def compute_warmup_center(model, data_loader, device) -> torch.Tensor:
    """Runs a warm-up forward pass over all real (live) training samples and
    returns the mean projected embedding, used as the fixed compactness
    center `c` (Section III-E)."""
    model.eval()
    total = None
    count = 0
    for images, labels in data_loader:
        real_mask = labels == 1
        if not real_mask.any():
            continue
        images = images[real_mask].to(device)
        projected = model(images)
        batch_sum = projected.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        count += real_mask.sum().item()
    model.train()
    if count == 0:
        raise RuntimeError("Warm-up pass found no live/real samples to initialise the center.")
    return (total / count).detach()


def attention_consistency_score(attn_maps: List[torch.Tensor]) -> torch.Tensor:
    """Cross-layer attention consistency C(x) — Eq. (consistency).

    Args:
        attn_maps: list of M tensors, each (B, heads, N, N) softmax
            attention weights from a selected layer, in layer order.

    Returns:
        (B,) tensor: average KL divergence between consecutive layers'
        attention signatures.
    """
    signatures = [attention_signature(a) for a in attn_maps]  # each (B, N)
    M = len(signatures)
    assert M >= 2, "Need at least two selected layers to compute cross-layer consistency."

    total = torch.zeros(signatures[0].shape[0], device=signatures[0].device)
    for m in range(M - 1):
        p = signatures[m].clamp_min(1e-8)
        q = signatures[m + 1].clamp_min(1e-8)
        # KL(p || q) = sum_i p_i * log(p_i / q_i), computed per-sample.
        kl = (p * (p.log() - q.log())).sum(dim=-1)
        total = total + kl
    return total / (M - 1)


def attention_consistency_loss_real(c_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """L_AC^r = E_{x ~ D_r}[C(x)] — Eq. (lac_real). Only over live samples."""
    mask_live = labels == 1
    if not mask_live.any():
        return torch.tensor(0.0, device=c_scores.device)
    return c_scores[mask_live].mean()


def attention_margin_loss(c_scores: torch.Tensor, labels: torch.Tensor, lambda_m: float = 2.0) -> torch.Tensor:
    """L_AM = E[max(0, lambda_m - C(x^s) + C(x^r))] — Eq. (lam).

    Implemented at the mini-batch level using the mean consistency score of
    the live and spoof samples present in the batch.
    """
    mask_live = labels == 1
    mask_spoof = labels == 0
    if not (mask_live.any() and mask_spoof.any()):
        return torch.tensor(0.0, device=c_scores.device)

    c_real = c_scores[mask_live].mean()
    c_spoof = c_scores[mask_spoof].mean()
    return F.relu(lambda_m - c_spoof + c_real)


def total_objective(mh_loss: torch.Tensor, lac_real: torch.Tensor, lam: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """L_total = L_MH + alpha * L_AC^r + beta * L_AM — Eq. (total).

    `alpha` follows the epoch-dependent ramp schedule (see train.py);
    `beta` is a fixed 0.3 throughout training (Section III-D).
    """
    return mh_loss + alpha * lac_real + beta * lam
