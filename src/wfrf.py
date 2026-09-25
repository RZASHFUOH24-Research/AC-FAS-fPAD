"""
Weighted Frequency Response Filtering (WFRF) — AC-FAS paper, Section III-B.

Implements Eqs. (2)-(6):
    F(u,v)      = FFT2(x)                                      (Eq. 2)
    d(u,v)      = max(|u - u_c|, |v - v_c|)   [Chebyshev/square]  (Eq. 3)
    W(u,v)      = w_low + (w_high - w_low) * d(u,v) / d_max     (Eq. 4)
    F'(u,v)     = F(u,v) * W(u,v)                                (Eq. 5)
    x'          = Re(IFFT2(F'))                                  (Eq. 6)

IMPORTANT — this is a *square* (Chebyshev) distance weighting, not a radial
(Euclidean) one. This is a deliberate correction relative to an earlier,
draft implementation that used a Euclidean distance grid; the paper is
explicit that the mask is square-based (Eq. 3).

Per Algorithm 1, WFRF is applied to *every* image (never skipped), and its
mode is selected once per mini-batch:
    - with probability p_aug: STOCHASTIC AUGMENTATION — w_low, w_high are
      independently resampled from U(0.5, 2.0) for this batch.
    - otherwise: DETERMINISTIC PREPROCESSING — fixed w_low=0.8, w_high=1.75.
At inference, only the deterministic mode (w_low=0.8, w_high=1.75) is used.

WFRF is intentionally implemented as a *batch-level* operation (applied once
per mini-batch, after collation, on images still in [0, 1] range i.e. before
ImageNet normalisation) rather than inside the per-sample Dataset transform,
because the paper's mode selection happens once per mini-batch, not once per
sample.
"""

import math

import torch

# Paper-confirmed deterministic weights (Section III-D, Implementation Details).
DETERMINISTIC_W_LOW = 0.8
DETERMINISTIC_W_HIGH = 1.75

# Stochastic augmentation sampling range (Algorithm 1).
STOCHASTIC_RANGE = (0.5, 2.0)

# Probability of using the stochastic-augmentation mode for a given
# mini-batch during training (Section III-D: p_aug = 0.5).
P_AUG = 0.5


def _chebyshev_weight_matrix(h: int, w: int, w_low: float, w_high: float, device, dtype) -> torch.Tensor:
    """Builds the (H, W) frequency weighting matrix W(u,v) from Eqs. (3)-(4)."""
    u = torch.arange(h, device=device, dtype=dtype) - h // 2
    v = torch.arange(w, device=device, dtype=dtype) - w // 2
    uu, vv = torch.meshgrid(u, v, indexing="ij")

    # Chebyshev (square) distance from the spectrum centre — Eq. (3).
    dist = torch.maximum(uu.abs(), vv.abs())
    d_max = max(h // 2, w // 2)

    # Linear ramp from w_low (centre / low freq) to w_high (edges / high freq) — Eq. (4).
    weights = w_low + (w_high - w_low) * (dist / d_max)
    return weights


def apply_wfrf(batch: torch.Tensor, w_low: float, w_high: float) -> torch.Tensor:
    """Applies WFRF to a batch of images.

    Args:
        batch: (B, C, H, W) tensor in [0, 1].
        w_low, w_high: frequency-weighting hyperparameters for this call.

    Returns:
        (B, C, H, W) tensor in [0, 1], frequency-reweighted.
    """
    B, C, H, W = batch.shape
    weights = _chebyshev_weight_matrix(H, W, w_low, w_high, batch.device, batch.dtype)
    weights = weights.view(1, 1, H, W)  # broadcast over batch and channel

    fft = torch.fft.fft2(batch)
    fshift = torch.fft.fftshift(fft, dim=(-2, -1))

    weighted = fshift * weights

    out = torch.fft.ifft2(torch.fft.ifftshift(weighted, dim=(-2, -1)))
    return torch.clamp(torch.abs(out), 0.0, 1.0)


def wfrf_train_batch(batch: torch.Tensor, p_aug: float = P_AUG):
    """Applies WFRF to a training mini-batch under Algorithm 1's dual-mode rule.

    With probability `p_aug`, uses the stochastic-augmentation mode (fresh
    w_low, w_high ~ U(0.5, 2.0) for this batch); otherwise uses the fixed
    deterministic weights. Returns (transformed_batch, mode, w_low, w_high)
    for logging purposes.
    """
    if torch.rand(1).item() < p_aug:
        w_low = float(torch.empty(1).uniform_(*STOCHASTIC_RANGE))
        w_high = float(torch.empty(1).uniform_(*STOCHASTIC_RANGE))
        mode = "stochastic"
    else:
        w_low, w_high = DETERMINISTIC_W_LOW, DETERMINISTIC_W_HIGH
        mode = "deterministic"

    return apply_wfrf(batch, w_low, w_high), mode, w_low, w_high


def wfrf_eval_batch(batch: torch.Tensor) -> torch.Tensor:
    """Applies the fixed deterministic WFRF used at inference / evaluation."""
    return apply_wfrf(batch, DETERMINISTIC_W_LOW, DETERMINISTIC_W_HIGH)
