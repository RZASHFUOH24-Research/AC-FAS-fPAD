"""
AC-FAS training — implements Algorithm 1 of the paper (initialisation,
per-batch WFRF mode selection, forward pass, loss computation, optimisation).

No CLI arguments by design: edit the CONFIG block below directly, exactly
like the earlier project scripts in this codebase.
"""

import os

import torch
import torch.optim as optim
import tqdm

from dataset import build_train_loader
from losses import (
    ManhattanCompactnessLoss,
    attention_consistency_loss_real,
    attention_consistency_score,
    attention_margin_loss,
    compute_warmup_center,
    total_objective,
)
from model import ACFASModel, AttentionMapExtractor, ATTENTION_HOOK_PATHS
from wfrf import wfrf_train_batch

# ==========================================
# CONFIG — edit these directly, no argparse
# ==========================================
TRAIN_FOLDER_REAL = r"..\..\Real"
TRAIN_FOLDER_SPOOF = [r"..\..\Spoof"]  # [] for pure one-class training

SAVE_DIR = r"..\checkpoints\run1"

BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 1e-4
LR_MILESTONES = [60, 85]
LR_GAMMA = 0.1

# Manhattan compactness loss (Section III-E) — paper-confirmed values.
R = 2.0
M = 20.0
LAMBDA_L = 1.0
LAMBDA_S = 1.0

# Attention consistency / margin loss (Section III-D) — paper-confirmed values.
ALPHA_ATTN_MAX = 0.5  # alpha ramps 0 -> 0.5
ATTN_START_EPOCH = 10  # alpha stays 0 for the first 10 epochs
ATTN_RAMP_EPOCHS = 10  # then linearly ramps to ALPHA_ATTN_MAX over the next 10 epochs
BETA_AM = 0.3  # beta is fixed (no ramp) from epoch 1
LAMBDA_M = 2.0  # margin in the attention-margin loss L_AM

# WFRF stochastic-augmentation probability (Section III-D): p_aug = 0.5.
P_AUG = 0.5

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def alpha_schedule(epoch: int) -> float:
    """Epoch-dependent ramp for alpha (Section III-D implementation details)."""
    if epoch < ATTN_START_EPOCH:
        return 0.0
    return min(ALPHA_ATTN_MAX, (epoch - ATTN_START_EPOCH) * (ALPHA_ATTN_MAX / ATTN_RAMP_EPOCHS))


def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(batch.device)
    std = IMAGENET_STD.to(batch.device)
    return (batch - mean) / std


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_loader = build_train_loader(TRAIN_FOLDER_REAL, TRAIN_FOLDER_SPOOF, BATCH_SIZE)

    model = ACFASModel().to(device)
    attn_extractor = AttentionMapExtractor(model.backbone)

    print("Running warm-up pass to initialise the fixed compactness center c ...")
    center = compute_warmup_center(model, train_loader, device)
    criterion = ManhattanCompactnessLoss(center, r=R, m=M, lambda_l=LAMBDA_L, lambda_s=LAMBDA_S).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)

    print(f"\n{'=' * 55}")
    print(f"Training started | r={R} m={M} lr={LR} p_aug={P_AUG}")
    print(f"{'=' * 55}\n")

    best_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        alpha = alpha_schedule(epoch)

        total_loss = 0.0
        pbar = tqdm.tqdm(train_loader, desc=f"Ep {epoch:03d}/{NUM_EPOCHS} [alpha:{alpha:.3f}]", leave=False)

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # --- WFRF: applied once per mini-batch, per Algorithm 1 ---
            wfrf_images, mode, w_low, w_high = wfrf_train_batch(images, p_aug=P_AUG)
            wfrf_images = normalize_batch(wfrf_images)

            optimizer.zero_grad()
            attn_extractor.clear()

            projected = model(wfrf_images)
            attn_maps = attn_extractor.get(ATTENTION_HOOK_PATHS)
            if any(m is None for m in attn_maps):
                raise RuntimeError(
                    "Attention maps were not captured. Check AttentionMapExtractor / "
                    "ATTENTION_HOOK_PATHS in model.py against your installed timm version."
                )

            c_scores = attention_consistency_score(attn_maps)

            mh_loss = criterion(projected, labels)
            lac_real = attention_consistency_loss_real(c_scores, labels)
            lam = attention_margin_loss(c_scores, labels, lambda_m=LAMBDA_M)

            loss = total_objective(mh_loss, lac_real, lam, alpha=alpha, beta=BETA_AM)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "WFRF": mode})

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "center": criterion.center},
                os.path.join(SAVE_DIR, "model_best.pth"),
            )

        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | alpha: {alpha:.3f}")
        torch.save(
            {"model_state_dict": model.state_dict(), "center": criterion.center},
            os.path.join(SAVE_DIR, f"model_ep{epoch:03d}.pth"),
        )

    attn_extractor.remove()
    print(f"\nFinished! Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()
