"""
AC-FAS training — implements Algorithm 1 of the paper (warm-up centre,
per-batch WFRF mode selection, forward pass, loss computation, optimisation).
Saves a checkpoint after every epoch.

No CLI arguments by design: edit the CONFIG block below directly, exactly
like the earlier project scripts in this codebase.

Strict one-class run: set TRAIN_FOLDER_SPOOF = [] (spoof-dependent terms are
then removed: LAMBDA_S = 0, BETA_AM = 0).
"""

import os
import random

import numpy as np
import torch
import torch.optim as optim
import tqdm

from dataset import build_train_loader, normalize_batch
from losses import (
    ManhattanCompactnessLoss,
    attention_consistency_loss_real,
    attention_consistency_score,
    attention_margin_loss,
    compute_warmup_center,
    total_objective,
)
from model import ACFASModel, AttentionMapExtractor
from wfrf import wfrf_train_batch

# ==========================================
# CONFIG — edit these directly, no argparse
# ==========================================
TRAIN_FOLDER_REAL = r"..\..\Real"
TRAIN_FOLDER_SPOOF = [r"..\..\Spoof"]  # [] for strict one-class training (live only)

SAVE_DIR = r"..\checkpoints\run1"

SEED = 42  # results are reported over five runs with different seeds

BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 1e-4
LR_MILESTONES = [60, 85]
LR_GAMMA = 0.1
GRAD_CLIP = 1.0

# Manhattan compactness loss (Section III-E).
R = 2.0
M = 20.0
LAMBDA_L = 1.0
LAMBDA_S = 1.0

# Attention consistency / margin loss (Section III-D).
ALPHA_ATTN_MAX = 0.5  # alpha ramps 0 -> 0.5
ATTN_START_EPOCH = 10  # alpha stays 0 up to epoch 10
ATTN_RAMP_EPOCHS = 10  # then linearly ramps to ALPHA_ATTN_MAX over the next 10 epochs
BETA_AM = 0.3  # beta is fixed (no ramp) from epoch 1
LAMBDA_M = 2.0  # margin in the attention-margin loss L_AM

# WFRF stochastic-augmentation probability: p_aug = 0.5.
P_AUG = 0.5

USE_SPOOF = len(TRAIN_FOLDER_SPOOF) > 0
if not USE_SPOOF:  # strict one-class: remove every spoof-dependent term
    LAMBDA_S, BETA_AM = 0.0, 0.0


def alpha_schedule(epoch: int) -> float:
    """Epoch-dependent ramp for alpha (Implementation Details)."""
    if epoch < ATTN_START_EPOCH:
        return 0.0
    return min(ALPHA_ATTN_MAX, (epoch - ATTN_START_EPOCH) * (ALPHA_ATTN_MAX / ATTN_RAMP_EPOCHS))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | seed: {SEED} | use_spoof: {USE_SPOOF}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_loader = build_train_loader(TRAIN_FOLDER_REAL, TRAIN_FOLDER_SPOOF, BATCH_SIZE)

    model = ACFASModel().to(device)
    attn_extractor = AttentionMapExtractor(model.backbone)  # all 10 attention blocks
    print(f"Attention blocks used for C(x): {len(attn_extractor.hook_paths)}")

    print("Running warm-up pass to initialise the fixed compactness center c ...")
    center = compute_warmup_center(model, train_loader, device)
    criterion = ManhattanCompactnessLoss(center, r=R, m=M, lambda_l=LAMBDA_L, lambda_s=LAMBDA_S).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, foreach=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)

    print(f"\n{'=' * 55}")
    print(f"Training started | r={R} m={M} lr={LR} p_aug={P_AUG}")
    print(f"{'=' * 55}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        alpha = alpha_schedule(epoch)

        total_loss = 0.0
        c_live_sum, n_live, c_spoof_sum, n_spoof = 0.0, 0, 0.0, 0
        pbar = tqdm.tqdm(train_loader, desc=f"Ep {epoch:03d}/{NUM_EPOCHS} [alpha:{alpha:.3f}]", leave=False)

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # --- WFRF: applied once per mini-batch, per Algorithm 1 ---
            wfrf_images, mode, w_low, w_high = wfrf_train_batch(images, p_aug=P_AUG)
            wfrf_images = normalize_batch(wfrf_images)

            optimizer.zero_grad()
            attn_extractor.clear()

            projected = model(wfrf_images)
            attn_maps = attn_extractor.get()  # not detached: L_AC / L_AM train the backbone
            c_scores = attention_consistency_score(attn_maps, wfrf_images.shape[0])

            mh_loss = criterion(projected, labels)
            lac_real = attention_consistency_loss_real(c_scores, labels)
            lam = attention_margin_loss(c_scores, labels, lambda_m=LAMBDA_M)

            loss = total_objective(mh_loss, lac_real, lam, alpha=alpha, beta=BETA_AM)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                live, spoof = labels == 1, labels == 0
                c_live_sum += c_scores[live].sum().item()
                n_live += int(live.sum())
                c_spoof_sum += c_scores[spoof].sum().item()
                n_spoof += int(spoof.sum())
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "WFRF": mode})

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        c_live = c_live_sum / n_live if n_live else float("nan")
        c_spoof = c_spoof_sum / n_spoof if n_spoof else float("nan")

        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | alpha: {alpha:.3f} | C live/spoof: {c_live:.3f}/{c_spoof:.3f}")
        torch.save(
            {"epoch": epoch, "seed": SEED, "model_state_dict": model.state_dict(), "center": criterion.center},
            os.path.join(SAVE_DIR, f"model_ep{epoch:03d}.pth"),
        )

    attn_extractor.remove()
    print("\nFinished!")


if __name__ == "__main__":
    train()
