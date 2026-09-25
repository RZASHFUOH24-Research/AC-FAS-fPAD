"""
AC-FAS evaluation — anomaly score S(x) (Eq. score) and the standard FAS
metrics reported in the paper (Section IV-C / Evaluation Metrics).

No CLI arguments by design: edit the CONFIG block below directly.
"""

import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset import build_eval_loader
from losses import attention_consistency_score
from model import ACFASModel, AttentionMapExtractor, ATTENTION_HOOK_PATHS
from wfrf import wfrf_eval_batch

# ==========================================
# CONFIG — edit these directly, no argparse
# ==========================================
CHECKPOINT_PATH = r"..\..\model.pth"

# Validation set (real faces ONLY) used to pick the anomaly threshold tau,
# per the paper: "determined from scores obtained on a validation set
# consisting exclusively of real faces ... without the need for spoof
# labels at test time."
VAL_FOLDER_REAL = r"C:\Phd(Reza)\Dataset\AC-FAS\Val\Real"

# Test set: real + spoof, used for final reporting.
TEST_FOLDER_REAL = r"..\..\Real"
TEST_FOLDER_SPOOF = [r"..\..\Spoof"]

BATCH_SIZE = 32

# Anomaly-score weighting (Eq. score) — paper-confirmed values.
ALPHA_S = 0.6  # weight on the Manhattan (L1) distance term
BETA_S = 0.4  # weight on the attention-inconsistency term

# Threshold selection strategy on the real-only validation set: tau is set
# to the q-th percentile of validation (real) anomaly scores, so that a
# chosen fraction of real faces are accepted. The paper does not pin a
# specific quantile in the main text; 95th percentile (~5% FRR budget on
# validation) is a common, reasonable default — adjust to match your setup.
VAL_PERCENTILE = 95.0

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_batch(batch: torch.Tensor, device) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (batch - mean) / std


@torch.no_grad()
def compute_scores(model, attn_extractor, center, loader, device):
    """Runs the deterministic WFRF + model forward pass and computes the
    anomaly score S(x) = alpha_s * ||z~ - c||_1 + beta_s * C(x) for every
    sample in `loader`. Returns (scores, labels) as numpy arrays."""
    model.eval()
    all_scores, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)

        wfrf_images = wfrf_eval_batch(images)
        wfrf_images = normalize_batch(wfrf_images, device)

        attn_extractor.clear()
        projected = model(wfrf_images)
        attn_maps = attn_extractor.get(ATTENTION_HOOK_PATHS)

        l1_dist = torch.norm(projected - center, p=1, dim=1)
        c_scores = attention_consistency_score(attn_maps)

        scores = ALPHA_S * l1_dist + BETA_S * c_scores

        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def select_threshold(val_scores: np.ndarray, percentile: float = VAL_PERCENTILE) -> float:
    """Sets tau from real-only validation scores (no spoof labels needed)."""
    return float(np.percentile(val_scores, percentile))


def compute_metrics(scores: np.ndarray, labels: np.ndarray, tau: float) -> dict:
    """labels: 1 = live, 0 = spoof. Predicted spoof when score > tau."""
    pred_spoof = scores > tau  # True => predicted spoof

    is_live = labels == 1
    is_spoof = labels == 0

    # APCER: proportion of spoof samples misclassified as live.
    apcer = float((~pred_spoof[is_spoof]).mean()) if is_spoof.any() else float("nan")
    # BPCER: proportion of live samples misclassified as spoof.
    bpcer = float(pred_spoof[is_live].mean()) if is_live.any() else float("nan")
    acer = (apcer + bpcer) / 2

    # FAR/FRR/HTER (FAR = attacks accepted as live = APCER here; FRR = BPCER here).
    far, frr = apcer, bpcer
    hter = (far + frr) / 2

    auc = roc_auc_score(labels, -scores) if is_live.any() and is_spoof.any() else float("nan")

    return {"APCER": apcer, "BPCER": bpcer, "ACER": acer, "HTER": hter, "AUC": auc}


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ACFASModel().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    center = checkpoint["center"].to(device)

    attn_extractor = AttentionMapExtractor(model.backbone)

    val_loader = build_eval_loader(VAL_FOLDER_REAL, [], BATCH_SIZE)
    val_scores, _ = compute_scores(model, attn_extractor, center, val_loader, device)
    tau = select_threshold(val_scores)
    print(f"Selected threshold tau = {tau:.4f} (from {VAL_PERCENTILE}th percentile of real-only validation scores)")

    test_loader = build_eval_loader(TEST_FOLDER_REAL, TEST_FOLDER_SPOOF, BATCH_SIZE)
    test_scores, test_labels = compute_scores(model, attn_extractor, center, test_loader, device)
    metrics = compute_metrics(test_scores, test_labels, tau)

    print("\n---- Test Evaluation ----")
    for k, v in metrics.items():
        print(f"{k}: {v * 100:.2f}%" if k != "AUC" else f"{k}: {v * 100:.2f}")

    attn_extractor.remove()
    return metrics


if __name__ == "__main__":
    evaluate()
