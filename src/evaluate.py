"""
AC-FAS evaluation — anomaly score S(x) (Eq. score) and the standard FAS
metrics reported in the paper (Evaluation Metrics).

No CLI arguments by design: edit the CONFIG block below directly.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from dataset import build_eval_loader, normalize_batch
from losses import attention_consistency_score
from model import ACFASModel, AttentionMapExtractor
from wfrf import wfrf_eval_batch

# ==========================================
# CONFIG — edit these directly, no argparse
# ==========================================
CHECKPOINT_PATH = r"..\checkpoints\run1\model_ep100.pth"

# Validation set (real faces ONLY) used to pick the anomaly threshold tau:
# no spoof labels are needed for calibration.
VAL_FOLDER_REAL = r"..\..\Val_Real"

# Test set: real + spoof, used for final reporting.
TEST_FOLDER_REAL = r"..\..\Test_Real"
TEST_FOLDER_SPOOF = [r"..\..\Test_Spoof"]

BATCH_SIZE = 32

# Anomaly-score weighting (Eq. score).
ALPHA_S = 0.6  # weight on the Manhattan (L1) distance term
BETA_S = 0.4  # weight on the attention-inconsistency term

# tau = 95th percentile of the anomaly scores of the real-only validation set.
VAL_PERCENTILE = 95.0


@torch.no_grad()
def compute_scores(model, attn_extractor, center, loader, device):
    """Runs the deterministic WFRF + model forward pass and computes the
    anomaly score S(x) = alpha_s * ||z~ - c||_1 + beta_s * C(x) for every
    sample in `loader`. Returns (scores, labels) as numpy arrays."""
    model.eval()
    all_scores, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)

        wfrf_images = normalize_batch(wfrf_eval_batch(images))

        attn_extractor.clear()
        projected = model(wfrf_images)
        c_scores = attention_consistency_score(attn_extractor.get(), wfrf_images.shape[0])

        l1_dist = torch.norm(projected - center, p=1, dim=1)
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

    # Live is the positive class; a lower anomaly score means "more live".
    auc = roc_auc_score(labels, -scores) if is_live.any() and is_spoof.any() else float("nan")

    return {"APCER": apcer, "BPCER": bpcer, "ACER": acer, "HTER": hter, "AUC": auc}


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ACFASModel(pretrained=False).to(device)  # weights come from the checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    center = checkpoint["center"].to(device)

    attn_extractor = AttentionMapExtractor(model.backbone)  # all 10 attention blocks

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
