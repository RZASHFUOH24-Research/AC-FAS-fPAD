# AC-FAS: Attention Consistency for Semi-Supervised One-Class Face Anti-Spoofing

**Status:** Under review (IEEE Transactions on Artificial Intelligence).

**Authors:** Mohammadreza (Reza) Sheikhfathollahi, Simon Parkinson, Saad Khan — University of Huddersfield.

## Abstract

Face anti-spoofing (FAS) is essential for the secure deployment of face
recognition (FR) systems, particularly in high-stakes applications.
However, most existing methods formulate FAS as a closed-set binary
classification problem, which limits their ability to generalise to unseen
attack types in shifting domains. In this paper, we propose Attention
Consistency for Face Anti-Spoofing (AC-FAS), a principled semi-supervised
one-class anomaly detection framework that reframes FAS by modelling the
intrinsic normality of live faces, optionally incorporating limited spoof
supervision to sharpen the decision boundary. Our approach is motivated by
the hypothesis that live faces exhibit lower cross-layer attention
divergence than spoof faces, a pattern independently observed in adjacent
domains such as generalisable synthetic-image detection, reflecting a
smoother evolution of spatial attention through the network. AC-FAS
leverages this finding by enforcing cross-layer attention consistency as a
domain-agnostic liveness feature. To further enhance robustness, we
introduce Weighted Frequency Response Filtering (WFRF), a Fourier-based
mechanism applied as both deterministic preprocessing and stochastic
augmentation to suppress domain-sensitive low frequencies while amplifying
high-frequency spoofing patterns. Furthermore, we employ a Manhattan
(L1-norm) Compactness Loss to optimise the feature space, which provides
superior robustness to outliers and ensures strict alignment between the
training objective and our parameter-free inference anomaly score.
Extensive experiments on multiple benchmarks, including OULU-NPU, SiW-Mv2,
and WMCA, demonstrate that AC-FAS consistently outperforms state-of-the-art
methods across intra-domain and leave-one-attack-out (LOAO) protocols,
including a combined cross-domain LOAO setting, effectively detecting both
known and novel presentation attacks. Notably, AC-FAS achieves a
state-of-the-art ACER of 0.08% on OULU-NPU and reduces the average ACER by
approximately 35% over the strongest competing method under the most
challenging cross-domain protocol, whilst maintaining a near-perfect AUC of
97.47% on the SiW-Mv2 leave-one-attack-out benchmark.

## Method Overview

AC-FAS reframes face anti-spoofing as **semi-supervised one-class anomaly
detection**: it models the normality of *live* faces, optionally using a
limited set of labelled spoof samples to sharpen the boundary. Three
components:

1. **WFRF (Weighted Frequency Response Filtering)** — a Fourier-domain
   reweighting of the input image using a square (Chebyshev-distance)
   mask, applied dually as deterministic preprocessing (fixed weights) and
   stochastic training-time augmentation (randomly resampled weights).
2. **Cross-layer Attention Consistency** — the KL divergence between the
   spatial-attention signatures of two selected transformer layers is used
   as a domain-agnostic liveness signal: live faces are regularised toward
   low divergence (`L_AC^r`), and a margin loss (`L_AM`) widens the gap
   between live and spoof consistency scores.
3. **Manhattan (L1-norm) Compactness Loss** — live embeddings are pulled
   within an L1 ball of radius `r` around a fixed center `c` (computed via
   a warm-up pass); spoof embeddings are pushed beyond margin `m`.

At inference, the anomaly score is
`S(x) = alpha_s * ||z~(x) - c||_1 + beta_s * C(x)`, thresholded against a
value `tau` selected from a real-only validation set (no spoof labels
needed at test time).

## Repository Structure

```
AC-FAS_fPAD/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── wfrf.py       # Weighted Frequency Response Filtering (Eqs. 2-6, Algorithm 1 dual-mode)
│   ├── model.py       # TinyViT-5M backbone, projection head, attention-map capture
│   ├── losses.py       # Manhattan compactness loss, attention consistency / margin losses
│   ├── dataset.py       # Folder-based real/spoof dataset loading
│   ├── train.py         # Main training script (Algorithm 1) — no CLI args, edit CONFIG block
│   └── evaluate.py      # Anomaly score, threshold selection, APCER/BPCER/ACER/HTER/AUC
├── data/README.md
├── results/README.md
└── images/README.md
```

No `argparse` is used anywhere — every script has a `CONFIG` block near the
top; edit the paths/hyperparameters directly and run the file.

## Installation

```bash
git clone https://github.com/<your-username>/AC-FAS_fPAD.git
cd AC-FAS_fPAD
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Edit the `CONFIG` block at the top of `src/train.py` (dataset paths,
   `SAVE_DIR`, batch size, epochs, etc.) and run:
   ```bash
   python src/train.py
   ```
2. Edit the `CONFIG` block at the top of `src/evaluate.py` (checkpoint
   path, validation/test folders) and run:
   ```bash
   python src/evaluate.py
   ```

## Confirmed Hyperparameters (Paper §III-D)

| Symbol | Value | Meaning |
|---|---|---|
| `r` | 2.0 | live compactness radius |
| `m` | 20.0 | spoof separation margin |
| `lambda_l`, `lambda_s` | 1.0, 1.0 | live/spoof term weights in `L_MH` |
| `alpha` | 0 → 0.5 (ramp, epochs 10-20) | weight on `L_AC^r` |
| `beta` | 0.3 (fixed from epoch 1) | weight on `L_AM` |
| `lambda_m` | 2.0 | margin in `L_AM` |
| `alpha_s`, `beta_s` | 0.6, 0.4 | inference score weights |
| `p_aug` | 0.5 | probability of WFRF stochastic-augmentation mode per batch |
| WFRF deterministic | `w_low=0.8`, `w_high=1.75` | |
| WFRF stochastic range | `U(0.5, 2.0)` | independently resampled `w_low`, `w_high` per batch |
| Backbone | TinyViT-5M | `d=320` CLS feature → projection head → `p=128` |
| Optimiser | Adam, lr `1e-4`, step ×0.1 at epochs 60, 85 | 100 epochs total |

## Implementation Notes — Corrections Made Against the Draft Code

An earlier, preliminary training script (TinyViT + a Euclidean two-stage
loss + a radial "FrequencyBoost" augmentation) was provided as a starting
point. Per your instruction that the manuscript is authoritative, this
codebase does **not** reuse that script directly; it re-implements the
pipeline from the paper's equations and Algorithm 1, correcting several
concrete mismatches found along the way:

- **Loss norm**: the draft used a squared-L2 ("Euclidean") hinge loss with
  an implicit center at the origin. The paper specifies an **L1
  (Manhattan)** hinge loss around a center `c` computed via a warm-up pass
  over live samples and then held fixed — implemented in
  `losses.ManhattanCompactnessLoss` / `losses.compute_warmup_center`.
- **alpha (attention-loss weight)**: the draft ramped to `0.1`; the paper's
  confirmed value is `alpha_max = 0.5`.
- **beta (margin-loss weight)**: the draft folded a hardcoded `0.3` weight
  *inside* the attention loss function, sharing the same ramp schedule as
  alpha. The paper treats `beta = 0.3` as a **separate, fixed** (non-ramping)
  coefficient in the total objective from epoch 1 — implemented as such in
  `train.py`.
- **WFRF distance metric**: the draft used a **radial (Euclidean)**
  distance grid for its frequency mask. The paper's WFRF explicitly uses a
  **square (Chebyshev)** distance, `d(u,v) = max(|u-u_c|, |v-v_c|)` (Eq. 3)
  — implemented in `wfrf.py`.
- **WFRF application**: the draft skipped the augmentation outright with
  probability `1-p`, and used fixed weights when applied. Per Algorithm 1,
  WFRF is applied to **every** image; only the *mode* (deterministic vs.
  stochastic weights) is chosen per mini-batch — implemented in
  `wfrf.wfrf_train_batch`.
- **Attention consistency score**: the draft computed a proxy from raw
  attention-block *output* norms. The paper defines `C(x)` as the KL
  divergence between proper post-softmax attention-weight *signatures* at
  selected layers (Eq. "consistency"). Extracting genuine attention-weight
  tensors from TinyViT required patching around two things not obvious
  from the paper text (see `model.py` for full detail): (a) TinyViT's
  fused-attention path never materialises attention weights, so `attn`
  must be captured via a forward-method patch, not a hook; and (b) not
  every layer pair has matching token count / batch semantics — the code
  uses stage 3's first and last blocks (both non-windowed, `N=49`), since
  the originally-implied stage-2/stage-3 pairing has mismatched shapes.
  If your own runs behind the paper's numbers used a different layer
  pair, update `ATTENTION_HOOK_PATHS` in `model.py` accordingly.
- **Projection head**: the paper specifies a final `LayerNorm` on the
  128-d projected embedding (`phi(z) = LayerNorm(...)`); the draft head
  had no such layer — added in `model.ProjectionHead`.
- **Input resolution**: the paper's implementation-details text says
  images are resized to 256×256, but the backbone name (`tiny_vit_5m_224`)
  and the draft code both use 224×224. This codebase keeps **224×224** to
  match the pinned backbone; flag this if your actual training run used
  256.

All of the above were verified against a live TinyViT-5M forward/backward
pass during development (shape and gradient-flow checks), not just against
the paper text.

## Citation

```bibtex
@article{acfas_under_review,
  author  = {Sheikhfathollahi, Mohammadreza and Parkinson, Simon and Khan, Saad},
  title   = {{AC-FAS}: Attention Consistency for Semi-Supervised One-Class Face Anti-Spoofing},
  journal = {IEEE Transactions on Artificial Intelligence},
  note    = {Under review},
  year    = {2026}
}
```

## Contact

Mohammadreza (Reza) Sheikhfathollahi — PhD student, University of Huddersfield.
