# PACENet

**PACENet** is a multi-stream deep learning framework for **non-invasive knee pathology classification and clinical screening** from bilateral gait kinematic time series. It discriminates among three conditions — **Healthy**, **ACLD** (Anterior Cruciate Ligament Deficiency), and **KOA** (Knee Osteoarthritis) — by jointly modeling unilateral temporal patterns, inter-limb (bilateral) asymmetry, frequency-domain content, and hand-crafted biomechanical gait parameters.

The repository contains the full pipeline: data loading, adaptive gait-cycle segmentation, acquisition-aware data augmentation, model training with stratified *k*-fold cross-validation, a comprehensive benchmark against strong baselines, ablation studies, clinical screening with sensitivity/specificity-constrained thresholds, and publication-ready plotting utilities.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation & Experiments](#evaluation--experiments)
- [Citation](#citation)

---

## Overview

Gait analysis captures subtle kinematic changes caused by knee pathology. PACENet is designed to exploit four complementary views of the same bilateral gait signal:

1. **Unilateral temporal dynamics** — what each leg does over time.
2. **Bilateral comparison** — how the affected and unaffected legs differ (a key clinical marker).
3. **Spectral content** — rhythm and frequency signatures of pathological gait.
4. **Domain kinematic parameters** — interpretable biomechanical features (range of motion, peak flexion, thrust, etc.).

A learned fusion module combines these streams, and the resulting classifier supports both standard 3-class diagnosis and **rule-based clinical screening** (e.g., "flag any abnormal gait at ≥95% sensitivity").

---

## Architecture

PACENet fuses four feature extractors. Each can be toggled via configuration for ablation studies.

| Stream | Module | Description |
|--------|--------|-------------|
| Unilateral | **UFE** (Unilateral Feature Extractor) | A 2D temporal-conv patchifier (shared across legs) produces patch embeddings; a self-attention block with learnable positional encoding and attention pooling summarizes each leg's temporal dynamics. |
| Bilateral | **CFE** (Comparative Feature Extractor) | Bidirectional **patch cross-attention** between left and right leg patches, gated by learnable scalars (`gamma_l`, `gamma_r`), captures inter-limb asymmetry. |
| Spectral | **SFE** (Spectral Feature Extractor) | Hamming-windowed FFT magnitude (log-scale) projected through an MLP extracts frequency-domain features per channel. |
| Kinematic | **KFE** (Kinematic Feature Extractor) | An MLP over 12 hand-crafted gait parameters computed per subject: flexion **ROM**, **extension** minima, stance/swing **peak flexion**, loading-response **thrust (EVVE)**, and **AP translation range (APTR)** for both legs. |

The four stream outputs are concatenated, passed through a batch-norm + dropout fusion MLP, and a linear classifier emits 3-class logits.

```
            ┌─────────────┐   ┌──────────────────┐
 Left leg ──►│   UFE (L)   │──►│                  │
            └─────────────┘   │  Patch Cross-Attn │── CFE (L) ──┐
            ┌─────────────┐   │   (Bilateral)     │             │
Right leg ──►│   UFE (R)   │──►│                  │── CFE (R) ──┤
            └─────────────┘   └──────────────────┘             │
                                                                 ►
            ┌─────────────┐                                     │
Full signal ►│    SFE      │─────────────────────────────────────┤  Fusion  ► Classifier ► logits
            └─────────────┘                                     │
            ┌─────────────┐                                     │
 KPF vector ►│    KFE      │─────────────────────────────────────┘
            └─────────────┘
```

### Baselines & benchmarks

The codebase also includes reimplemented baselines for comparison under an identical pipeline:

- **Deep time-series models:** ResNet, TimesNet, PatchTST, iTransformer, Medformer
- **Traditional ML:** SVM, XGBoost (operating on hand-crafted features)

---

## Dataset

The project uses the **KGKD** (Knee Gait Kinematic Dataset), provided under `Dataset/KGKD/`:

- `dev_dataset.csv` — development set (used for *k*-fold CV training/validation, plus an internal held-out test split)
- `test_dataset.csv` — fully independent external test set

### Format

Each row corresponds to one leg of a subject. Two rows (left + right) share a `person_id`:

| Column | Description |
|--------|-------------|
| `person_id` | Subject identifier (paired left/right legs) |
| `source_file`, `original_id` | Provenance / traceability metadata |
| `gender`, `age`, `bmi` | Demographics |
| `leg` | `left` or `right` |
| `label` | `Healthy`, `ACLD`, or `KOA` |
| `features` | Stringified 6-channel × 600-sample kinematic array (6 DoFs per leg, sampled at 60 Hz) |

A subject is labeled **ACLD** if either leg is ACLD, **KOA** if either leg is KOA, otherwise **Healthy**.

### Preprocessing pipeline

1. **Adaptive gait-cycle segmentation** — autocorrelation-based period estimation and swing-peak detection split each 10-second trial into normalized 100-sample cycles (per leg, then paired).
2. **Quality control** — cycles are filtered by inter-cycle correlation; subjects with too few valid cycles are dropped.
3. **Standardization** — per-DoF `StandardScaler` fitted on the training fold (left + right channels combined), then applied to all splits.
4. **Acquisition-aware augmentation** (training only) — class-balanced augmentation combining jitter, scaling, magnitude warping, time warping, random bias, and kinematic crosstalk (simulated sensor/anatomical-axis misalignment).

---

## Repository Structure

```
PACENet/
├── main.py                    # Entry point: run modes (main / exp / replot)
├── trainer.py                 # k-fold runner, training/eval loops, screening hooks
├── clinical_screening.py      # Sensitivity/specificity-constrained thresholding & DCA
├── utils.py                   # Config setup, seeding, device init, metrics
├── plot_utils.py              # All publication-ready plotting utilities
├── Dataset/
│   ├── dataset_loader.py      # KGKD loading, splitting, Dataset class
│   ├── data_augmentation.py   # Acquisition-aware gait augmentation
│   ├── utils.py               # Gait-cycle segmentation & quality control
│   └── KGKD/
│       ├── dev_dataset.csv
│       └── test_dataset.csv
└── Models/
    ├── PACENet.py             # PACENet (UFE, SFE, CFE, KFE, fusion) + KPF calculator
    ├── Attention.py           # Self-attention, patch cross-attention, attention pooling
    ├── Exp_Models.py          # Unified model selector/wrapper
    ├── ML_Models.py           # SVM / XGBoost feature-based baselines
    ├── ResNet.py
    ├── TimesNet.py
    ├── PatchTST.py
    ├── iTransformer.py
    └── Medformer.py
```

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch (CUDA recommended for GPU training)
- NumPy, Pandas, SciPy, scikit-learn
- XGBoost (only required for the XGBoost baseline)
- Matplotlib / Seaborn (for plotting)
- tqdm

```bash
git clone https://github.com/FT-MANIA/PACENet.git
cd PACENet
pip install torch numpy pandas scipy scikit-learn xgboost matplotlib seaborn tqdm
```

---

## Usage

All functionality is driven by `main.py` and the `--run_mode` flag.

### 1. Main analysis (train & evaluate PACENet)

Runs 5-fold stratified cross-validation of the full PACENet model, collects predictions/attention, and saves the plot-data cache.

```bash
python main.py --run_mode main --gpu 0
```

### 2. Full experiments (benchmark + ablation)

Runs every baseline model and every architecture/augmentation ablation, persisting intermediate results for downstream plotting.

```bash
python main.py --run_mode exp --gpu 0
```

### 3. Re-render plots only

Regenerates all figures from previously saved `.pkl` caches without retraining.

```bash
python main.py --run_mode replot
```

Outputs are written under `Results/`:
- `Results/<timestamp>/` — per-run checkpoints, fold reports, and experiment CSVs
- `Results/exp_figures/` — generated figures and plot-data caches

---

## Configuration

Key command-line arguments (see `main.py` for the full list):

**Data & splits**
- `--dataset_path` (default `Dataset/KGKD`)
- `--internal_test_size` (default `0.2`)
- `--seed` (default `172`)
- `--Norm` (default `True`) — per-DoF standardization
- `--use_data_augmentation` (default `True`), `--aug_ratios` (per-class multipliers)

**Model architecture**
- `--kernel_size` (default `6 40`), `--stride` (default `3`) — UFE patchifier
- `--embed_dim` `64`, `--dim_ff` `256`, `--num_heads` `4`
- `--UFE_dim`, `--CFE_dim`, `--SFE_dim`, `--KFE_dim` (default `32`), `--num_KF` `12`
- `--dropout` `0.4`, `--num_classes` `3`
- `--use_UFE / --use_CFE / --use_SFE / --use_KFE` — toggle streams (for ablation)

**Training**
- `--batch_size` `64`, `--lr` `3e-4`, `--weight_decay` `1e-2`, `--epochs` `100`, `--k_folds` `5`

**Clinical screening**
- `--enable_clinical_screening` (default `True`)
- `--screen_abnormal_classes` (default `1 2` → ACLD & KOA treated as "abnormal")

---

## Evaluation & Experiments

- **Cross-validation:** Stratified 5-fold CV on the development set. Each fold trains on the training partition, selects the best model by validation macro-F1, then evaluates on both the **internal test split** and the **external test set**.
- **Metrics:** accuracy, macro precision/recall/F1, macro AUROC and AUPRC, per-fold confusion matrices.
- **Benchmark:** PACENet vs. ResNet, TimesNet, PatchTST, iTransformer, Medformer, SVM, XGBoost — compared via ROC/PR curves and metric boxplots across folds, on both internal and external tests.
- **Ablations:**
  - *Architecture:* without UFE / CFE / SFE / KFE.
  - *Augmentation:* without each augmentation component (jitter, scaling, magnitude warp, time warp, random bias, crosstalk).
  - Paired delta plots quantify the performance change from each component.
- **Clinical screening:** Thresholds are fit on validation data to guarantee a target sensitivity (or specificity) for the *Healthy vs. Abnormal (ACLD/KOA)* binary screening task, then applied unchanged to the test sets. **Decision Curve Analysis (DCA)** quantifies net benefit across threshold probabilities for overall screening, ACLD detection, and KOA detection.
- **Interpretability:** Spectral-weight visualizations, raw-signal attention overlays, and affected-vs-unaffected leg attention comparison for ACLD subjects.

---

## Citation

If you use this code or the KGKD dataset, please cite the accompanying work. Citation details will be added upon publication.
