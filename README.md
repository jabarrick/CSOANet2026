# CSOANet2026

**From Perception to Imagination: A Robust Deep Learning Architecture for Real-Time EEG Decoding and Neurophysiological Validation**

Yu Gao¹², José Miguel Diniz²

¹ Department of Electrical and Computer Engineering, Faculty of Engineering, University of Porto
² PhD Program in Health Data Science, Faculty of Medicine, University of Porto

---

## Summary

This repository provides the complete codebase to reproduce all results in the CMBBE-IV paper. It includes:

- **Classical ML baselines** (LDA, SVM, RF) with ERP + spectral features
- **Deep-learning baselines**: EEGNet (Lawhern et al. 2018), ShallowConvNet, DeepConvNet (Schirrmeister et al. 2017)
- **CSOANet2026** — proposed lightweight CNN with Cross-Scale Oscillatory Attention (CSOA), 3,028 parameters
- **Cross-subject validation** via leave-one-subject-out (LOSO) on 46 subjects (`LOSO/` subdirectory)
- **Grad-CAM / saliency** interpretability visualizations
- **Statistical tests** (Wilcoxon signed-rank with Bonferroni correction)
- **Inference benchmarking** (CPU and GPU latency, 1,000 single-trial iterations)

---

## Key Results (N = 46)

### 5-fold CV (within-subject)

| Model           | Parameters | Mean Acc (%) ± SD | Median |
|-----------------|------------|-------------------|--------|
| LDA             | —          | 85.77 ± 6.85       | 0.864  |
| SVM             | —          | 97.14 ± 2.80       | 0.982  |
| RF              | —          | 97.48 ± 3.16       | 0.982  |
| EEGNet          | 8,874      | 98.25 ± 2.65       | 0.989  |
| ShallowConvNet  | 123,242    | 98.24 ± 3.16       | 0.989  |
| DeepConvNet     | 319,327    | 94.72 ± 5.95       | 0.964  |
| **CSOANet2026** | **3,028**  | **98.67 ± 2.46**   | **0.989** |

### LOSO (cross-subject)

| Model           | Parameters | Mean Acc (%) ± SD | Median | ≥ 0.95 |
|-----------------|------------|-------------------|--------|--------|
| LDA             | —          | 86.45 ± 12.21      | 0.909  | 13/46  |
| SVM             | —          | 89.47 ± 11.43      | 0.946  | 20/46  |
| RF              | —          | 90.29 ± 11.30      | 0.948  | 23/46  |
| EEGNet          | 8,874      | 96.16 ± 4.90       | 0.976  | 34/46  |
| **CSOANet2026** | **3,028**  | **96.73 ± 6.31**   | 0.990  | 38/46  |
| ShallowConvNet  | 123,242    | 96.88 ± 6.37       | 0.991  | 41/46  |
| DeepConvNet     | 319,327    | 97.91 ± 4.92       | 0.994  | 42/46  |

CSOANet2026 reaches near-DeepConvNet LOSO performance at **1/105** of its parameter count and is statistically indistinguishable from ShallowConvNet (Wilcoxon p = 0.41) at **1/40** of its parameter count.

### Inference latency (1,000 single-trial iterations)

CSOANet2026 — **3.99 ± 0.60 ms (CPU) / 0.81 ± 0.16 ms (GPU)**, well within the 10-ms real-time deadline for closed-loop BCI.

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/jabarrick/CSOANet2026.git
cd CSOANet2026
pip install -r requirements.txt
```

### 2. Prepare data

The dataset (OpenNeuro **ds005697**, PerceiveImagine, 46 subjects, **65-channel** EEG @ 1 kHz) is publicly available at https://openneuro.org/datasets/ds005697.

Place preprocessed MNE epoch files in `eeg_project/data/processed/`:

```
eeg_project/data/processed/
  sub_01_epo.fif
  sub_02_epo.fif
  ...
  sub_54_epo.fif
```

### 3. Run experiments

**Within-subject 5-fold cross-validation:**

```bash
# Classical ML (LDA / SVM / RF)
python batch_run_ml.py

# EEGNet baseline
python batch_run_eegnet.py

# ShallowConvNet + DeepConvNet baselines
python batch_run_convnets.py --model shallow
python batch_run_convnets.py --model deep

# CSOANet2026 (proposed)
python batch_run_csoanet2026.py
```

**Cross-subject LOSO** (see `LOSO/` for full details):

```bash
cd LOSO
python batch_run_loso.py                              # CSOANet2026
python batch_run_loso_baselines.py --model lda
python batch_run_loso_baselines.py --model svm
python batch_run_loso_baselines.py --model rf
python batch_run_loso_baselines.py --model eegnet
python batch_run_loso_baselines.py --model shallow
python batch_run_loso_baselines.py --model deep
```

Each LOSO run produces a per-subject CSV in `eeg_project/results/`.

**Generate figures and tables:**

```bash
python generate_figures.py
```

**Quick test (3 subjects):**

```bash
python batch_run_ml.py --subjects 04 10 22
python batch_run_csoanet2026.py --subjects 04 10 22
```

---

## Project Structure

```
CSOANet2026/
├── README.md
├── requirements.txt
├── LICENSE                                 # MIT
│
├── batch_run_ml.py                         # LDA / SVM / RF, 5-fold
├── batch_run_eegnet.py                     # EEGNet 5-fold
├── batch_run_convnets.py                   # Shallow + Deep ConvNet 5-fold
├── batch_run_csoanet2026.py                # CSOANet2026 5-fold
│
├── generate_figures.py                     # Figs 1B, 3, 4, 6, 7, 8 + tables
├── generate_figures.md                     # Figure regeneration documentation
│
├── LOSO/                                   # Cross-subject validation (R1.2 / R3.3)
│   ├── README.md                           # LOSO-specific reproduction guide
│   ├── batch_run_loso.py                   # CSOANet2026 LOSO runner
│   ├── batch_run_loso_baselines.py         # 6-baseline LOSO runner
│   ├── figure_3_kfold.py                   # Fig 3 generator
│   ├── figure_8_loso.py                    # Fig 8 generator
│   ├── results/
│   │   ├── csoa_loso_n46.csv               # CSOANet2026 LOSO (with gate weights)
│   │   ├── loso_lda.csv                    # 6 baseline LOSO CSVs
│   │   ├── loso_svm.csv
│   │   ├── loso_rf.csv
│   │   ├── loso_eegnet.csv
│   │   ├── loso_shallow.csv
│   │   ├── loso_deep.csv
│   │   ├── loso_7way_summary.csv           # 7-method per-subject merged table
│   │   └── kfold_5fold/                    # 5-fold CSVs for LOSO-drop computation
│   │       ├── csoa_5fold.csv
│   │       ├── deep_5fold.csv
│   │       ├── eegnet_5fold.csv
│   │       ├── ml_5fold.csv
│   │       └── shallow_5fold.csv
│   └── figures/
│       ├── fig3_kfold.png
│       └── fig8_loso.png
│
├── eeg_project/
│   ├── scripts/
│   │   ├── models/
│   │   │   ├── csoanet2026.py              # CSOANet2026 (3,028 params)
│   │   │   ├── eegnet.py                   # EEGNet (8,874 params)
│   │   │   └── convnets.py                 # ShallowConvNet, DeepConvNet
│   │   ├── datasets/
│   │   │   └── time_dataset.py             # Time-domain EEG dataset loader
│   │   ├── preprocess.py                   # CAR + FIR + ICA + epoching
│   │   ├── load_data.py                    # Raw data loader
│   │   ├── features.py                     # ERP + spectral feature extraction
│   │   ├── train_eval.py                   # Classical ML classifiers
│   │   └── trainer_dl.py                   # DL training utilities
│   ├── data/
│   │   ├── processed/                      # Preprocessed .fif epochs
│   │   └── time_cache/                     # Auto-generated tensor cache
│   ├── results/                            # CSV outputs
│   └── figures/                            # Generated plots
```

---

## Reproducibility Details

### Hardware

All experiments performed on a **Dell G16 7630** laptop (Windows 11):

- Intel Core i9-13900HX (24 cores, 2.20 GHz base)
- 64 GB DDR5-4800 RAM
- NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
- CUDA 12.1, PyTorch 2.5.1, Python 3.10

The DeepConvNet LOSO (46 folds) and the trailing 10 ShallowConvNet folds were additionally run on a NVIDIA RTX 3090 (24 GB) workstation with the same software stack.

### Preprocessing pipeline (§2.3)

- Common-average reference (CAR)
- FIR band-pass 1–40 Hz
- FastICA (MNE-Python, `n_components = 15`, `random_state = 42`); ocular components flagged via `mne.preprocessing.ICA.find_bads_eog` (correlation with frontal channel Fp1) and excluded from reconstruction
- Epoching: single 3.5 s window from **−0.5 to +3.0 s** relative to each event marker (3,501 samples at 1 kHz)

### Training settings (fixed, not searched) (§2.5)

- Optimizer: AdamW, lr = 1e-3, weight decay = 1e-3
- Loss: cross-entropy with label smoothing 0.1
- Schedule: cosine annealing over 50 epochs
- Early stopping: patience = 15 on validation accuracy; best checkpoint selected for evaluation
- Batch size: 32, validation fraction: 10 %
- Augmentation: Mixup α = 0.2, sliding-window temporal crop (90 %), channel dropout 10 %, Gaussian noise σ = 0.02

### Random seed

Seed = 42 set globally in `torch`, `numpy`, `random`, and CUDA. `StratifiedKFold` uses `random_state = 42`; PyTorch is seeded with `42 + fold_index` per fold. The same seeding scheme is used for both the 5-fold and the LOSO experiments.

### Cross-validation protocols

- **5-fold (within-subject)**: stratified `StratifiedKFold` per subject; 80/20 train/test split.
- **LOSO (cross-subject)**: for each held-out subject *i*, the model is trained on the remaining 45 subjects (90 % train, 10 % per-subject stratified validation for early stopping) and tested on subject *i*.

---

## Novel Contribution: Cross-Scale Oscillatory Attention (CSOA)

CSOA replaces dense channel attention with a sparse, neurophysiologically interpretable softmax gate over six oscillatory terms (42 trainable parameters within the 3,028 total).

```
Input (B, 1, 65, 3501)
  → Temporal Conv + Spatial Conv + Pool         → (B, 16, 1, 437)
  → Three dilated branches:
        d=1  (Beta/Gamma)
        d=3  (Alpha)
        d=12 (Theta/Delta)
  → Three cross-frequency couplings:
        d0 × d1   (Beta × Alpha)
        d1 × d2   (Alpha × Theta)
        d0 × d2   (Beta × Theta)
  → CSOA gate: softmax(Linear(6 → 6)) → weighted sum + residual
  → Separable Conv → Depth Attention → FC(16 → 2)
```

Under LOSO training, the gate weights converge consistently across the cohort to a **TD + BxA majority** (97.5 % ± 3.2 % of total weight; TD: 0.59 ± 0.08, BxA: 0.38 ± 0.08), providing a per-cohort summary of which oscillatory mechanisms dominate the perception–imagination discrimination.

---

## Dataset

**OpenNeuro ds005697** — PerceiveImagine
46 subjects, 65-channel EEG, 1 kHz, perception vs imagination paradigm.

- https://openneuro.org/datasets/ds005697
- Naselaris et al., dataset description (in the dataset's `README.md`).

---

## Citation

If this code or data is useful in your research, please cite:

```
Gao Y, Diniz JM. From perception to imagination: a robust deep learning
architecture for real-time EEG decoding and neurophysiological validation.
Computer Methods in Biomechanics and Biomedical Engineering: Imaging &
Visualization (in revision, 2026).
```

---

## References

1. Lawhern VJ et al. *EEGNet*. J Neural Eng 15:056013 (2018).
2. Schirrmeister RT et al. *Deep learning with convolutional neural networks for EEG decoding and visualization*. Hum Brain Mapp 38:5391–5420 (2017).
3. Hu J et al. *Squeeze-and-Excitation Networks*. CVPR (2018).
4. Selvaraju RR et al. *Grad-CAM*. ICCV (2017).
5. Ji Z et al. *Subject-specific CNN with parameter-based transfer learning for SSVEP detection*. Biomed Signal Process Control 103:107404 (2025).
6. Zhao W et al. *TCANet: temporal convolutional attention network for motor imagery EEG decoding*. Cogn Neurodyn 19:91 (2025).
7. Zhao W et al. *MSCFormer*. Sci Rep 15:12935 (2025).
8. Zhao W et al. *CTNet*. Sci Rep 14:20237 (2024).

## License

MIT

