# CSOANet2026

**From Perception to Imagination: A Robust Deep Learning Architecture for Real-Time EEG Decoding and Neurophysiological Validation**

Yu Gao<sup>1,2*</sup>, José Miguel Diniz<sup>2</sup>

<sup>1</sup>Department of Electrical and Computer Engineering, Faculty of Engineering, University of Porto  
<sup>2</sup>PhD Program in Health Data Science, Faculty of Medicine, University of Porto

## Summary

This repository provides the complete codebase to reproduce all results in the paper. It includes:

- **Classical ML baselines** (LDA, SVM, RF) with ERP + spectral features
- **EEGNet-8,2** deep learning baseline (Lawhern et al., 2018)
- **CSOANet2026** — proposed lightweight CNN with Cross-Scale Oscillatory Attention (CSOA)
- **Grad-CAM / Saliency** interpretability visualizations
- **Statistical tests** (Wilcoxon signed-rank with Bonferroni correction)
- **Inference benchmarking** (GPU and CPU latency)

## Key Results

| Model | Parameters | Mean Acc (%) | Min Acc (%) | Latency (ms) |
|---|---|---|---|---|
| LDA | — | — | — | 0.06 |
| SVM | — | — | — | 0.22 |
| RF | — | — | — | 3.95 |
| EEGNet-8,2 | 7,322 | — | — | — |
| **CSOANet2026 (ours)** | **2,388** | **—** | **—** | **1.15** |

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/CSOANet2026.git
cd CSOANet2026
pip install -r requirements.txt
```

### 2. Prepare data

Place preprocessed MNE epoch files in `eeg_project/data/processed/`:
```
eeg_project/data/processed/
  sub_01_epo.fif
  sub_02_epo.fif
  ...
  sub_54_epo.fif
```

### 3. Run all experiments

**Windows one-click:**
```bat
run_all.bat
```

**Or step by step:**
```bash
# Speed benchmark
python benchmark.py

# Classical ML (LDA/SVM/RF)
python batch_run_ml.py

# EEGNet baseline
python batch_run_eegnet.py

# CSOANet2026 (proposed)
python batch_run_csoanet2026.py

# Generate Table 1, Wilcoxon tests, and comparison figure
python summarize_results.py

# Grad-CAM interpretability (per subject)
python gradcam.py --subject 04
python gradcam.py --subject 10
```

**Quick test (3 subjects):**
```bash
python batch_run_ml.py --subjects 04 10 22
python batch_run_eegnet.py --subjects 04 10 22
python batch_run_csoanet2026.py --subjects 04 10 22
python summarize_results.py
```

## Project Structure

```
CSOANet2026/
├── README.md
├── requirements.txt
├── LICENSE
├── run_all.bat                         # Windows one-click
│
├── benchmark.py                        # Inference speed: CSOANet2026 vs EEGNet
├── batch_run_ml.py                     # Classical ML (LDA/SVM/RF), all subjects
├── batch_run_eegnet.py                 # EEGNet-8,2 baseline, all subjects
├── batch_run_csoanet2026.py            # Proposed model, all subjects
├── gradcam.py                          # Grad-CAM + Saliency visualization
├── summarize_results.py                # Table 1, Wilcoxon tests, box plot
│
├── eeg_project/
│   ├── scripts/
│   │   ├── models/
│   │   │   ├── csoanet2026.py          # CSOANet2026 + CSOA (2,388 params)
│   │   │   └── eegnet.py              # EEGNet-8,2 (7,322 params)
│   │   ├── datasets/
│   │   │   └── time_dataset.py         # Time-domain EEG dataset
│   │   ├── preprocess.py               # CAR + FIR + ICA + epoching
│   │   ├── load_data.py                # Raw data loader
│   │   ├── features.py                 # ERP + spectral feature extraction
│   │   ├── train_eval.py               # Classical ML classifiers
│   │   └── trainer_dl.py               # DL training utilities
│   ├── data/
│   │   ├── processed/                  # Preprocessed .fif epochs
│   │   └── time_cache/                 # Auto-generated tensor cache
│   ├── results/                        # CSV outputs
│   └── figures/                        # Generated plots
```

## Novel Contribution: Cross-Scale Oscillatory Attention (CSOA)

Three depthwise dilated branches capture neural activity at different oscillatory timescales. Pairwise products model cross-frequency coupling. A learned gate adaptively selects the most discriminative patterns per trial.

```
Input (B, 1, 62, 2000)
  → Temporal Conv + Spatial Conv + Pool → (B, 16, 1, 250)
  → d0 (d=1): Beta/Gamma | d1 (d=3): Alpha | d2 (d=12): Theta/Delta
  → Cross-freq coupling: d0*d1, d1*d2, d0*d2
  → CSOA Gate: softmax(Linear(6→6)) → weighted sum + residual
  → Separable Conv → Depth Attention → FC(16→2)
```

## Dataset

[OpenNeuro ds005697](https://openneuro.org/datasets/ds005697) — PerceiveImagine  
46 subjects, 62-channel EEG, 1000 Hz, perception vs. imagination classification.

## References

1. Lawhern et al., "EEGNet", *J. Neural Eng.* 15 (2018) 056013.
2. Hu et al., "Squeeze-and-Excitation Networks", *CVPR* 2018.
3. Altaheri et al., "ATCNet", *IEEE TNSRE* 30 (2022).
4. Miao et al., "LMDA-Net", *NeuroImage* 276 (2023).

## License

MIT
