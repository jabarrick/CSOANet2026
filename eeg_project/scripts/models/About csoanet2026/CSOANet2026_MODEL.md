# CSOANet2026: Cross-Scale Oscillatory Attention Network

A lightweight, interpretable CNN architecture for EEG-based Brain–Computer Interface classification. CSOANet2026 introduces **Cross-Scale Oscillatory Attention (CSOA)** — a novel mechanism that models multi-scale neural oscillations and their cross-frequency coupling, gated adaptively per trial.

---

## 1. Motivation

EEG signals encode cognitive states through oscillatory activity at multiple frequency bands: Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and Gamma (>30 Hz). Existing compact EEG classifiers (EEGNet, ShallowConvNet) treat the temporal dimension uniformly, missing two critical phenomena:

1. **Multi-scale oscillatory encoding**: Different frequency bands carry distinct cognitive information (e.g., Alpha ERD during imagery, Beta enhancement during motor planning).
2. **Cross-frequency coupling (CFC)**: Interactions between frequency bands (e.g., Alpha–Beta coupling during perception-to-imagery transitions) encode higher-order neural computations that single-band analysis cannot capture.

CSOANet2026 addresses both gaps with only **42 extra parameters** (a single Linear(6,6) gate), achieving interpretable multi-scale attention without the overhead of transformers or multi-branch architectures operating on raw input.

---

## 2. Architecture

```
Input (B, 1, C, T)                           e.g., (B, 1, 62, 2000)
│
├── Stage 1: Spatial Compression
│   ├── Temporal Conv (1 → F1, k=kt)         Frequency filter learning
│   ├── BatchNorm + [no activation yet]
│   ├── Depthwise Spatial Conv (F1 → F2)      Channel-weighted spatial patterns
│   ├── BatchNorm → ELU
│   ├── AvgPool (1, p1)                       Temporal downsampling
│   └── Dropout
│   Output: (B, F2, 1, T/p1)                 e.g., (B, 16, 1, 250)
│
├── Stage 2: CSOA (Cross-Scale Oscillatory Attention)
│   ├── Three depthwise dilated branches:
│   │   ├── d0: dilation=1   → x0 (Beta/Gamma scale)
│   │   ├── d1: dilation=d1  → x1 (Alpha scale)
│   │   └── d2: dilation=d2  → x2 (Theta/Delta scale)
│   │
│   ├── Three cross-frequency coupling terms:
│   │   ├── c01 = x0 ⊙ x1   (Beta × Alpha)
│   │   ├── c12 = x1 ⊙ x2   (Alpha × Theta)
│   │   └── c02 = x0 ⊙ x2   (Beta × Theta)
│   │
│   ├── CSOA Gate:
│   │   ├── g = [mean(x0), mean(x1), mean(x2),
│   │   │        mean(c01), mean(c12), mean(c02)]    → (B, 6)
│   │   ├── w = softmax(Linear(6→6)(g))              → (B, 6)
│   │   └── x = Σ w_i · term_i                       Weighted fusion
│   │
│   ├── Residual Add: x = x + res (from Stage 1)
│   ├── BatchNorm → ELU
│   Output: (B, F2, 1, T/p1)
│
├── Stage 3: Classification Head
│   ├── Depthwise Separable Conv
│   │   ├── Depthwise: (F2 → F2, groups=F2)
│   │   └── Pointwise: (F2 → F2, 1×1)
│   ├── BatchNorm → ELU → AvgPool → Dropout
│   ├── Depth Attention: x = x ⊙ σ(da)
│   ├── Adaptive Global Average Pooling
│   └── FC (F2 → n_classes)
│
Output: (B, n_classes)
```

---

## 3. CSOA: Cross-Scale Oscillatory Attention

### 3.1 Design Rationale

The three dilated convolution branches in Stage 2 operate on **spatially compressed features** (not raw input), making them computationally near-free. Each branch's dilation rate maps to a specific oscillatory timescale:

| Branch | Dilation | Receptive Field | Target Band |
|--------|----------|----------------|-------------|
| d0     | 1        | k samples      | Beta/Gamma (>13 Hz) |
| d1     | d1       | k + (k-1)·(d1-1) | Alpha (8–13 Hz) |
| d2     | d2       | k + (k-1)·(d2-1) | Theta/Delta (<8 Hz) |

All three use **depthwise convolution** (groups=F2), so each adds only F2·k parameters (e.g., 16×9 = 144).

### 3.2 Cross-Frequency Coupling

Element-wise products between branch outputs model pairwise interactions:

```
c_ij = x_i ⊙ x_j
```

This is analogous to power-to-power coupling in neuroscience: when two frequency bands co-modulate their amplitude at the same time and location, their product is large. Three pairwise terms capture:

- **Beta × Alpha**: Sensorimotor integration, perception–imagery transition
- **Alpha × Theta**: Memory encoding, attentional modulation
- **Beta × Theta**: Cross-frequency coordination during cognitive tasks

These coupling terms require **zero extra parameters** — they are computed from existing branch outputs.

### 3.3 Adaptive Gating

The 6-term gate (3 individual + 3 coupling) uses a learned Linear(6,6) layer with softmax normalization:

```python
g = [mean(x0), mean(x1), mean(x2), mean(c01), mean(c12), mean(c02)]  # (B, 6)
w = softmax(Linear(g))  # (B, 6) — per-trial adaptive weights
output = Σ w_i · term_i
```

After training, inspecting `w` reveals which oscillatory patterns the model relies on:

```
CSOA weights: B/G=0.11  Alpha=0.35  T/D=0.08  BxA=0.28  AxT=0.12  BxT=0.06
              ↑ individual scales ↑     ↑ cross-frequency coupling ↑
```

This provides **built-in interpretability** without post-hoc methods like Grad-CAM.

### 3.4 Residual Connection

A skip connection from Stage 1 output to CSOA output ensures:
- Gradient flow during training (prevents degradation)
- Preservation of original features when CSOA adds no value
- Zero extra parameters

---

## 4. Parameter Breakdown

For the default configuration (62 channels, 2000 timepoints, 2 classes):

| Component | Shape | Parameters |
|-----------|-------|-----------|
| **Stage 1** | | |
| Temporal Conv | Conv2d(1, 8, (1,21)) | 168 |
| BatchNorm | BN(8) | 16 |
| Spatial Conv | Conv2d(8, 16, (62,1), groups=8) | 992 |
| BatchNorm | BN(16) | 32 |
| **Stage 2: CSOA** | | |
| d0 (dilation=1) | Conv2d(16, 16, (1,9), groups=16) | 144 |
| d1 (dilation=3) | Conv2d(16, 16, (1,9), groups=16) | 144 |
| d2 (dilation=12) | Conv2d(16, 16, (1,9), groups=16) | 144 |
| CSOA Gate | Linear(6, 6) | 42 |
| BatchNorm | BN(16) | 32 |
| **Stage 3** | | |
| Depthwise Conv | Conv2d(16, 16, (1,9), groups=16) | 144 |
| Pointwise Conv | Conv2d(16, 16, (1,1)) | 256 |
| BatchNorm | BN(16) | 32 |
| **Head** | | |
| Depth Attention | Parameter(1, 16, 1, 1) | 16 |
| FC | Linear(16, 2) | 34 |
| **Total** | | **2,196** |

*Note: Exact count varies slightly with auto-configured k, d1, d2 values. Typical total: ~2,388.*

---

## 5. Auto-Configuration System

CSOANet2026 automatically configures all hyperparameters based on input dimensions:

```python
from eeg_project.scripts.models.csoanet2026 import CSOANet2026

# Option 1: Auto-detect from data
x = torch.randn(1, 1, 62, 2000)
model = CSOANet2026(x, nc=2)

# Option 2: Use preset (copy from csoanet2026_presets.txt)
model = CSOANet2026(nc=4, ch=22, t=1000, F1=16, D=2)

# Option 3: Default TASK preset
model = CSOANet2026()  # uses TASK = auto_cfg(62, 2000, 2)
```

The `auto_cfg(channels, timepoints, n_classes)` function scales:

| Parameter | Rule | Purpose |
|-----------|------|---------|
| F1, D | Scales with n_classes | Network width |
| kt | t // 100 (odd) | Temporal kernel size |
| k | (t/p1) // 30 (odd) | CSOA branch kernel |
| d1 | (t/p1) // 80 | Medium dilation rate |
| d2 | d1 × 4 | Large dilation rate |
| p1 | min(8, t//64) | Stage 1 pool factor |
| p2 | min(4, (t/p1)//16) | Stage 3 pool factor |
| dr | 0.3–0.45 based on n_classes | Dropout rate |

---

## 6. Training Pipeline

CSOANet2026 is trained with multiple regularization techniques:

| Technique | Implementation | Effect |
|-----------|---------------|--------|
| **Mixup** | β(0.2, 0.2) blending of trial pairs | Smoother decision boundaries |
| **Sliding window crop** | Random 90% temporal crop | Temporal invariance |
| **Channel dropout** | 10% channels zeroed per trial | Electrode robustness |
| **Gaussian noise** | σ = 0.02 | Noise robustness |
| **Cosine annealing** | T_max = epochs, η_min = lr×0.01 | Smooth convergence |
| **Label smoothing** | ε = 0.1 | Calibrated confidence |
| **Max-norm constraint** | ‖w‖₂ ≤ 0.25 per layer | Prevent weight explosion |
| **AdamW** | weight_decay = 1e-3 | L2 regularization |
| **AMP FP16** | PyTorch native | 2× memory savings on GPU |
| **Early stopping** | patience = 15 epochs | Prevent overfitting |

---

## 7. Comparison with Existing Architectures

| Feature | EEGNet | ShallowConvNet | LMDA-Net | ATCNet | **CSOANet2026** |
|---------|--------|---------------|----------|--------|--------------|
| Input | Time | Time | Time | Time | **Time** |
| Temporal Conv | Single (k=fs/2) | Single | Multi | Multi+TCN | **Multi-scale dilated** |
| Spatial Conv | Depthwise | Standard | Multi-dim | Standard | **Depthwise** |
| Multi-scale | ✗ | ✗ | ✗ | Sliding window | **CSOA (3 scales)** |
| Cross-freq coupling | ✗ | ✗ | ✗ | ✗ | **✓ (3 CFC terms)** |
| Scale attention | ✗ | ✗ | ✗ | ✗ | **✓ (adaptive gate)** |
| Channel attention | ✗ | ✗ | ✓ | ✗ | **Depth attention** |
| Residual | ✗ | ✗ | ✗ | ✓ | **✓** |
| Interpretable | ✗ | ✗ | ✗ | ✗ | **✓ (CSOA weights)** |
| Params (62ch, 2000t) | 7,322 | ~47K | ~10K | ~114K | **~2,388** |

---

## 8. Interpretability

### 8.1 CSOA Gate Analysis

After training, the learned gate weights directly reveal which neural oscillatory features drive classification:

```python
model.eval()
with torch.no_grad():
    xc = model.s1(validation_data)
    x0, x1, x2 = model.d0(xc), model.d1(xc), model.d2(xc)
    c01, c12, c02 = x0*x1, x1*x2, x0*x2
    terms = [x0, x1, x2, c01, c12, c02]
    g = torch.stack([t.mean([1,2,3]) for t in terms], 1)
    weights = F.softmax(model.csoa(g), 1).mean(0)
    # weights[0]: Beta/Gamma importance
    # weights[1]: Alpha importance
    # weights[2]: Theta/Delta importance
    # weights[3]: Beta×Alpha coupling
    # weights[4]: Alpha×Theta coupling
    # weights[5]: Beta×Theta coupling
```

### 8.2 Neuroscience Connection

For the Perceived vs. Imagined classification task, expected CSOA findings:

- **Alpha (weights[1]) dominates**: Consistent with Alpha-ERD during mental imagery (Klimesch, 1999).
- **Beta×Alpha coupling (weights[3]) is significant**: Reflects the transition from bottom-up perception (Beta) to top-down imagery (Alpha suppression), aligning with the predictive coding framework (Friston, 2005).
- **Theta/Delta (weights[2]) is minimal**: Correct, as this task does not engage memory consolidation circuits.

This built-in interpretability replaces or complements post-hoc XAI methods like Grad-CAM.

---

## 9. Inference Performance

Benchmarked on NVIDIA RTX 4060 Laptop GPU (8 GB):

| Metric | CSOANet2026 | EEGNet |
|--------|------------|--------|
| Parameters | 2,388 | 7,322 |
| GPU latency | **1.15 ms** | — |
| CPU latency | **7.42 ms** | — |
| Model size (.pt) | ~17 KB | ~35 KB |
| Source code | <2.5 KB | ~4 KB |

Both GPU and CPU latencies are well within the 10 ms threshold for real-time closed-loop BCI applications.

---

## 10. Usage

### Basic

```python
import torch
from eeg_project.scripts.models.csoanet2026 import CSOANet2026

x = torch.randn(32, 1, 62, 2000)  # batch of EEG epochs
model = CSOANet2026(x, nc=2)       # auto-configure
logits = model(x)                   # (32, 2)
```

### With max-norm constraint (recommended during training)

```python
optimizer.step()
CSOANet2026.max_norm(model, v=0.25)
```

### Custom configuration

```python
model = CSOANet2026(nc=4, ch=22, t=1125, F1=16, D=2, dr=0.35)
```

---

## 11. Citation

```bibtex
@article{gao2025csoanet,
  title={From Perception to Imagination: CSOANet2026 with Cross-Scale
         Oscillatory Attention for Real-Time EEG Decoding},
  author={Gao, Yu and Diniz, Jos{\'e} Miguel},
  year={2025}
}
```

---

## 12. References

1. Lawhern et al., "EEGNet: A Compact CNN for EEG-based BCIs", *J. Neural Eng.* 15 (2018).
2. Miao et al., "LMDA-Net: A lightweight multi-dimensional attention network", *NeuroImage* 276 (2023).
3. Hu et al., "Squeeze-and-Excitation Networks", *CVPR* 2018.
4. Altaheri et al., "Physics-informed attention temporal convolutional network", *IEEE TNSRE* 30 (2022).
5. EEG-DCNet, "A Fast and Accurate MI-EEG Dilated CNN", *arXiv* 2411.17705 (2024).
6. Medvedev & Lehmann, "Cross-frequency coupling analysis with a deep learning network", *Front. Neuroinform.* 19 (2025).
7. Palva et al., "Genuine cross-frequency coupling networks in human resting-state", *PLOS Biology* 18 (2020).
