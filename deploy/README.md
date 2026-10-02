# CSOANet2026-v2: MCU deployment

A deployment-oriented variant of CSOANet2026 for int8 Cortex-M class targets, plus the
tooling to profile it, quantize it and compare it against the published v1.

Unpack at the repository root (nothing in the existing code is modified), or use the
folder on its own: it carries a copy of v1's `csoanet2026.py`, so the scripts run from
any location and any working directory.

```
eeg_project/scripts/models/csoanet2026_v2.py   the model (train form + fuse() for export)
deploy/profile_model.py                        params, MACs, peak RAM, read off the ONNX graph
deploy/quantize.py                             ONNX export, int8 static quantization, int8 inference
deploy/train_physionet_mi.py                   v1 vs v2 on PhysioNet-MI, float and int8, cross-subject
deploy/test_v2.py                              checks
```

## What changed and why

| # | v1 | v2 | Why |
|---|----|----|-----|
| 1 | Temporal conv on all C raw channels, then spatial conv | Spatial conv first, then depthwise temporal conv | The temporal conv on raw channels is 84-94 % of v1's MACs and produces its largest tensor (F1 x C x T/2). Projecting to F2 virtual channels first removes both. The projection can run per sample as data arrives, so the raw window need not be buffered. |
| 2 | Input at native rate | Optional `decim` (stride in the spatial conv) | Input band-passed to 40 Hz does not need 1 kHz. Only valid when the input is band-limited below fs / (2 x decim). |
| 3 | Unbounded branch outputs and products | Branch BN (folded at export) + clip to [-1, 1] | All six CSOA terms and their weighted sum stay in [-1, 1] and share one fixed-point scale. Unbounded products square the dynamic range. |
| 4 | Per-trial gate: Linear(6,6) on global means | Six learned logits (`gate="static"`) | No second pass over the trial, no stored product maps. `gate="dynamic"` restores v1's gate. |
| 5 | ELU | ReLU | Fuses into the conv before it. |
| 6 | Depth attention `da` before the FC | Removed | A per-channel scale in front of a linear layer is redundant with that layer. |

Changes 3-5 are constructor flags, so each can be ablated separately. Kernel sizes
come from v1's `auto_cfg`, evaluated on the decimated length; with `decim=1` the two
models share all temporal hyper-parameters.

## Static profile

`python deploy/profile_model.py`

| Input | Model | Params | MACs | Peak RAM (int8) | Peak RAM, streaming input |
|-------|-------|-------:|-----:|----------------:|--------------------------:|
| 64 x 640, 4 classes (PhysioNet-MI) | v1 | 3,918 | 5.05 M | 640 KB | 640 KB |
| | v2 | 4,068 | 1.58 M | 60 KB | 30 KB |
| 64 x 640, 2 classes | v1 | 1,692 | 2.51 M | 320 KB | 320 KB |
| | v2 | 1,746 | 0.77 M | 50 KB | 15 KB |
| 62 x 2000, 2 classes (1 kHz, 2 s) | v1 | 2,268 | 21.57 M | 969 KB | 969 KB |
| | v2 | 2,546 | 2.88 M | 152 KB | 47 KB |
| | v2, decim=4 | 1,682 | 0.59 M | 129 KB | 12 KB |

How to read these numbers:

- Peak RAM is a tensor-liveness estimate at one byte per activation, the way a
  TFLite-Micro arena planner allocates. It is not a measurement on a board.
- ReLU and Clip are counted as fused into the previous op. ELU is counted as a
  separate op with its own output tensor, which accounts for half of v1's peak.
  A hand-written in-place ELU would bring v1 to about half the figure shown.
- "Streaming input" excludes the raw input window. It applies when the spatial
  projection is computed per sample at acquisition time.

Single-trial latency on an x86 CPU (ONNX Runtime, one thread, median of 300), for
direction only: 64 x 640, 4 classes: v1 1.70 ms, v2 0.09 ms in float32; v1 5.07 ms,
v2 0.56 ms in int8. Latency on the target has to be measured on the target.

## Accuracy: not measured yet

`train_physionet_mi.py` is the experiment that decides whether v2 is acceptable. It has
been run end to end on synthetic data and on stand-in EDF files, not on the real
recordings. Run it locally:

```bash
pip install -r deploy/requirements.txt
python deploy/test_v2.py
python deploy/train_physionet_mi.py --variants v1 v2 v2-dynamic v2-unbounded v2-elu
```

It trains each variant with v1's recipe on cross-subject folds, selects the epoch on
held-out validation subjects, and reports test accuracy per subject in float32 and in
int8, with a paired Wilcoxon test across subjects. The ablation variants show which of
changes 3-5 costs accuracy, if any does.

Two things to keep in mind when reading its output:

- ONNX Runtime's quantizer has no int8 kernel for ELU or ReduceMean, so in v1's
  "int8" graph those ops (and the pooling next to them) stay in float. v1's int8
  accuracy from this script is therefore an optimistic bound. v2's graph is int8
  throughout (`test_v2.py` asserts this).
- `--norm fixed` (default) uses one gain per channel from the training subjects,
  which folds into the spatial weights on the device. `--norm epoch_z` is v1's
  per-epoch z-score and needs the whole raw window in RAM.

## Getting it onto a board

`quantize.py` writes a QDQ-format int8 ONNX file (per-channel int8 weights, int8
activations). ST documents import of QDQ quantized ONNX in X-CUBE-AI
(https://wiki.st.com/stm32mcu/wiki/AI:X-CUBE-AI_support_of_ONNX_and_TensorFlow_quantized_models).
That import, and conversion to TFLite Micro, have not been tested here.
