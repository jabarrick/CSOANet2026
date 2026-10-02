"""Static resource profile of EEGNet and CSOANet2026 v1 / v2, read off the exported ONNX graph.

Reports, per model:
  params      weights stored in the graph after BN folding
  flash       the same weights as int8 (biases as int32)
  MACs        multiply-accumulates of Conv / Gemm, plus element-wise tensor products
  peak RAM    largest set of activations alive at once, int8 (1 byte per value)
  RAM (stream) the same without the raw input window.  This applies when the
              first layer consumes samples as they arrive: v2's spatial
              projection can, v1's temporal conv on raw channels gains nothing

Peak RAM is a liveness estimate over the graph in execution order, which is
what a TFLite-Micro style arena planner does.  ReLU and Clip are treated as
fused into the op before them; ELU allocates its own output.  These are
estimates for comparing architectures, not measurements on a board.

    python deploy/profile_model.py                       # the three reference shapes
    python deploy/profile_model.py --ch 22 --t 1000 --nc 4
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.baselines import make_eegnet  # noqa: E402
from eeg_project.scripts.models.csoanet2026 import CSOANet2026, auto_cfg  # noqa: E402
from eeg_project.scripts.models.csoanet2026_v2 import CSOANetV2  # noqa: E402

FUSED_ACTIVATIONS = {"Relu", "Clip"}          # no tensor of their own on device
FREE_OPS = {"Flatten", "Reshape", "Squeeze", "Unsqueeze", "Identity"}


def export_onnx(model, example, path):
    """Export an eval-mode model with a fixed batch of 1."""
    model = model.eval()
    kw = dict(input_names=["eeg"], output_names=["logits"], opset_version=17, do_constant_folding=True)
    with torch.no_grad():
        try:
            torch.onnx.export(model, example, str(path), dynamo=False, **kw)
        except TypeError:                      # older torch without the `dynamo` argument
            torch.onnx.export(model, example, str(path), **kw)
    return Path(path)


def profile_onnx(path):
    g = shape_inference.infer_shapes(onnx.load(str(path))).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    size = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        size[vi.name] = int(np.prod(dims)) if dims else 1
    const_out = {o for n in g.node if n.op_type == "Constant" for o in n.output}

    def is_act(name):
        return name and name not in init and name not in const_out

    # weights: conv / linear tensors and their biases
    params = flash = 0
    for n in g.node:
        if n.op_type in ("Conv", "Gemm", "MatMul", "BatchNormalization"):
            for j, name in enumerate(n.input[1:], 1):
                if name in init:
                    params += init[name].size
                    flash += init[name].size * (4 if (j == 2 and n.op_type != "BatchNormalization") else 1)

    # MACs
    macs = {"conv": 0, "linear": 0, "elementwise_mul": 0}
    per_layer = []
    for n in g.node:
        out = size.get(n.output[0], 0)
        if n.op_type == "Conv":
            w = init[n.input[1]]
            m = out * int(np.prod(w.shape[1:]))
            macs["conv"] += m
            per_layer.append((n.name or n.output[0], "Conv", tuple(w.shape), m))
        elif n.op_type in ("Gemm", "MatMul"):
            w = init.get(n.input[1])
            if w is not None:
                macs["linear"] += w.size
        elif n.op_type == "Mul" and all(is_act(i) for i in n.input):
            macs["elementwise_mul"] += out

    # liveness: map fused activations / shape ops onto the tensor they alias
    alias = {}

    def root(name):
        while name in alias:
            name = alias[name]
        return name

    for n in g.node:
        if n.op_type in FUSED_ACTIVATIONS or n.op_type in FREE_OPS:
            src = next((i for i in n.input if is_act(i)), None)
            if src is not None:
                alias[n.output[0]] = src
    last_use = {}
    for idx, n in enumerate(g.node):
        for i in n.input:
            if is_act(i):
                last_use[root(i)] = idx
    for o in g.output:
        last_use[root(o.name)] = len(g.node)
    born = {root(i.name): -1 for i in g.input if i.name not in init}
    for idx, n in enumerate(g.node):
        if n.op_type == "Constant":
            continue
        for o in n.output:
            born.setdefault(root(o), idx)
    graph_inputs = {t for t, b in born.items() if b == -1}
    peak, peak_at, peak_stream = 0, "", 0
    for idx, n in enumerate(g.node):
        live = [t for t, b in born.items() if b <= idx and last_use.get(t, -2) >= idx]
        tot = sum(size.get(t, 0) for t in live)
        if tot > peak:
            peak, peak_at = tot, f"{n.op_type} -> {n.output[0]}"
        peak_stream = max(peak_stream, sum(size.get(t, 0) for t in live if t not in graph_inputs))
    return dict(params=params, flash_bytes=flash, macs=sum(macs.values()), macs_detail=macs,
                peak_ram_bytes=peak, peak_ram_stream_bytes=peak_stream, peak_at=peak_at,
                per_layer=per_layer)


def profile_model(model, example):
    with tempfile.TemporaryDirectory() as d:
        return profile_onnx(export_onnx(model, example, Path(d) / "m.onnx"))


def compare(ch, t, nc, decim=1, verbose=False, fs=160.0):
    """fs sets EEGNet's kernel lengths; fs=None uses the repository's 1 kHz EEGNet defaults."""
    x = torch.randn(1, 1, ch, t)
    variants = {
        "EEGNet-8,2": make_eegnet(ch, t, nc, fs),
        "v1 (published)": CSOANet2026(auto_cfg(ch, t, nc), nc=nc),
        "v2": CSOANetV2(ch, t, nc).fuse(),
    }
    if decim > 1:
        variants[f"v2, decim={decim}"] = CSOANetV2(ch, t, nc, decim=decim).fuse()
    rows = {k: profile_model(m, x) for k, m in variants.items()}
    base = rows["v1 (published)"]
    print(f"\ninput {ch} ch x {t} samples, {nc} classes")
    print(f"  {'model':<18}{'params':>8}{'flash':>10}{'MACs':>10}{'peak RAM':>11}{'RAM (stream)':>14}"
          f"{'MACs vs v1':>12}{'RAM vs v1':>11}")
    for k, r in rows.items():
        print(f"  {k:<18}{r['params']:>8,}{r['flash_bytes'] / 1024:>8.1f} KB{r['macs'] / 1e6:>8.2f} M"
              f"{r['peak_ram_bytes'] / 1024:>8.1f} KB{r['peak_ram_stream_bytes'] / 1024:>11.1f} KB"
              f"{base['macs'] / r['macs']:>11.1f}x{base['peak_ram_bytes'] / r['peak_ram_bytes']:>10.1f}x")
        if verbose:
            for name, op, shape, m in r["per_layer"]:
                print(f"      {op} {shape}: {m / 1e3:.1f} k MACs")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ch", type=int)
    ap.add_argument("--t", type=int)
    ap.add_argument("--nc", type=int, default=2)
    ap.add_argument("--decim", type=int, default=1)
    ap.add_argument("--fs", type=float, default=160.0, help="sampling rate, sets EEGNet's kernel lengths")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if a.ch and a.t:
        compare(a.ch, a.t, a.nc, a.decim, a.verbose, a.fs)
    else:
        compare(64, 640, 4, verbose=a.verbose)             # PhysioNet-MI, 160 Hz, 4 s
        compare(64, 640, 2, verbose=a.verbose)
        compare(62, 2000, 2, decim=4, verbose=a.verbose, fs=None)   # 1 kHz, 2 s (paper one's input shape)
