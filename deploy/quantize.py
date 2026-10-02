"""ONNX export and int8 post-training quantization for CSOANet2026 v1 / v2.

Produces a QDQ-format int8 ONNX file (per-channel symmetric int8 weights,
asymmetric int8 activations - the same scheme TFLite and CMSIS-NN use) and
runs it with ONNX Runtime, so the accuracy cost of int8 can be measured on a
PC before anything is flashed.

    from deploy.quantize import export_fp32, quantize_int8, run_onnx
    export_fp32(model.fuse(), X[:1], "csoa_v2.onnx")
    quantize_int8("csoa_v2.onnx", X_calib, "csoa_v2_int8.onnx")
    logits = run_onnx("csoa_v2_int8.onnx", X_test)

Run this file directly for a self-test on random input.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CALIB = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
         "entropy": CalibrationMethod.Entropy}


def export_fp32(model, example, path, dynamic_batch=True):
    """Export an eval-mode model.  For v2 pass `model.fuse()`."""
    kw = dict(input_names=["eeg"], output_names=["logits"], opset_version=17, do_constant_folding=True)
    if dynamic_batch:
        kw["dynamic_axes"] = {"eeg": {0: "batch"}, "logits": {0: "batch"}}
    model = model.eval()
    with torch.no_grad():
        try:
            torch.onnx.export(model, example, str(path), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(model, example, str(path), **kw)
    return Path(path)


class _Reader(CalibrationDataReader):
    def __init__(self, X, batch):
        self.batches = iter([{"eeg": X[i:i + batch]} for i in range(0, len(X), batch)])

    def get_next(self):
        return next(self.batches, None)


def quantize_int8(fp32_path, calib, out_path, method="minmax", per_channel=True, batch=32):
    """Static int8 quantization.  `calib`: float32 array (N, 1, C, T) of training trials."""
    calib = np.ascontiguousarray(calib, dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        pre = Path(d) / "pre.onnx"
        try:
            quant_pre_process(str(fp32_path), str(pre))
        except Exception:                       # symbolic shape inference can fail on dynamic graphs
            pre = Path(fp32_path)
        quantize_static(str(pre), str(out_path), _Reader(calib, batch),
                        quant_format=QuantFormat.QDQ, per_channel=per_channel,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CALIB[method],
                        extra_options={"WeightSymmetric": True, "ActivationSymmetric": False})
    return Path(out_path)


def run_onnx(path, X, batch=64):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    X = np.ascontiguousarray(X, dtype=np.float32)
    return np.concatenate([sess.run(None, {"eeg": X[i:i + batch]})[0] for i in range(0, len(X), batch)])


def float_ops(path):
    """Op types in a quantized graph that are not wrapped in Q/DQ, i.e. would run in float."""
    import onnx
    g = onnx.load(str(path)).graph
    init = {i.name for i in g.initializer}
    produced_by = {o: n.op_type for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n.op_type)
    shape_ops = {"QuantizeLinear", "DequantizeLinear", "Constant", "Shape", "Gather", "Unsqueeze", "Concat",
                 "Reshape", "Flatten", "Squeeze"}
    graph_out = {o.name for o in g.output}
    out = set()
    for n in g.node:
        if n.op_type in shape_ops:
            continue
        ins = [produced_by[i] for i in n.input if i in produced_by and i not in init]
        ins_ok = all(t == "DequantizeLinear" for t in ins)
        outs_ok = all(o in graph_out or all(c == "QuantizeLinear" for c in consumers.get(o, []))
                      for o in n.output)
        if not (ins_ok and outs_ok):
            out.add(n.op_type)
    return sorted(out)


def int8_report(model, X_calib, X_eval, workdir, name="model", method="minmax"):
    """Export + quantize `model`, return (fp32 logits, int8 logits, paths)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    fp, q = workdir / f"{name}_fp32.onnx", workdir / f"{name}_int8.onnx"
    example = torch.from_numpy(np.asarray(X_calib[:1], dtype=np.float32))
    export_fp32(model, example, fp)
    quantize_int8(fp, X_calib, q, method=method)
    return run_onnx(fp, X_eval), run_onnx(q, X_eval), (fp, q)


if __name__ == "__main__":
    from eeg_project.scripts.models.csoanet2026 import CSOANet2026, auto_cfg
    from eeg_project.scripts.models.csoanet2026_v2 import CSOANetV2

    torch.manual_seed(0)
    ch, t, nc = 64, 640, 4
    X = torch.randn(256, 1, ch, t).numpy()
    with tempfile.TemporaryDirectory() as d:
        for name, m in [("v1", CSOANet2026(auto_cfg(ch, t, nc), nc=nc)), ("v2", CSOANetV2(ch, t, nc).fuse())]:
            f, q, (_, qp) = int8_report(m, X[:128], X[128:], d, name)
            rel = np.abs(f - q).mean() / (np.abs(f).mean() + 1e-12)
            print(f"{name}: untrained, random input | argmax agreement {np.mean(f.argmax(1) == q.argmax(1)):.3f}"
                  f" | mean rel. logit error {rel:.3f} | ops left in float: {float_ops(qp) or 'none'}")
    print("Pipeline OK.  Agreement on random weights says nothing about accuracy on EEG;"
          " use train_physionet_mi.py for that.")
