"""Checks for CSOANetV2.  Run: python deploy/test_v2.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.profile_model import profile_model  # noqa: E402
from deploy.quantize import export_fp32, float_ops, quantize_int8, run_onnx  # noqa: E402
from eeg_project.scripts.models.csoanet2026 import CSOANet2026, auto_cfg  # noqa: E402
from eeg_project.scripts.models.csoanet2026_v2 import CSOANetV2  # noqa: E402

SHAPES = [(64, 640, 4), (62, 2000, 2), (22, 1000, 4), (3, 1000, 2)]
FLAGS = [dict(), dict(gate="dynamic"), dict(bound=False), dict(act="elu"),
         dict(gate="dynamic", bound=False, act="elu")]


def randomise(model, seed=0):
    """Give BN layers and the gate non-trivial values, as after training."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.3)
            m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(m.num_features, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(m.num_features, generator=g) * 0.3)
    if hasattr(model, "gate_logits"):
        model.gate_logits.data.copy_(torch.randn(6, generator=g))
    return model.eval()


@torch.no_grad()
def test_fuse_matches_unfused():
    for ch, t, nc in SHAPES:
        for kw in FLAGS:
            m = randomise(CSOANetV2(ch, t, nc, **kw))
            x = torch.randn(8, 1, ch, t)
            a, b = m(x), m.fuse()(x)
            assert a.shape == (8, nc)
            assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (ch, t, nc, kw, float((a - b).abs().max()))
            assert not any(isinstance(mod, (nn.BatchNorm2d, nn.Dropout)) for mod in m.fuse().modules())


@torch.no_grad()
def test_decimation_is_input_subsampling():
    ch, t, nc, d = 62, 2000, 2, 4
    a = randomise(CSOANetV2(ch, t, nc, decim=d))
    b = randomise(CSOANetV2(ch, t // d, nc))
    b.load_state_dict(a.state_dict())
    x = torch.randn(4, 1, ch, t)
    assert torch.allclose(a(x), b(x[..., ::d]), atol=1e-5)


@torch.no_grad()
def test_bounded_terms_and_gate():
    m = randomise(CSOANetV2(64, 640, 4))
    x = torch.randn(8, 1, 64, 640) * 5
    terms = m._terms(m.stage1(x))
    assert all(float(tm.abs().max()) <= 1.0 for tm in terms)
    for gate in ("static", "dynamic"):
        w = randomise(CSOANetV2(64, 640, 4, gate=gate)).gate_weights(x)
        assert w.shape == (6,) and abs(float(w.sum()) - 1) < 1e-5


def test_trains():
    torch.manual_seed(0)
    m = CSOANetV2(16, 320, 2)
    x, y = torch.randn(32, 1, 16, 320), torch.randint(0, 2, (32,))
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    first = None
    for _ in range(40):
        loss = nn.functional.cross_entropy(m(x), y)
        first = first or float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        CSOANetV2.max_norm(m)
    assert float(loss) < first, (first, float(loss))


def test_onnx_and_int8():
    torch.manual_seed(0)
    ch, t, nc = 64, 640, 4
    m = randomise(CSOANetV2(ch, t, nc))
    X = torch.randn(96, 1, ch, t)
    with torch.no_grad():
        ref = m(X).numpy()
    with tempfile.TemporaryDirectory() as d:
        fp = export_fp32(m.fuse(), X[:1], Path(d) / "m.onnx")
        assert np.allclose(run_onnx(fp, X.numpy()), ref, atol=1e-4)
        q = quantize_int8(fp, X[:64].numpy(), Path(d) / "q.onnx")
        out = run_onnx(q, X[64:].numpy())
        assert out.shape == (32, nc) and np.isfinite(out).all()
        assert float_ops(q) == [], float_ops(q)          # the whole v2 graph runs in int8


def test_cheaper_than_v1():
    for ch, t, nc in [(64, 640, 4), (62, 2000, 2)]:
        x = torch.randn(1, 1, ch, t)
        v1 = profile_model(CSOANet2026(auto_cfg(ch, t, nc), nc=nc), x)
        v2 = profile_model(CSOANetV2(ch, t, nc).fuse(), x)
        assert v2["macs"] < v1["macs"] / 3 and v2["peak_ram_bytes"] < v1["peak_ram_bytes"] / 5


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
