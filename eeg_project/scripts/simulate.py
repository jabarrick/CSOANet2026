from typing import List, Tuple
import numpy as np
import mne


def generate_simulated_raw(
    n_channels: int = 16,
    sfreq: float = 256.0,
    n_trials: int = 30,
    random_state: int = 42,
) -> mne.io.BaseRaw:
    """
    生成带注释（perception/imagimation/rest）的模拟 EEG 原始数据。

    一个试次时序：perception(1s) -> imagination(3s) -> rest(12s)。
    """
    rng = np.random.default_rng(random_state)

    # 选择 16 个常见 10-20 EEG 通道名（包含 Fp1 以便 ICA EOG 检测示例）
    ch_names = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    ][:n_channels]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")
    except Exception:
        pass

    trial_durations = {
        "perception": 1.0,
        "imagination": 3.0,
        "rest": 12.0,
    }
    order = ["perception", "imagination", "rest"]
    cycle_len = sum(trial_durations[s] for s in order)
    total_duration = n_trials * cycle_len
    n_samples = int(total_duration * sfreq)
    times = np.arange(n_samples) / sfreq

    # 基础噪声 + alpha 波
    noise_scale = 1e-6
    base = rng.normal(scale=noise_scale, size=(n_channels, n_samples))
    phases = rng.uniform(0, 2 * np.pi, size=n_channels)
    alpha_amp = 2.0e-6
    base += (alpha_amp * np.sin(2 * np.pi * 10.0 * times)[None, :] * np.cos(phases)[:, None])

    data = base.copy()

    # 各阶段的调制
    mod = {
        "perception": (18.0, 6.0e-6),  # 高频更强且幅度更大
        "imagination": (10.0, 4.5e-6),
        "rest": (6.0, 1.0e-6),
    }

    onsets: List[float] = []
    durations: List[float] = []
    descriptions: List[str] = []

    t0 = 0.0
    for _ in range(n_trials):
        for stage in order:
            d = trial_durations[stage]
            f, a = mod[stage]
            mask = (times >= t0) & (times < t0 + d)
            rel_t = times[mask] - t0
            data[:, mask] += a * np.sin(2 * np.pi * f * rel_t)[None, :]

            # 阶段特异 ERP 峰，提升可分性
            if stage == "perception":
                for peak_t, amp in [(0.10, 8e-6), (0.17, -6e-6), (0.25, 5e-6)]:
                    gaussian = np.exp(-0.5 * ((rel_t - peak_t) / 0.02) ** 2)
                    data[:, mask] += amp * gaussian[None, :]
            elif stage == "imagination":
                for peak_t, amp in [(0.20, 5e-6), (0.30, -4e-6)]:
                    gaussian = np.exp(-0.5 * ((rel_t - peak_t) / 0.03) ** 2)
                    data[:, mask] += amp * gaussian[None, :]

            onsets.append(t0)
            durations.append(d)
            descriptions.append(stage)

            t0 += d

    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))
    return raw


