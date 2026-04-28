from typing import Dict, List, Tuple
import numpy as np
import mne


def extract_erp_features(epochs: mne.Epochs) -> np.ndarray:
    """提取每个 epoch 在若干关键时间点处的通道电位，形成 epoch 级 ERP 特征。"""
    times_of_interest = [0.1, 0.17, 0.25, 0.35]
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    # 将时间转为索引（兼容旧版 API）
    idxs: List[int] = []
    for t in times_of_interest:
        try:
            idx = epochs.time_as_index(t)[0]
        except Exception:
            idx = int(np.argmin(np.abs(epochs.times - t)))
        idxs.append(idx)

    # 取这些时间点处的电位并拼接
    feats = [data[:, :, i] for i in idxs]  # list of (n_epochs, n_channels)
    X = np.hstack(feats)  # (n_epochs, n_channels * len(times))
    return X


def extract_spectral_features(epochs: mne.Epochs) -> Dict[str, np.ndarray]:
    """Welch 功率谱密度，返回各频带功率的均值。

    兼容 MNE 0.23（使用 mne.time_frequency.psd_welch）和 1.x（优先调用 Epochs.compute_psd）。
    """
    try:
        # MNE 1.x API
        psds, freqs = epochs.compute_psd(
            method="welch", fmin=1, fmax=40, n_fft=256
        ).get_data(return_freqs=True)
    except Exception:
        # MNE 0.23 兼容路径
        from mne.time_frequency import psd_welch

        psds, freqs = psd_welch(epochs, fmin=1, fmax=40, n_fft=256)
    # psds: (n_epochs, n_channels, n_freqs)

    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
    }

    band_powers: Dict[str, np.ndarray] = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_powers[band] = psds[:, :, idx].mean(axis=2)

    return band_powers


def extract_tf_features(epochs: mne.Epochs) -> np.ndarray:
    """计算 ERDS 时频图后在时间窗上取平均作为特征（兼容 MNE 0.20+）。"""
    freqs = np.arange(4, 40, 2)
    n_cycles = freqs / 2
    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=False,  # 返回每个 epoch 的 TFR
    )

    # 选取时间窗口
    time_windows: List[Tuple[float, float]] = [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0)]
    tf_features: List[np.ndarray] = []
    for tmin, tmax in time_windows:
        mask = (power.times >= tmin) & (power.times <= tmax)
        # power.data 形状: (n_epochs, n_channels, n_freqs, n_times)
        window_mean = power.data[..., mask].mean(axis=-1).mean(axis=-1)
        tf_features.append(window_mean)  # (n_epochs, n_channels)

    return np.hstack(tf_features)  # (n_epochs, n_channels * n_windows)


def prepare_features(
    epochs: mne.Epochs, feature_types: List[str] = None, scale: bool = False
):
    """汇总多种特征，返回 X, y。

    注意：默认不在此处做全局标准化，避免信息泄漏。
    如需在此标准化，可传入 scale=True，但推荐在模型 Pipeline 中完成。
    """

    if feature_types is None:
        feature_types = ["erp", "spectral"]

    blocks: List[np.ndarray] = []

    if "erp" in feature_types:
        blocks.append(extract_erp_features(epochs))

    if "spectral" in feature_types:
        band_powers = extract_spectral_features(epochs)
        # 频带顺序固定
        X_band = np.hstack([
            band_powers["theta"], band_powers["alpha"], band_powers["beta"],
        ])
        blocks.append(X_band)

    if "tfr" in feature_types:
        blocks.append(extract_tf_features(epochs))

    X = np.hstack(blocks) if blocks else np.array([])

    # 使用事件的第三列作为标签（事件码）
    y = epochs.events[:, 2]

    if scale:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X, y


