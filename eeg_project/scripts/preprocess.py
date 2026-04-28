from typing import Dict, Tuple
import mne


def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """基础预处理：参考与带通滤波。"""
    # EEG average reference
    raw.set_eeg_reference("average", projection=True)
    # 1-40 Hz 带通
    raw.filter(l_freq=1.0, h_freq=40.0)
    return raw


def run_ica(raw: mne.io.BaseRaw, n_components: int = 15, random_state: int = 42) -> Tuple[mne.io.BaseRaw, mne.preprocessing.ICA]:
    """使用 ICA 去除伪迹（自动检测 EOG 成分）。"""
    from mne.preprocessing import ICA

    ica = ICA(n_components=n_components, random_state=random_state)
    ica.fit(raw)

    # 简单的自动 EOG 检测（Fp1 作为参考通道名，按需调整）
    try:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name="Fp1")
        ica.exclude = eog_indices
    except Exception:
        # 若通道名或数据不匹配，则跳过自动排除
        ica.exclude = []

    raw_clean = ica.apply(raw.copy())
    return raw_clean, ica


def create_epochs(
    raw: mne.io.BaseRaw,
    event_dict: Dict[str, int] = None,
    tmin: float = -0.5,
    tmax: float = 3.0,
) -> mne.Epochs:
    """根据注释生成时段（Epochs）。

    - 若未提供 event_dict，则进行简单自适应：
      - 如果注释中包含 '1'/'2' 这样的数字标签，优先仅使用 {'1':1, '2':2}，
        忽略诸如 '255' 等边界/占位标记；
      - 否则回退到项目默认映射 {perception, imagination, rest}。
    """

    if event_dict is None:
        # 尝试读取注释标签以做自适应
        ann = getattr(raw, "annotations", None)
        descriptions = set(ann.description) if ann is not None else set()
        if {"1", "2"}.issubset(descriptions):
            # 数据集为数字标签，且常见存在 '255'（忽略）
            event_dict = {"1": 1, "2": 2}
        else:
            event_dict = {"perception": 1, "imagination": 2, "rest": 3}

    events, _ = mne.events_from_annotations(raw, event_id=event_dict)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=(None, 0),
        preload=True,
    )

    epochs.drop_bad()
    return epochs


