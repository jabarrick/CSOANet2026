import os
from typing import Optional
import mne


def load_subject_raw(subject_id: Optional[str] = None, raw_file: Optional[str] = None) -> mne.io.BaseRaw:
    """
    加载单个受试者的原始 EEG 数据。

    优先使用 raw_file；否则按约定路径:
    eeg_project/data/raw/sub{subject_id}/eeg/sub{subject_id}_taskimagine_eeg.fif
    """
    if raw_file is None:
        if subject_id is None:
            raise ValueError("必须提供 subject_id 或 raw_file 之一")
        raw_file = os.path.join(
            "eeg_project",
            "data",
            "raw",
            f"sub{subject_id}",
            "eeg",
            f"sub{subject_id}_taskimagine_eeg.fif",
        )

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"未找到原始数据文件: {raw_file}")

    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose="ERROR")
    return raw


