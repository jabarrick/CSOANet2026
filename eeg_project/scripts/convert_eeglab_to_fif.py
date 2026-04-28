import os
from typing import List, Tuple
import mne


def convert_one(src_set_path: str, dst_fif_path: str) -> None:
    """读取 .set 文件并保存为 .fif 格式"""
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(dst_fif_path), exist_ok=True)

    try:
        # 读取 EEGLAB .set 文件 (preload=True 以便保存)
        # 注意：如果遇到报错，请确保 numpy 版本是 1.23.5 (pip install numpy==1.23.5)
        raw = mne.io.read_raw_eeglab(src_set_path, preload=True, verbose="ERROR")

        # 保存为 .fif 文件
        raw.save(dst_fif_path, overwrite=True)
    except Exception as e:
        print(f"❌ 转换失败 {src_set_path}: {e}")


def main() -> None:
    cases: List[Tuple[str, str]] = []

    # === 自动化生成 1 到 54 号被试的路径 ===
    # range(1, 55) 会产生 1, 2, ..., 54
    for i in range(39, 41):
        # 将数字格式化为两位数字符串，例如 1 -> "01", 54 -> "54"
        sid = f"{i:02d}"

        # 1. 构建源文件路径 (OpenNeuro 格式: sub-01)
        # 文件夹: openneuro_ds005672/sub-01/eeg/
        # 文件名: sub-01_task-PerceiveImagine_eeg.set
        src_folder = f"sub-{sid}"
        src_filename = f"sub-{sid}_task-PerceiveImagine_eeg.set"
        src_path = os.path.join("openneuro_ds005672", src_folder, "eeg", src_filename)

        # 2. 构建目标文件路径 (项目内部格式: sub01)
        # 文件夹: eeg_project/data/raw/sub01/eeg/
        # 文件名: sub01_taskimagine_eeg.fif
        dst_folder = f"sub{sid}"
        dst_filename = f"sub{sid}_taskimagine_eeg.fif"
        dst_path = os.path.join("eeg_project", "data", "raw", dst_folder, "eeg", dst_filename)

        # 加入列表
        cases.append((src_path, dst_path))

    # === 开始批量处理 ===
    print(f"准备处理 {len(cases)} 个文件 (Subject 01-54)...")

    count_success = 0
    count_skip = 0

    for src, dst in cases:
        if not os.path.exists(src):
            # 如果源文件不存在（比如只有 sub-01 到 sub-10，后面没有），就跳过
            # print(f"[跳过] 源文件不存在: {src}")
            # (注释掉上面这行可以减少刷屏，只显示转换进度的信息)
            count_skip += 1
            continue

        print(f"[转换中] Subject {src.split('sub-')[1][:2]} ...")
        convert_one(src, dst)
        count_success += 1

    print(f"\n全部结束: 成功 {count_success} 个, 跳过 {count_skip} 个。")


if __name__ == "__main__":
    main()

