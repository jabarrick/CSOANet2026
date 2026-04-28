import os
import argparse
from pathlib import Path
import numpy as np

# 使用无交互后端
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
from typing import List  # noqa: E402

from eeg_project.scripts.load_data import load_subject_raw  # noqa: E402
from eeg_project.scripts.preprocess import (  # noqa: E402
    preprocess_raw,
    run_ica,
    create_epochs,
)
from eeg_project.scripts.features import prepare_features  # noqa: E402
from eeg_project.scripts.train_eval import (  # noqa: E402
    train_classifiers,
    evaluate_best_classifier,
)
from eeg_project.scripts.visualize import (  # noqa: E402
    plot_cv_results,
    plot_confusion_matrix,
)


def _pick_class_names(epochs: mne.Epochs) -> List[str]:
    names = list(epochs.event_id.keys())
    # 优先使用 '1' 和 '2'
    if set(['1','2']).issubset(set(names)):
        return ['1','2']
    # 否则取前两类
    return names[:2]


def save_erp_and_topo(epochs: mne.Epochs, subject: str, out_dir: Path) -> None:
    class_names = _pick_class_names(epochs)
    if len(class_names) < 2:
        print("可用事件不足 2 类，跳过 ERP/Topomap 保存。可用:", class_names)
        return

    evokeds = []
    for cname in class_names:
        try:
            evokeds.append((cname, epochs[cname].average()))
        except Exception:
            # 尝试按整数 ID 访问
            try:
                cid = epochs.event_id.get(cname)
                evokeds.append((cname, epochs[cid].average()))
            except Exception as e:
                print(f"无法生成 ERP: {cname}", e)
                return

    # ERP 时序曲线（避免依赖通道位置信息）
    for cname, evk in evokeds:
        fig = evk.plot(spatial_colors=False, selectable=False, show=False)
        fig.savefig(out_dir / f"erp_timeseries_class{cname}_sub{subject}.png", dpi=160)

    # Topomap at critical latencies
    for cname, evk in evokeds:
        try:
            fig = evk.plot_topomap(times=[0.10, 0.17, 0.25, 0.35], ch_type="eeg", show=False)
            fig.savefig(out_dir / f"erp_topomap_class{cname}_sub{subject}.png", dpi=160)
        except Exception as e:
            print("Topomap 绘制失败，跳过:", e)


def save_tfr_topo(epochs: mne.Epochs, subject: str, out_dir: Path) -> None:
    # 输出通道平均的时频图，避免拓扑依赖与高内存
    freqs = np.arange(4, 40, 4)
    n_cycles = freqs / 2
    times = epochs.times
    for cls in _pick_class_names(epochs):
        power = mne.time_frequency.tfr_morlet(
            epochs[cls], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True, decim=2, n_jobs=1
        )
        # power.data 形状: (n_channels, n_freqs, n_times)
        data_mean = power.data.mean(axis=0)  # (n_freqs, n_times)
        plt.figure(figsize=(6, 3))
        extent = [times[0], times[-1], freqs[0], freqs[-1]]
        plt.imshow(data_mean, aspect='auto', origin='lower', extent=extent, cmap='magma')
        plt.colorbar(label='Power (a.u.)')
        plt.xlabel('Time (s)'); plt.ylabel('Freq (Hz)')
        plt.title(f'TFR mean (class {cls}, sub{subject})')
        plt.tight_layout()
        plt.savefig(out_dir / f"tfr_mean_class{cls}_sub{subject}.png", dpi=160)
        plt.close()


def save_classification_figs(X, y, out_dir: Path, subject: str) -> None:
    results = train_classifiers(X, y)
    fig1 = plot_cv_results(results)
    fig1.savefig(out_dir / f"cv_accuracy_sub{subject}.png", dpi=160)
    clf, cm, report, best_name = evaluate_best_classifier(X, y, results)
    fig2 = plot_confusion_matrix(cm)
    fig2.savefig(out_dir / f"confusion_matrix_sub{subject}.png", dpi=160)
    print("使用最佳模型:", best_name)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ERP/Topomap/TFR/Accuracy/ConfusionMatrix figures")
    parser.add_argument("--subject", default="02", help="subject id, e.g., 01/02/03")
    args = parser.parse_args()

    # 输出到按受试者分目录的路径，如 figures/01、figures/02、figures/03
    out_dir = Path("eeg_project") / "figures" / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_subject_raw(subject_id=args.subject)
    raw_filt = preprocess_raw(raw.copy())
    raw_clean, _ = run_ica(raw_filt)
    epochs = create_epochs(raw_clean)
    # 尝试设置标准电极位置信息，若不可用则继续使用时间序列图
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore")
    except Exception:
        pass

    # ERP & Topomap & TFR
    save_erp_and_topo(epochs, args.subject, out_dir)
    save_tfr_topo(epochs, args.subject, out_dir)

    # Classification figs
    X, y = prepare_features(epochs, ["erp", "spectral"], scale=False)
    save_classification_figs(X, y, out_dir, args.subject)

    print("图已导出到:", out_dir)


if __name__ == "__main__":
    main()


