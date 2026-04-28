import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# 设置全局绘图风格 (Seaborn + Matplotlib)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


def plot_cv_results(results):
    """
    【图1：模型对比箱线图】
    美化点：
    1. 叠加散点 (Swarmplot) 展示每次 CV 的具体分布。
    2. 标注平均分数值。
    3. 使用学术风配色 (Pastel)。
    """
    # 1. 数据转换: Dict -> DataFrame
    data = []
    for name, res in results.items():
        for score in res["scores"]:
            data.append({"Model": name, "Accuracy": score})
    df = pd.DataFrame(data)

    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))

    # 3. 绘制箱线图 (Boxplot) - 底层
    sns.boxplot(x="Model", y="Accuracy", data=df, ax=ax,
                palette="pastel", width=0.5, linewidth=2, fliersize=0)

    # 4. 绘制散点图 (Stripplot) - 顶层，展示真实数据点
    sns.stripplot(x="Model", y="Accuracy", data=df, ax=ax,
                  color=".3", size=8, jitter=True, alpha=0.7)

    # 5. 添加平均值文字
    means = df.groupby("Model")["Accuracy"].mean()
    for i, name in enumerate(df["Model"].unique()):
        mu = means[name]
        ax.text(i, mu + 0.01, f"{mu:.3f}",
                ha='center', va='bottom', fontweight='bold', color='black', fontsize=12)

    # 6. 细节修饰
    ax.set_title("Model Performance Comparison (5-Fold CV)", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Accuracy Score", fontsize=14, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylim(0.4, 1.05)  # 这里的 Y 轴范围根据需要调整
    sns.despine(trim=True)  # 去除上方和右侧边框

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names=["Perceive", "Imagine"]):
    """
    【图2：混淆矩阵热力图】
    美化点：
    1. 同时显示【数量】和【百分比】。
    2. 颜色深浅自动适配文字颜色（深底白字，浅底黑字）。
    3. 清晰的轴标签。
    """
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))

    # 绘制热力图 (不使用默认 annot，因为我们要自定义格式)
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=False,
                linewidths=1.5, linecolor='white', square=True, ax=ax)

    # 手动添加精美的文字标签
    rows, cols = cm.shape
    for i in range(rows):
        for j in range(cols):
            # 逻辑：如果背景太深，文字就用白色；否则用黑色
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"

            # 格式：数量 (换行) 百分比
            text = f"{cm[i, j]}\n({cm_percent[i, j]:.1%})"

            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center', color=text_color,
                    fontsize=16, fontweight='bold')

    # 轴标签设置
    ax.set_ylabel('True Label (真实)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label (预测)', fontsize=14, fontweight='bold')

    # 刻度设置
    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12, rotation=0)

    ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


# visualize.py
import ptitprince as pt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_raincloud_47(csv_path):
    """绘制 47 人准确率分布的雨云图"""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    # 💡 修复：将 y="accuracy" 改为 y="holdout_acc" 以对齐您的数据列名
    pt.RainCloud(y="holdout_acc", data=df, orient='h', ax=ax, palette="Set2", bw=.2)

    ax.axvline(0.5, color='r', linestyle='--', label='Chance (50%)')
    ax.set_title("Classification Accuracy Distribution (N=47)", fontsize=16)

    # 自动保存图片到 figures 目录
    save_path = Path(csv_path).parent.parent / "figures" / "raincloud_47.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 雨云图已保存: {save_path}")
    return fig


def plot_snr_correlation(csv_path):
    """绘制信号质量与准确率的相关性图"""
    df = pd.read_csv(csv_path)

    # 💡 检查数据中是否有 snr 列
    if 'snr' not in df.columns:
        print("⚠️ 警告：数据集中缺少 'snr' 列，无法绘制相关性图。请更新 WP6 汇总逻辑。")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x="snr", y="holdout_acc", data=df, ax=ax, scatter_kws={'s': 80}, line_kws={'color': 'red'})
    ax.set_title("SNR vs. Accuracy Correlation", fontsize=16)

    save_path = Path(csv_path).parent.parent / "figures" / "snr_correlation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 相关性图已保存: {save_path}")
    return fig