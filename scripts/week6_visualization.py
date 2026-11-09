# 6주차: 최종 모델 성능 비교 그림
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2,<4", "matplotlib>=3.8"]
# ///
# How to run: python scripts/week6_visualization.py
# 먼저 python scripts/week6_summary.py를 실행합니다.
from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT: Final = Path(__file__).resolve().parents[1]


def save_figure() -> None:
    """기존 서체와 색상으로 랜덤 및 시간순 평가의 MAE와 RMSE를 비교합니다."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "axes.linewidth": .7, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    data = pd.read_csv(ROOT / "reports/week6_model_comparison.csv")
    labels = ["Mean-rating\nbaseline", "Structured\nRidge", "A: Text\nonly",
              "B: Struct.\n+ title", "C: Struct.\n+ body", "D: Struct.\n+ title + body"]
    models = data.loc[data.evaluation.eq("random"), "model"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    x, width = np.arange(len(models)), .36
    for axis, metric, title in zip(axes, ["mae", "rmse"], ["(a) Mean absolute error", "(b) Root mean squared error"]):
        for offset, evaluation, color, label in [(-.5, "random", "#9CAAB7", "Random holdout (n=729)"),
                                                 (.5, "temporal", "#263E56", "Temporal holdout (n=713)")]:
            values = data.loc[data.evaluation.eq(evaluation)].set_index("model").loc[models, metric]
            bars = axis.bar(x + offset * width, values, width, color=color, label=label)
            axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=7, rotation=90)
        axis.set_title(title, loc="left", pad=12)
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylabel(f"{metric.upper()} (lower is better)")
        axis.set_ylim(0, .205)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E7E7E7", linewidth=.5)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(.5, .015, "Different product cohorts: the gap does not isolate the effect of time alone.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .055, 1, 1), w_pad=2)
    output = ROOT / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"week6_final_comparison.{extension}", dpi=400, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)
    print(output / "week6_final_comparison.png")


if __name__ == "__main__":
    save_figure()
