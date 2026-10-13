# 2주차 결과 시각화
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2,<4", "matplotlib>=3.8"]
# ///
# How to run: python scripts/week2_visualization.py
# 또는 uv run scripts/week2_visualization.py

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib
import pandas as pd  # 기존 작업자 스타일을 따르라는 사용자 요청

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT: Final = Path(__file__).resolve().parents[1]


def main() -> None:
    """1주차와 같은 서체·색상으로 분포와 관계를 시각화합니다."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "axes.linewidth": .7, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    products = pd.read_csv(ROOT / "reports/week2_products.csv")
    navy, light, accent = "#263E56", "#9CAAB7", "#A16B5D"
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    counts = products.followup_low_count.value_counts().reindex(range(21), fill_value=0)
    axes[0].bar(counts.index / 20, counts / len(products) * 100, width=.04, color=navy)
    axes[0].set(title="(a) Future low ratings", xlabel="Low-rating share in next 20 reviews",
                ylabel="Products (%)", xlim=(-.04, 1.04))
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    r = products.initial_mean_rating.corr(products.followup_low_ratio)
    axes[1].scatter(products.initial_mean_rating, products.followup_low_ratio,
                    s=7, alpha=.15, color=navy, edgecolors="none", rasterized=True)
    axes[1].set(title=f"(b) Initial ratings (r = {r:.3f})", xlabel="Mean rating in first 10 reviews",
                ylabel="Future low-rating share", ylim=(-.04, 1.04), xlim=(.9, 5.1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    groups = pd.cut(products.initial_mean_rating, [.99, 2, 3, 4, 5],
                    labels=["1–2", ">2–3", ">3–4", ">4–5"])
    samples = [products.loc[groups.eq(label), "followup_low_ratio"].to_numpy()
               for label in groups.cat.categories]
    boxes = axes[2].boxplot(samples, patch_artist=True,
                           flierprops={"markersize": 2, "alpha": .2},
                           medianprops={"color": accent, "linewidth": 1.1})
    axes[2].set_xticks(range(1, 5), list(groups.cat.categories))
    for box in boxes["boxes"]:
        box.set_facecolor(light)
    axes[2].set(title="(c) Variation within rating groups", xlabel="Initial mean-rating group",
                ylabel="Future low-rating share", ylim=(-.04, 1.13))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    for index, sample in enumerate(samples, 1):
        axes[2].text(index, 1.04, f"n={len(sample):,}", ha="center", fontsize=8)
    for ax in axes:
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(title, loc="left", pad=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E7E7E7", linewidth=.5)
    fig.tight_layout(w_pad=1.8)
    output = ROOT / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"week2_overview.{extension}", dpi=400, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    print(output / "week2_overview.png")


if __name__ == "__main__":
    main()
