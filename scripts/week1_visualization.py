# 1주차 결과 시각화
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/All_Beauty.jsonl.gz"


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "axes.linewidth": .7, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.width": .7, "ytick.major.width": .7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    df = pd.read_json(DATA_PATH, lines=True, convert_dates=False)
    ratings = df.rating.value_counts().reindex(range(1, 6), fill_value=0)
    products = df.groupby("parent_asin").size()
    bins = pd.cut(products, [0, 1, 4, 9, 29, float("inf")],
                  labels=["1", "2–4", "5–9", "10–29", "≥30"])
    counts = bins.value_counts(sort=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    navy, light, accent = "#263E56", "#9CAAB7", "#A16B5D"
    panels = [
        (axes[0], ratings / ratings.sum() * 100, [str(i) for i in range(1, 6)],
         [accent, accent, light, light, navy], "(a) Review ratings", "Rating", "Reviews (%)", 70),
        (axes[1], counts / counts.sum() * 100, list(counts.index.astype(str)),
         [light] * 4 + [navy], "(b) Reviews per product", "Number of reviews", "Products (%)", 50),
    ]
    for ax, values, labels, colors, title, xlabel, ylabel, ymax in panels:
        bars = ax.bar(labels, values, width=.64, color=colors, edgecolor="white", linewidth=.35)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+.9, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, loc="left", pad=13)
        ax.set_xlabel(xlabel, labelpad=6)
        ax.set_ylabel(ylabel, labelpad=5)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.tick_params(axis="y", direction="out", length=3)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E7E7E7", linewidth=.5)
    fig.subplots_adjust(left=.085, right=.985, bottom=.20, top=.86, wspace=.36)
    output = ROOT / "reports/figures"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"week1_overview.{extension}", dpi=400, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    (output / "week1_caption.md").write_text(
        "Figure 1. Distribution of (a) review ratings and (b) review counts per product "
        "in All_Beauty (701,528 reviews; 112,565 products). Bars show percentages. "
        "Ratings of 1–2 account for 20.69% of reviews. Products with at least 30 reviews "
        "number 3,692 (3.28%). Counts are computed before data cleaning.\n")
    print(output / "week1_overview.png")


if __name__ == "__main__":
    main()
