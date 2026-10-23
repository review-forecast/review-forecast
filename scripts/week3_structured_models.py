# 3주차: 초기 별점 기준 모델과 정형 특성 확장 모델
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2,<4", "scikit-learn>=1.5,<2", "matplotlib>=3.8"]
# ///
# How to run: python scripts/week3_structured_models.py

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT: Final = Path(__file__).resolve().parents[1]
INPUT: Final = ROOT / "reports/week2_products.csv"
OUTPUT: Final = ROOT / "reports"
FIGURES: Final = OUTPUT / "figures"
TARGET: Final = "followup_low_ratio"
RANDOM_STATE: Final = 42
TEST_SIZE: Final = 0.20

BASELINE_FEATURES: Final = ["initial_mean_rating"]
STRUCTURED_FEATURES: Final = [
    "initial_mean_rating",
    "initial_rating_std",
    "initial_low_ratio",
    "initial_rating_1_ratio",
    "initial_rating_2_ratio",
    "initial_rating_3_ratio",
    "initial_rating_4_ratio",
    "initial_rating_5_ratio",
    "initial_mean_text_length",
    "initial_mean_title_length",
    "initial_text_present_ratio",
    "initial_verified_ratio",
]


def load_products() -> pd.DataFrame:
    """2주차에 구성한 상품별 자료와 필수 열을 검증합니다."""
    products = pd.read_csv(INPUT, dtype={"parent_asin": "string"})
    required = ["parent_asin", TARGET, *STRUCTURED_FEATURES]
    missing = sorted(set(required) - set(products.columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")
    if products.parent_asin.duplicated().any():
        raise ValueError("상품 식별자는 한 행에 하나씩만 존재해야 합니다.")
    if products[required].isna().any().any():
        raise ValueError("모델링 열에 결측치가 있습니다.")
    if not products[TARGET].between(0, 1).all():
        raise ValueError("타깃은 0과 1 사이여야 합니다.")
    return products


def split_products(products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """상품 단위 임시 홀드아웃을 고정된 난수로 나눕니다.

    정식 시간순 평가는 5주차 범위이므로, 3주차에서는 모델 구현과
    특성 확장의 차이를 재현 가능하게 확인하는 용도로만 사용합니다.
    """
    train, test = train_test_split(
        products, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    return train.sort_values("parent_asin"), test.sort_values("parent_asin")


def evaluate(name: str, y_true: pd.Series, raw_prediction: np.ndarray) -> tuple[dict, np.ndarray]:
    """비율 타깃의 유효 범위에 맞게 예측을 잘라낸 뒤 성능을 계산합니다."""
    clipped = np.clip(raw_prediction, 0.0, 1.0)
    metrics = {
        "model": name,
        "mae": mean_absolute_error(y_true, clipped),
        "rmse": mean_squared_error(y_true, clipped) ** 0.5,
        "r2": r2_score(y_true, clipped),
        "raw_below_zero": int((raw_prediction < 0).sum()),
        "raw_above_one": int((raw_prediction > 1).sum()),
    }
    return metrics, clipped


def fit_models(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """기준 선형회귀와 표준화 Ridge 회귀를 학습·평가합니다."""
    y_train, y_test = train[TARGET], test[TARGET]
    baseline = LinearRegression().fit(train[BASELINE_FEATURES], y_train)
    baseline_raw = baseline.predict(test[BASELINE_FEATURES])

    alphas = np.logspace(-4, 4, 81)
    structured = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, scoring="neg_mean_absolute_error", cv=5),
    )
    structured.fit(train[STRUCTURED_FEATURES], y_train)
    structured_raw = structured.predict(test[STRUCTURED_FEATURES])

    baseline_metrics, baseline_prediction = evaluate("baseline_linear", y_test, baseline_raw)
    structured_metrics, structured_prediction = evaluate("structured_ridge", y_test, structured_raw)
    metrics = pd.DataFrame([baseline_metrics, structured_metrics])

    predictions = test[["parent_asin", TARGET]].copy()
    predictions["baseline_prediction"] = baseline_prediction
    predictions["structured_prediction"] = structured_prediction
    predictions["baseline_absolute_error"] = (predictions[TARGET] - baseline_prediction).abs()
    predictions["structured_absolute_error"] = (predictions[TARGET] - structured_prediction).abs()
    predictions["split"] = "test"

    ridge = structured.named_steps["ridgecv"]
    coefficients = pd.DataFrame({
        "feature": STRUCTURED_FEATURES,
        "standardized_coefficient": ridge.coef_,
    })
    coefficients["absolute_coefficient"] = coefficients.standardized_coefficient.abs()
    coefficients = coefficients.sort_values("absolute_coefficient", ascending=False)
    return metrics, predictions, coefficients, float(ridge.alpha_)


def save_figure(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """모델별 오차와 관측값-예측값 관계를 저장합니다."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "axes.linewidth": .7, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    navy, light, accent = "#263E56", "#9CAAB7", "#A16B5D"
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))

    x = np.arange(len(metrics))
    width = .34
    axes[0].bar(x - width / 2, metrics.mae, width, label="MAE", color=navy)
    axes[0].bar(x + width / 2, metrics.rmse, width, label="RMSE", color=light)
    axes[0].set_xticks(x, ["Mean-rating\nbaseline", "Structured\nRidge"])
    axes[0].set(title="(a) Holdout error", ylabel="Error", ylim=(0, metrics.rmse.max() * 1.25))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, fontsize=8)

    panels = [
        ("baseline_prediction", "(b) Mean-rating baseline", navy),
        ("structured_prediction", "(c) Structured Ridge", accent),
    ]
    for ax, (column, title, color) in zip(axes[1:], panels):
        ax.scatter(predictions[TARGET], predictions[column], s=9, alpha=.28,
                   color=color, edgecolors="none", rasterized=True)
        ax.plot([0, 1], [0, 1], color="#777777", linewidth=.8, linestyle="--")
        ax.set(title=title, xlabel="Observed future low-rating share",
               ylabel="Predicted share", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    for ax in axes:
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(title, loc="left", pad=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E7E7E7", linewidth=.5)
    fig.tight_layout(w_pad=1.8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"week3_model_comparison.{extension}", dpi=400,
                    bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def main() -> None:
    start = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    products = load_products()
    train, test = split_products(products)

    assert set(train.parent_asin).isdisjoint(test.parent_asin)
    assert len(train) + len(test) == len(products)
    assert not any(feature.startswith("followup_") for feature in STRUCTURED_FEATURES)

    metrics, predictions, coefficients, alpha = fit_models(train, test)
    metrics.to_csv(OUTPUT / "week3_metrics.csv", index=False)
    predictions.to_csv(OUTPUT / "week3_predictions.csv", index=False)
    coefficients.to_csv(OUTPUT / "week3_structured_coefficients.csv", index=False)
    pd.concat([
        train[["parent_asin"]].assign(split="train"),
        test[["parent_asin"]].assign(split="test"),
    ]).sort_values("parent_asin").to_csv(OUTPUT / "week3_product_split.csv", index=False)
    save_figure(metrics, predictions)

    with (OUTPUT / "week3_output.txt").open("w", encoding="utf-8") as output, redirect_stdout(output):
        print("# 3주차 별점·정형 특성 회귀 모델")
        print("입력 파일:", INPUT)
        print("분석 상품 수:", len(products))
        print(f"임시 상품 단위 분할: 학습 {len(train):,}개 / 평가 {len(test):,}개")
        print(f"분할 기준: test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
        print("정식 시간순 평가는 5주차에 수행할 예정입니다.")
        print("\n기준 모델 특성:", ", ".join(BASELINE_FEATURES))
        print("확장 모델 특성:", ", ".join(STRUCTURED_FEATURES))
        print(f"Ridge 5-fold CV 선택 alpha: {alpha:g}")
        print("\n평가 결과(예측값을 0~1로 제한):")
        print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        improvement = 1 - metrics.loc[metrics.model.eq("structured_ridge"), "mae"].iloc[0] / metrics.loc[
            metrics.model.eq("baseline_linear"), "mae"
        ].iloc[0]
        print(f"\n기준 모델 대비 확장 모델 MAE 변화율: {improvement:.2%}")
        print("\n표준화 계수 절대값 상위 특성:")
        print(coefficients.head(8).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        print("\n상품 분할·타깃 범위·후속 정보 미사용 검사: PASS")
        print(f"실행 시간: {perf_counter() - start:.1f}초")
    print(OUTPUT / "week3_output.txt")


if __name__ == "__main__":
    main()
