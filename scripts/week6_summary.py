# 6주차: 저장된 예측 결과 검증과 최종 성능 종합
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2,<4", "matplotlib>=3.8"]
# ///
# How to run: python scripts/week6_summary.py
# 또는 uv run scripts/week6_summary.py
from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
import pandas as pd  # 기존 작업자의 Pandas 기반 분석 방식 유지

ROOT: Final = Path(__file__).resolve().parents[1]
OUTPUT: Final = ROOT / "reports"
TARGET: Final = "followup_low_ratio"
TOLERANCE: Final = 1e-7  # CSV의 float32 예측값 저장 정밀도 고려


def check(condition: bool, description: str) -> None:
    """검사 실패 시 산출물 생성을 중단합니다. python -O에서도 동작합니다."""
    if not condition:
        raise AssertionError(description)
    print("PASS:", description)


def verify_splits(products: pd.DataFrame) -> None:
    """원본 상품 특성과 저장된 랜덤 및 시간순 분할을 대조합니다."""
    random = pd.read_csv(OUTPUT / "week3_product_split.csv").set_index("parent_asin")
    temporal = pd.read_csv(OUTPUT / "week5_temporal_split.csv").set_index("parent_asin")
    for split in [random, temporal]:
        check(split.index.is_unique and set(split.index) == set(products.index), "분할별 상품 중복 없음 및 전체 상품 일치")
    check(set(random.split) == {"train", "test"}, "랜덤 분할 집합 이름")
    check(set(temporal.split) == {"train", "test", "embargo"}, "시간순 분할 집합 이름")
    initial = pd.to_datetime(products.initial_end_timestamp, unit="ms", utc=True)
    future = pd.to_datetime(products.followup_end_timestamp, unit="ms", utc=True)
    check(initial.le(future).all(), "초기 완료 시각이 후속 완료 시각보다 늦지 않음")
    for column, source in [("initial_end_timestamp_utc", initial), ("followup_end_timestamp_utc", future)]:
        check(pd.to_datetime(temporal[column], utc=True).eq(source.reindex(temporal.index)).all(), "저장된 분할 시각과 2주차 상품 데이터 일치")
    cutoff = pd.Timestamp("2020-07-01", tz="UTC")
    expected = pd.Series("embargo", index=products.index)
    expected.loc[future.lt(cutoff)] = "train"
    expected.loc[initial.ge(cutoff)] = "test"
    check(temporal.split.eq(expected.reindex(temporal.index)).all(), "절단일 기준 학습, 평가, 임벌고 분류 일치")
    train_ids = temporal.index[temporal.split.eq("train")]
    test_ids = temporal.index[temporal.split.eq("test")]
    check(future.loc[train_ids].max() < initial.loc[test_ids].min(), "학습 목표값이 평가 예측 시점보다 앞섬")
    folds = pd.read_csv(OUTPUT / "week5_temporal_cv_folds.csv")
    for row in folds.itertuples():
        start, end = pd.Timestamp(row.prediction_window_start), pd.Timestamp(row.prediction_window_end)
        fit = train_ids[future.loc[train_ids].lt(start)]
        valid = train_ids[initial.loc[train_ids].ge(start) & initial.loc[train_ids].lt(end)]
        check(len(fit) == row.fit_products and len(valid) == row.validation_products, f"시간순 CV {row.fold} 상품 수")
        check(future.loc[fit].max() < initial.loc[valid].min(), f"시간순 CV {row.fold} 학습 목표값 선행")
    print("랜덤 분할:", random.split.value_counts().to_dict())
    print("시간순 분할:", temporal.split.value_counts().to_dict())


def verify_predictions(week: int, products: pd.DataFrame) -> pd.DataFrame:
    """저장된 예측값에서 지표와 절대 오차를 독립적으로 재계산합니다."""
    predictions = pd.read_csv(OUTPUT / f"week{week}_predictions.csv").set_index("parent_asin")
    metrics = pd.read_csv(OUTPUT / f"week{week}_metrics.csv")
    if "setting" in metrics:
        metrics = metrics.loc[metrics.setting.eq("main")]
    split_file = "week5_temporal_split.csv" if week == 5 else "week3_product_split.csv"
    split = pd.read_csv(OUTPUT / split_file)
    ids = set(split.loc[split.split.eq("test"), "parent_asin"])
    check(predictions.index.is_unique and set(predictions.index) == ids, f"{week}주차 평가 상품 일치 및 중복 없음")
    y = predictions[TARGET].to_numpy()
    check(np.isfinite(y).all() and np.allclose(y, products.loc[predictions.index, TARGET], rtol=0, atol=1e-12), f"{week}주차 목표값 원본 일치")
    rows = []
    for row in metrics.itertuples():
        model = row.model
        prefix = model
        if week == 3:
            prefix = {"baseline_linear": "baseline", "structured_ridge": "structured"}[model]
        prediction = predictions[f"{prefix}_prediction"].to_numpy()
        check(np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all(), f"{week}주차 {model} 예측값 유효 범위")
        error = np.abs(y - prediction)
        check(np.allclose(error, predictions[f"{prefix}_absolute_error"], rtol=0, atol=TOLERANCE), f"{week}주차 {model} 상품별 절대 오차")
        values = [error.mean(), np.sqrt(np.mean((y - prediction) ** 2)), 1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)]
        for metric, value in zip(["mae", "rmse", "r2"], values):
            recorded = getattr(row, metric)
            check(np.isclose(value, recorded, rtol=0, atol=TOLERANCE), f"{week}주차 {model} {metric} 재계산")
            rows.append({"week": week, "model": model, "metric": metric, "recorded": recorded,
                         "recalculated": value, "absolute_difference": abs(value - recorded)})
    if week >= 4:
        bootstrap = pd.read_csv(OUTPUT / f"week{week}_bootstrap.csv")
        sampled = np.random.default_rng(42).integers(0, len(y), size=(1000, len(y)))
        for row in bootstrap.itertuples():
            ref = np.abs(y - predictions[f"{row.reference}_prediction"].to_numpy())
            error = np.abs(y - predictions[f"{row.model}_prediction"].to_numpy())
            difference = ref[sampled].mean(axis=1) - error[sampled].mean(axis=1)
            values = [ref.mean() - error.mean(), *np.quantile(difference, [.025, .975])]
            check(np.allclose(values, [row.mae_reduction, row.ci_low, row.ci_high], rtol=0, atol=TOLERANCE), f"{week}주차 {row.reference} 대비 {row.model} 부트스트랩 재계산")
    return pd.DataFrame(rows)


def main() -> None:
    """학습을 수행하지 않고 검증 통과 후 종합 CSV와 최종 그림을 생성합니다."""
    start = perf_counter()
    with (OUTPUT / "week6_output.txt").open("w", encoding="utf-8") as output, redirect_stdout(output):
        print("# 6주차 저장 결과 검증 및 종합")
        print("검증 범위: 저장된 예측값, 지표, 상품 분할, 시간 조건, 부트스트랩. 모델 재학습은 수행하지 않음.")
        print(f"지표 허용 절대 오차: {TOLERANCE:g}")
        products = pd.read_csv(OUTPUT / "week2_products.csv").set_index("parent_asin")
        check(products.index.is_unique, "상품 분석표의 식별자 고유성")
        verify_splits(products)
        verification = pd.concat([verify_predictions(week, products) for week in [3, 4, 5]], ignore_index=True)
        summary = []
        for week, evaluation in [(4, "random"), (5, "temporal")]:
            metrics = pd.read_csv(OUTPUT / f"week{week}_metrics.csv")
            if "setting" in metrics:
                metrics = metrics.loc[metrics.setting.eq("main")]
            summary.append(metrics[["model", "mae", "rmse", "r2"]].assign(evaluation=evaluation))
        comparison = pd.concat(summary, ignore_index=True)
        previous = pd.read_csv(OUTPUT / "week5_random_vs_temporal.csv")
        keys = ["evaluation", "model"]
        check(np.allclose(comparison.set_index(keys).sort_index(), previous.set_index(keys).sort_index(), rtol=0, atol=TOLERANCE), "5주차 비교표와 최종 종합표 일치")
        verification.to_csv(OUTPUT / "week6_verification.csv", index=False)
        comparison.to_csv(OUTPUT / "week6_model_comparison.csv", index=False)
        print("\n최종 성능 비교:\n", comparison.to_string(index=False))
        print("\n검증한 성능 지표 수:", len(verification))
        print("최대 지표 차이:", verification.absolute_difference.max())
        print(f"실행 시간: {perf_counter() - start:.2f}초")
    from week6_visualization import save_figure

    save_figure()
    print(OUTPUT / "week6_output.txt")


if __name__ == "__main__":
    main()
