# 5주차: 상품 단위 시간순 평가·모델 비교·표현·오차 사례 분석
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2,<4", "scikit-learn>=1.5,<2", "matplotlib>=3.8"]
# ///
# How to run: python scripts/week5_temporal_evaluation.py

from __future__ import annotations

import re
import sys
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Final

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import week3_structured_models as week3  # noqa: E402
import week4_text_models as week4  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT: Final = week3.ROOT
OUTPUT: Final = week3.OUTPUT
FIGURES: Final = week3.FIGURES
TARGET: Final = week3.TARGET
RANDOM_STATE: Final = week3.RANDOM_STATE
STRUCTURED_FEATURES: Final = week3.STRUCTURED_FEATURES
WEEK4_METRICS: Final = ROOT / "reports/week4_metrics.csv"

# 임시 랜덤 홀드아웃과 평가 규모를 유사하게 맞추되,
# 해석하기 쉬운 분기 시작일로 2020-07-01을 고정합니다.
TEMPORAL_CUTOFF: Final = pd.Timestamp("2020-07-01", tz="UTC")
CV_WINDOWS: Final = [
    (pd.Timestamp("2017-01-01", tz="UTC"), pd.Timestamp("2018-01-01", tz="UTC")),
    (pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2019-01-01", tz="UTC")),
    (pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC")),
]
STRUCTURED_ALPHAS: Final = np.logspace(-4, 4, 81)
BOOTSTRAP_ROUNDS: Final = 1000
CASE_COUNT_PER_DIRECTION: Final = 5


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """상품 정형 특성과 초기 10개 리뷰의 제목·본문 문서를 결합합니다."""
    products = week3.load_products().set_index("parent_asin")
    initial = week4.load_initial_reviews()
    documents = week4.build_documents(initial)
    assert documents.index.equals(products.index.sort_values())
    data = products.join(documents).reset_index()
    for column in ["initial_end_timestamp", "followup_end_timestamp"]:
        data[f"{column}_utc"] = pd.to_datetime(data[column], unit="ms", utc=True)
    return data, initial


def make_temporal_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """타깃이 절단일 전에 완성된 상품만 학습하고, 절단일 이후 예측 상품을 평가합니다.

    절단일 전에 초기 10개는 관측됐지만 후속 20개가 아직 완성되지 않은
    상품은 embargo로 분류해 학습과 평가 모두에서 제외합니다.
    """
    train_mask = data.followup_end_timestamp_utc.lt(TEMPORAL_CUTOFF)
    test_mask = data.initial_end_timestamp_utc.ge(TEMPORAL_CUTOFF)
    assert not (train_mask & test_mask).any()
    embargo_mask = ~(train_mask | test_mask)
    train = data.loc[train_mask].sort_values(["followup_end_timestamp_utc", "parent_asin"]).copy()
    test = data.loc[test_mask].sort_values(["initial_end_timestamp_utc", "parent_asin"]).copy()
    embargo = data.loc[embargo_mask].sort_values(["initial_end_timestamp_utc", "parent_asin"]).copy()
    assert train.followup_end_timestamp_utc.max() < test.initial_end_timestamp_utc.min()
    return train, test, embargo


def temporal_cv_splits(train: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """각 fold에서 학습 타깃은 검증 예측 시점보다 앞서도록 표본 위치를 만듭니다."""
    splits, rows = [], []
    for fold, (start, end) in enumerate(CV_WINDOWS, 1):
        fit_mask = train.followup_end_timestamp_utc.lt(start)
        valid_mask = train.initial_end_timestamp_utc.ge(start) & train.initial_end_timestamp_utc.lt(end)
        fit_index = np.flatnonzero(fit_mask.to_numpy())
        valid_index = np.flatnonzero(valid_mask.to_numpy())
        if not len(fit_index) or not len(valid_index):
            raise ValueError(f"시간순 CV fold {fold}에 표본이 없습니다.")
        assert train.iloc[fit_index].followup_end_timestamp_utc.max() < train.iloc[valid_index].initial_end_timestamp_utc.min()
        splits.append((fit_index, valid_index))
        rows.append({
            "fold": fold,
            "prediction_window_start": start.isoformat(),
            "prediction_window_end": end.isoformat(),
            "fit_products": len(fit_index),
            "validation_products": len(valid_index),
            "latest_fit_target": train.iloc[fit_index].followup_end_timestamp_utc.max().isoformat(),
            "earliest_validation_prediction": train.iloc[valid_index].initial_end_timestamp_utc.min().isoformat(),
        })
    return splits, pd.DataFrame(rows)


def tune_structured(train: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[object, float, pd.DataFrame]:
    """시간순 CV로 정형 Ridge의 alpha를 선택하고 전체 시간순 학습 세트에 재학습합니다."""
    rows = []
    for fold, (fit_index, valid_index) in enumerate(folds, 1):
        fit, valid = train.iloc[fit_index], train.iloc[valid_index]
        scaler = StandardScaler().fit(fit[STRUCTURED_FEATURES])
        x_fit = scaler.transform(fit[STRUCTURED_FEATURES])
        x_valid = scaler.transform(valid[STRUCTURED_FEATURES])
        for alpha in STRUCTURED_ALPHAS:
            model = Ridge(alpha=alpha).fit(x_fit, fit[TARGET])
            rows.append({
                "model": "structured_ridge", "fold": fold, "alpha": alpha,
                "mae": mean_absolute_error(valid[TARGET], model.predict(x_valid)),
            })
    scores = pd.DataFrame(rows)
    summary = scores.groupby("alpha").mae.agg(cv_mae="mean", cv_mae_std="std").reset_index()
    best_alpha = float(summary.sort_values("cv_mae", kind="stable").iloc[0].alpha)
    fitted = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)).fit(
        train[STRUCTURED_FEATURES], train[TARGET])
    return fitted, best_alpha, scores


def fit_text_models_temporal(train: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """4주차 A~D 모델을 랜덤 K-fold 대신 시간순 fold로 탐색·학습합니다."""
    fitted, summaries, all_tables = {}, [], []
    for name, parts in week4.TEXT_MODELS.items():
        start = perf_counter()
        jobs = [
            delayed(week4.score_fold)(parts, ngram, min_df, fold, train, fit_index, valid_index)
            for ngram, min_df in product(week4.NGRAM_RANGES, week4.MIN_DFS)
            for fold, (fit_index, valid_index) in enumerate(folds, 1)
        ]
        scores = pd.DataFrame([row for group in Parallel(n_jobs=-1)(jobs) for row in group])
        keys = ["ngram_range", "min_df", "text_weight", "alpha"]
        table = scores.groupby(keys, sort=False).mae.agg(cv_mae="mean", cv_mae_std="std").reset_index()
        table = table.sort_values("cv_mae", kind="stable").assign(model=name)
        best = table.iloc[0]
        ngram = tuple(int(value) for value in best.ngram_range.strip("()").split(","))
        model = week4.make_model(parts, ngram, int(best.min_df), float(best.text_weight), float(best.alpha))
        fitted[name] = model.fit(train, train[TARGET])
        text_weight = f"{best.text_weight:g}" if "structured" in parts else "-"
        summaries.append({
            "model": name,
            "cv_mae": best.cv_mae,
            "best_params": f"alpha={best.alpha:.4g}, ngram_range={best.ngram_range}, "
                           f"min_df={best.min_df}, text_weight={text_weight}",
            "fit_seconds": perf_counter() - start,
        })
        all_tables.append(table)
    return fitted, pd.DataFrame(summaries), pd.concat(all_tables, ignore_index=True)


def evaluate_models(train: pd.DataFrame, test: pd.DataFrame,
                    folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """기준·정형·텍스트 A~D를 같은 시간순 평가 상품에서 비교합니다."""
    rows = []
    predictions = test[["parent_asin", TARGET, "initial_end_timestamp_utc", "followup_end_timestamp_utc"]].copy()

    baseline = LinearRegression().fit(train[["initial_mean_rating"]], train[TARGET])
    raw = baseline.predict(test[["initial_mean_rating"]])
    metrics, clipped = week3.evaluate("baseline_linear", test[TARGET], raw)
    rows.append({**metrics, "best_params": "ordinary least squares"})
    predictions["baseline_linear_prediction"] = clipped
    predictions["baseline_linear_absolute_error"] = np.abs(test[TARGET].to_numpy() - clipped)

    structured, alpha, structured_cv = tune_structured(train, folds)
    raw = structured.predict(test[STRUCTURED_FEATURES])
    metrics, clipped = week3.evaluate("structured_ridge", test[TARGET], raw)
    rows.append({**metrics, "best_params": f"alpha={alpha:g}"})
    predictions["structured_ridge_prediction"] = clipped
    predictions["structured_ridge_absolute_error"] = np.abs(test[TARGET].to_numpy() - clipped)

    text_models, search_summary, text_cv = fit_text_models_temporal(train, folds)
    for name, model in text_models.items():
        raw = model.predict(test)
        metrics, clipped = week3.evaluate(name, test[TARGET], raw)
        params = search_summary.set_index("model").loc[name, "best_params"]
        rows.append({**metrics, "best_params": params})
        predictions[f"{name}_prediction"] = clipped
        predictions[f"{name}_absolute_error"] = np.abs(test[TARGET].to_numpy() - clipped)

    cv_results = pd.concat([
        structured_cv.assign(ngram_range="-", min_df=np.nan, text_weight=np.nan),
        text_cv,
    ], ignore_index=True, sort=False)
    metrics = pd.DataFrame(rows)
    assert predictions.filter(like="_prediction").apply(lambda values: values.between(0, 1).all()).all()
    return metrics, predictions, text_models, cv_results


def paired_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    """같은 상품을 복원추출해 기준 모델 대비 MAE 감소량의 95% 구간을 계산합니다."""
    rng = np.random.default_rng(RANDOM_STATE)
    sampled = rng.integers(0, len(predictions), size=(BOOTSTRAP_ROUNDS, len(predictions)))
    rows = []
    models = ["structured_ridge", *week4.TEXT_MODELS]
    for reference in ["baseline_linear", "structured_ridge"]:
        reference_error = predictions[f"{reference}_absolute_error"].to_numpy()
        for model in models:
            if model == reference:
                continue
            model_error = predictions[f"{model}_absolute_error"].to_numpy()
            differences = reference_error[sampled].mean(axis=1) - model_error[sampled].mean(axis=1)
            rows.append({
                "reference": reference,
                "model": model,
                "mae_reduction": reference_error.mean() - model_error.mean(),
                "ci_low": np.quantile(differences, .025),
                "ci_high": np.quantile(differences, .975),
                "share_improved": np.mean(differences > 0),
            })
    return pd.DataFrame(rows)


def add_term_prevalence(terms: pd.DataFrame, models: dict,
                        train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """상위 표현이 시간순 학습·평가 상품에 나타난 비율을 계산합니다."""
    result = terms.copy()
    result["train_document_share"] = np.nan
    result["test_document_share"] = np.nan
    for (model_name, source), index in result.groupby(["model", "source"]).groups.items():
        vectorizer = models[model_name].named_steps["features"].named_transformers_[source]
        analyzer = vectorizer.build_analyzer()
        column = week4.TEXT_COLUMNS[source]
        train_tokens = [set(analyzer(document)) for document in train[column]]
        test_tokens = [set(analyzer(document)) for document in test[column]]
        for row_index in index:
            term = result.at[row_index, "term"]
            result.at[row_index, "train_document_share"] = np.mean([term in values for values in train_tokens])
            result.at[row_index, "test_document_share"] = np.mean([term in values for values in test_tokens])
    return result


def analyze_terms(models: dict, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """텍스트 단독 A와 결합 D의 상위 표현을 뽑고 4주차 결과와 안정성을 비교합니다."""
    terms = pd.concat([
        week4.top_terms(models["A_text_only"], "A_text_only"),
        week4.top_terms(models["D_structured_title_body"], "D_structured_title_body"),
    ], ignore_index=True)
    terms = add_term_prevalence(terms, models, train, test)
    random_terms = pd.read_csv(OUTPUT / "week4_top_terms.csv")
    stability = terms.merge(
        random_terms[["model", "direction", "rank", "source", "term"]].rename(columns={"rank": "random_split_rank"}),
        on=["model", "direction", "source", "term"], how="left",
    )
    stability["appeared_in_random_top20"] = stability.random_split_rank.notna()
    return terms, stability


def star_title_sensitivity(data: pd.DataFrame, initial: pd.DataFrame,
                           train: pd.DataFrame, test: pd.DataFrame,
                           model: object) -> pd.DataFrame:
    """N Stars 자동 제목의 시기별 불균형이 결과를 왜곡하는지 점검합니다."""
    without_star_documents = week4.build_documents(initial, drop_star_titles=True)
    no_star = data.drop(columns=["title_doc", "body_doc"]).set_index("parent_asin").join(
        without_star_documents).reset_index()
    no_star_train = no_star.set_index("parent_asin").loc[train.parent_asin].reset_index()
    no_star_test = no_star.set_index("parent_asin").loc[test.parent_asin].reset_index()
    sensitivity_model = clone(model).fit(no_star_train, no_star_train[TARGET])
    metrics, _ = week3.evaluate(
        "D_without_star_titles", no_star_test[TARGET], sensitivity_model.predict(no_star_test))
    train_review_mask = initial.parent_asin.isin(train.parent_asin)
    test_review_mask = initial.parent_asin.isin(test.parent_asin)
    metrics.update({
        "train_star_title_review_share": initial.loc[train_review_mask, "title"].str.match(week4.STAR_TITLE).mean(),
        "test_star_title_review_share": initial.loc[test_review_mask, "title"].str.match(week4.STAR_TITLE).mean(),
    })
    return pd.DataFrame([metrics])


def build_error_cases(predictions: pd.DataFrame, data: pd.DataFrame,
                      initial: pd.DataFrame, terms: pd.DataFrame) -> pd.DataFrame:
    """결합 D 모델의 과소·과대 예측 상품과 초기 리뷰 근거를 저장합니다."""
    model = "D_structured_title_body"
    table = predictions.copy()
    table["residual"] = table[f"{model}_prediction"] - table[TARGET]
    under = table.nsmallest(CASE_COUNT_PER_DIRECTION, "residual").assign(case_type="underprediction")
    over = table.nlargest(CASE_COUNT_PER_DIRECTION, "residual").assign(case_type="overprediction")
    cases = pd.concat([under, over], ignore_index=True)

    lookup = data.set_index("parent_asin")
    for column in ["initial_mean_rating", "initial_rating_std", "initial_low_ratio", "title_doc", "body_doc"]:
        cases[column] = cases.parent_asin.map(lookup[column])

    # CSV에서 사례를 바로 읽을 수 있도록 초기 10개 제목·본문을 짧게 연결합니다.
    review_rows = pd.read_csv(
        week4.WINDOWS,
        usecols=["parent_asin", "window", "review_rank", "rating", "title", "text"],
        dtype={"parent_asin": "string", "title": "string", "text": "string"},
    )
    review_rows = review_rows.loc[review_rows.window.eq("initial")].fillna({"title": "", "text": ""})
    review_rows["excerpt"] = review_rows.apply(
        lambda row: f"[{int(row.review_rank)}] {int(row.rating)}★ {row.title}: "
                    f"{re.sub(r'\s+', ' ', row.text).strip()[:180]}", axis=1)
    excerpts = review_rows.groupby("parent_asin").excerpt.agg(" | ".join)
    cases["initial_review_excerpts"] = cases.parent_asin.map(excerpts)

    d_terms = terms.loc[terms.model.eq(model)].sort_values("coefficient", key=lambda values: values.abs(), ascending=False)
    def matched(row: pd.Series, direction: str) -> str:
        matches = []
        for term in d_terms.loc[d_terms.direction.eq(direction)].itertuples():
            document = row.title_doc if term.source == "title" else row.body_doc
            if re.search(rf"(?<!\w){re.escape(term.term)}(?!\w)", document):
                matches.append(f"{term.source}:{term.term}({term.coefficient:+.3f})")
            if len(matches) == 5:
                break
        return "; ".join(matches)

    cases["matched_higher_risk_terms"] = cases.apply(
        matched, axis=1, direction="higher_future_low_ratio")
    cases["matched_lower_risk_terms"] = cases.apply(
        matched, axis=1, direction="lower_future_low_ratio")
    cases["absolute_error"] = cases.residual.abs()
    keep = [
        "case_type", "parent_asin", TARGET, f"{model}_prediction", "absolute_error",
        "baseline_linear_prediction", "structured_ridge_prediction", "initial_mean_rating",
        "initial_rating_std", "initial_low_ratio", "matched_higher_risk_terms",
        "matched_lower_risk_terms", "initial_review_excerpts",
    ]
    return cases[keep].sort_values(["case_type", "absolute_error"], ascending=[True, False])


def metrics_by_period(predictions: pd.DataFrame) -> pd.DataFrame:
    """최근 상품에서도 성능이 유지되는지 예측 연도별로 정리합니다."""
    data = predictions.copy()
    data["prediction_year"] = pd.to_datetime(data.initial_end_timestamp_utc, utc=True).dt.year
    rows = []
    for year, group in data.groupby("prediction_year"):
        for model in ["baseline_linear", "structured_ridge", *week4.TEXT_MODELS]:
            prediction = group[f"{model}_prediction"]
            rows.append({
                "prediction_year": year, "products": len(group), "model": model,
                "mae": mean_absolute_error(group[TARGET], prediction),
                "rmse": mean_squared_error(group[TARGET], prediction) ** .5,
                "r2": r2_score(group[TARGET], prediction) if len(group) >= 2 else np.nan,
            })
    return pd.DataFrame(rows)


def save_figure(split: pd.DataFrame, comparison: pd.DataFrame,
                predictions: pd.DataFrame, error_cases: pd.DataFrame) -> None:
    """시간 분할·랜덤 대비 오차·결합 모델 오차 사례를 한 장으로 저장합니다."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "axes.linewidth": .7, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    navy, light, accent = "#263E56", "#9CAAB7", "#A16B5D"
    # 주별 활동 보고서의 가로형 사진 칸에서 글자가 지나치게 작아지지 않도록
    # 3개 패널을 약 2.8:1 비율로 구성합니다.
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), gridspec_kw={"width_ratios": [1.1, 1.25, 1]})

    timeline = split.copy()
    timeline["year"] = pd.to_datetime(timeline.initial_end_timestamp_utc, utc=True).dt.year.clip(lower=2013)
    counts = timeline.groupby(["year", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["train", "embargo", "test"], fill_value=0)
    bottom = np.zeros(len(counts))
    for label, color in [("train", navy), ("embargo", light), ("test", accent)]:
        axes[0].bar(counts.index, counts[label], bottom=bottom, label=label, color=color)
        bottom += counts[label].to_numpy()
    axes[0].axvline(2020.5, color="#777777", linestyle="--", linewidth=.8)
    axes[0].set(title="(a) Product split by prediction year", xlabel="Initial-window end year",
                ylabel="Products", xlim=(2012.5, 2023.5))
    axes[0].set_xticks([2013, 2015, 2017, 2019, 2021, 2023],
                      ["≤2013", "2015", "2017", "2019", "2021", "2023"])
    axes[0].legend(frameon=False, fontsize=8)

    labels = {
        "baseline_linear": "Mean-rating\nbaseline",
        "structured_ridge": "Structured\nRidge",
        "D_structured_title_body": "D: Struct. +\ntitle + body",
    }
    show = comparison.loc[comparison.model.isin(labels)].copy()
    show["order"] = show.model.map({name: index for index, name in enumerate(labels)})
    show = show.sort_values(["order", "evaluation"])
    x = np.arange(len(labels))
    width = .34
    random = show.loc[show.evaluation.eq("random")].set_index("model").reindex(labels).mae
    temporal = show.loc[show.evaluation.eq("temporal")].set_index("model").reindex(labels).mae
    axes[1].bar(x - width / 2, random, width, label="Random holdout", color=light)
    axes[1].bar(x + width / 2, temporal, width, label="Temporal holdout", color=navy)
    axes[1].set_xticks(x, labels.values(), fontsize=8)
    axes[1].set(title="(b) Random vs. temporal MAE", ylabel="MAE",
                ylim=(0, max(random.max(), temporal.max()) * 1.25))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, fontsize=8)

    model = "D_structured_title_body"
    axes[2].scatter(predictions[TARGET], predictions[f"{model}_prediction"], s=10,
                    alpha=.26, color=accent, edgecolors="none", rasterized=True)
    axes[2].plot([0, 1], [0, 1], color="#777777", linewidth=.8, linestyle="--")
    highlighted = predictions.loc[predictions.parent_asin.isin(error_cases.parent_asin)]
    axes[2].scatter(highlighted[TARGET], highlighted[f"{model}_prediction"], s=28,
                    facecolors="none", edgecolors=navy, linewidths=.8)
    axes[2].set(title="(c) Temporal errors of model D", xlabel="Observed future low-rating share",
                ylabel="Predicted share", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
    axes[2].xaxis.set_major_formatter(PercentFormatter(1))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))

    for axis in axes:
        title = axis.get_title()
        axis.set_title("")
        axis.set_title(title, loc="left", pad=13)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E7E7E7", linewidth=.5)
    fig.tight_layout(w_pad=1.8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "week5_temporal_evaluation.png", dpi=400,
                bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def main() -> None:
    start = perf_counter()
    data, initial = prepare_data()
    train, test, embargo = make_temporal_split(data)
    folds, fold_summary = temporal_cv_splits(train)

    metrics, predictions, text_models, cv_results = evaluate_models(train, test, folds)
    bootstrap = paired_bootstrap(predictions)
    terms, stability = analyze_terms(text_models, train, test)
    sensitivity = star_title_sensitivity(
        data, initial, train, test, text_models["D_structured_title_body"])
    error_cases = build_error_cases(predictions, data, initial, terms)
    period_metrics = metrics_by_period(predictions)

    random = pd.read_csv(WEEK4_METRICS).loc[lambda frame: frame.setting.eq("main")]
    random = random[["model", "mae", "rmse", "r2"]].assign(evaluation="random")
    temporal = metrics[["model", "mae", "rmse", "r2"]].assign(evaluation="temporal")
    comparison = pd.concat([random, temporal], ignore_index=True)

    split = data[["parent_asin", "initial_end_timestamp_utc", "followup_end_timestamp_utc"]].copy()
    split["split"] = "embargo"
    split.loc[split.parent_asin.isin(train.parent_asin), "split"] = "train"
    split.loc[split.parent_asin.isin(test.parent_asin), "split"] = "test"
    assert split.split.value_counts().to_dict() == {
        "train": len(train), "test": len(test), "embargo": len(embargo)}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    split.to_csv(OUTPUT / "week5_temporal_split.csv", index=False)
    fold_summary.to_csv(OUTPUT / "week5_temporal_cv_folds.csv", index=False)
    cv_results.to_csv(OUTPUT / "week5_cv_results.csv", index=False)
    metrics.to_csv(OUTPUT / "week5_metrics.csv", index=False)
    predictions.to_csv(OUTPUT / "week5_predictions.csv", index=False)
    bootstrap.to_csv(OUTPUT / "week5_bootstrap.csv", index=False)
    comparison.to_csv(OUTPUT / "week5_random_vs_temporal.csv", index=False)
    terms.to_csv(OUTPUT / "week5_top_terms.csv", index=False)
    stability.to_csv(OUTPUT / "week5_term_stability.csv", index=False)
    sensitivity.to_csv(OUTPUT / "week5_star_title_sensitivity.csv", index=False)
    error_cases.to_csv(OUTPUT / "week5_error_cases.csv", index=False)
    period_metrics.to_csv(OUTPUT / "week5_metrics_by_year.csv", index=False)
    save_figure(split, comparison, predictions, error_cases)

    output_path = OUTPUT / "week5_output.txt"
    with output_path.open("w", encoding="utf-8") as output, redirect_stdout(output):
        fmt = lambda value: f"{value:.6f}"  # noqa: E731
        print("# 5주차 상품 단위 시간순 평가·모델 비교·표현·오차 사례")
        print("시간 절단일:", TEMPORAL_CUTOFF.isoformat())
        print(f"학습 {len(train):,}개: followup_end < 절단일")
        print(f"평가 {len(test):,}개: initial_end >= 절단일")
        print(f"임벌고 {len(embargo):,}개: 절단일 전 초기 관측은 존재하나 타깃이 아직 완성되지 않음")
        print("가장 늦은 학습 타깃 시각:", train.followup_end_timestamp_utc.max().isoformat())
        print("가장 빠른 평가 예측 시각:", test.initial_end_timestamp_utc.min().isoformat())
        print("상품 중복·시간 누수 방지 검사: PASS")

        print("\n[시간순 교차 검증 fold]")
        print(fold_summary.to_string(index=False))

        print("\n[시간순 평가 결과] 예측값을 0~1로 제한")
        print(metrics.to_string(index=False, float_format=fmt))

        print("\n[랜덤 홀드아웃과 시간순 홀드아웃 비교]")
        print(comparison.pivot(index="model", columns="evaluation", values="mae")
              .assign(change=lambda frame: frame.temporal - frame.random)
              .to_string(float_format=fmt))

        print(f"\n[대응 부트스트랩 {BOOTSTRAP_ROUNDS}회] MAE 감소량(양수=개선)")
        print(bootstrap.to_string(index=False, float_format=fmt))

        print("\n[텍스트 단독 모델 A: 후속 저평점 비율 증가 방향 표현]")
        print(terms.loc[(terms.model.eq("A_text_only")) & terms.direction.str.startswith("higher")]
              .head(15)[["source", "term", "coefficient", "train_document_share", "test_document_share"]]
              .to_string(index=False, float_format=fmt))
        print("\n[텍스트 단독 모델 A: 후속 저평점 비율 감소 방향 표현]")
        print(terms.loc[(terms.model.eq("A_text_only")) & terms.direction.str.startswith("lower")]
              .head(15)[["source", "term", "coefficient", "train_document_share", "test_document_share"]]
              .to_string(index=False, float_format=fmt))
        stable = stability.loc[stability.model.eq("A_text_only")].appeared_in_random_top20.mean()
        print(f"\nA 모델 시간순 상위 표현 중 4주차 랜덤 분할 상위 20과 일치: {stable:.1%}")

        print("\n[N Stars 자동 제목 민감도 점검]")
        print(sensitivity.to_string(index=False, float_format=fmt))
        print("학습과 평가 시기의 자동 별점 제목 빈도 차이를 확인하되, "
              "제목 제거 후에도 결합 D가 정형 Ridge를 뛰어넘는지는 별도로 판단해야 합니다.")

        print("\n[결합 모델 D 최대 오차 사례]")
        print(error_cases.drop(columns="initial_review_excerpts").to_string(index=False, float_format=fmt))
        print("\n상품 단위 분리·학습 타깃 선행·초기 텍스트만 사용·학습 데이터 전처리 학습·예측 범위 검사: PASS")
        print(f"실행 시간: {perf_counter() - start:.1f}초")
    # DataFrame.to_string이 빈 칸 뒤에 남긴 공백을 제거해 Git 프린트를 깨끗하게 유지합니다.
    cleaned = "\n".join(line.rstrip() for line in output_path.read_text(encoding="utf-8").splitlines()) + "\n"
    output_path.write_text(cleaned, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
