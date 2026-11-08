# 4주차: 텍스트 전처리, TF-IDF, 리뷰 제목·본문 결합 모델
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2,<4", "scikit-learn>=1.5,<2", "matplotlib>=3.8"]
# ///
# How to run: python scripts/week4_text_models.py

from __future__ import annotations

import html
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
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import week3_structured_models as week3  # noqa: E402  3주차 기준 모델·분할·평가 재사용

import matplotlib.pyplot as plt  # noqa: E402  week3에서 Agg 백엔드를 설정함
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT: Final = week3.ROOT
WINDOWS: Final = ROOT / "reports/week2_review_windows.csv.gz"
SPLIT: Final = ROOT / "reports/week3_product_split.csv"
WEEK3_METRICS: Final = ROOT / "reports/week3_metrics.csv"
OUTPUT: Final = week3.OUTPUT
FIGURES: Final = week3.FIGURES
TARGET: Final = week3.TARGET
STRUCTURED_FEATURES: Final = week3.STRUCTURED_FEATURES
RANDOM_STATE: Final = week3.RANDOM_STATE
CV_FOLDS: Final = 5
BOOTSTRAP_ROUNDS: Final = 1000

# 하이퍼파라미터 탐색 범위(학습 상품의 5겹 교차 검증으로만 선택)
ALPHAS: Final = np.logspace(-2, 3, 16)
NGRAM_RANGES: Final = [(1, 1), (1, 2)]
MIN_DFS: Final = [5, 10]
TEXT_WEIGHTS: Final = [0.1, 0.3, 1.0, 3.0]  # 표준화된 정형 특성 대비 TF-IDF 블록의 상대 가중치
TFIDF_BASE: Final = {"max_df": 0.9, "sublinear_tf": True, "dtype": np.float32}

# 모델 이름 → 사용하는 입력 블록
TEXT_MODELS: Final = {
    "A_text_only": ["title", "body"],
    "B_structured_title": ["structured", "title"],
    "C_structured_body": ["structured", "body"],
    "D_structured_title_body": ["structured", "title", "body"],
}
TEXT_COLUMNS: Final = {"title": "title_doc", "body": "body_doc"}

TAG: Final = re.compile(r"<[^>]+>")
URL: Final = re.compile(r"https?://\S+|www\.\S+")
NUMBER: Final = re.compile(r"\d+(?:[.,]\d+)*")
SPACE: Final = re.compile(r"\s+")
NEGATION_SPECIAL: Final = [(re.compile(r"\bwon't\b"), "will not"), (re.compile(r"\bcan't\b"), "can not")]
NEGATION: Final = re.compile(r"n't\b")
STAR_TITLE: Final = re.compile(r"^\s*(one|two|three|four|five|[1-5])\s+stars?\s*$", re.IGNORECASE)


def clean_text(value: str) -> str:
    """HTML·URL·숫자를 정리하고 부정 표현(n't → not)을 보존합니다.

    sklearn 기본 영어 불용어 목록에는 not/no/nor가 포함되어 불만 표현이 사라지므로
    불용어 목록은 쓰지 않고, 흔한 단어는 TfidfVectorizer의 max_df로 거릅니다.
    """
    text = html.unescape(value)
    text = TAG.sub(" ", text).replace("’", "'").lower()
    for pattern, replacement in NEGATION_SPECIAL:
        text = pattern.sub(replacement, text)
    text = NEGATION.sub(" not", text)
    text = URL.sub(" urltoken ", text)
    text = NUMBER.sub(" numtoken ", text)
    return SPACE.sub(" ", text).strip()


def load_initial_reviews() -> pd.DataFrame:
    """초기 10개 리뷰만 불러오며 후속 구간 텍스트는 읽은 즉시 버립니다."""
    reviews = pd.read_csv(
        WINDOWS,
        usecols=["parent_asin", "window", "review_rank", "title", "text"],
        dtype={"parent_asin": "string", "title": "string", "text": "string"},
    )
    initial = reviews.loc[reviews.window.eq("initial")].copy()
    initial[["title", "text"]] = initial[["title", "text"]].fillna("")
    counts = initial.groupby("parent_asin").size()
    assert counts.eq(10).all(), "모든 상품의 초기 리뷰는 10개여야 합니다."
    assert initial.review_rank.between(1, 10).all()
    return initial.sort_values(["parent_asin", "review_rank"])


def build_documents(initial: pd.DataFrame, drop_star_titles: bool = False) -> pd.DataFrame:
    """상품별로 초기 10개 리뷰의 제목 문서와 본문 문서를 따로 만듭니다."""
    reviews = initial.copy()
    if drop_star_titles:
        reviews.loc[reviews.title.str.match(STAR_TITLE), "title"] = ""
    reviews["title_clean"] = reviews.title.map(clean_text)
    reviews["body_clean"] = reviews.text.map(clean_text)
    return reviews.groupby("parent_asin").agg(
        title_doc=("title_clean", " ".join), body_doc=("body_clean", " ".join)
    )


def make_model(parts: list[str], ngram: tuple[int, int] = (1, 1), min_df: int = 5,
               weight: float = 1.0, alpha: float = 1.0) -> Pipeline:
    """정형 표준화와 제목·본문 TF-IDF를 Pipeline 안에서 학습해 어휘 누수를 막습니다.

    제목·본문 벡터라이저는 같은 n-gram·min_df를 공유하고, 두 텍스트 블록에는
    같은 가중치(weight)를 곱합니다. 정형 특성 블록의 가중치는 1로 고정합니다.
    """
    transformers, weights = [], {}
    if "structured" in parts:
        transformers.append(("structured", StandardScaler(), STRUCTURED_FEATURES))
        weights["structured"] = 1.0
    for part in ["title", "body"]:
        if part in parts:
            vectorizer = TfidfVectorizer(ngram_range=ngram, min_df=min_df, **TFIDF_BASE)
            transformers.append((part, vectorizer, TEXT_COLUMNS[part]))
            weights[part] = weight
    features = ColumnTransformer(transformers, transformer_weights=weights)
    return Pipeline([("features", features), ("ridge", Ridge(alpha=alpha))])


def score_fold(parts: list[str], ngram: tuple[int, int], min_df: int, fold: int,
               train: pd.DataFrame, fit_index: np.ndarray, valid_index: np.ndarray) -> list[dict]:
    """한 fold에서 TF-IDF를 한 번만 학습하고, 텍스트 가중치·alpha만 바꿔 검증 MAE를 계산합니다.

    ColumnTransformer의 transformer_weights는 변환 결과 열에 상수를 곱하는 것과 같으므로
    GridSearchCV와 동일한 탐색 결과를 내면서 벡터라이저 재학습을 피할 수 있습니다.
    어휘·IDF·표준화 기준은 해당 fold의 학습 부분에서만 산출합니다.
    """
    fit_part, valid_part = train.iloc[fit_index], train.iloc[valid_index]
    features = make_model(parts, ngram, min_df).named_steps["features"]
    x_fit = sparse.csr_matrix(features.fit_transform(fit_part))
    x_valid = sparse.csr_matrix(features.transform(valid_part))
    is_text = np.zeros(x_fit.shape[1], dtype=bool)
    for part in ["title", "body"]:
        if part in features.output_indices_:
            is_text[features.output_indices_[part]] = True
    weights = TEXT_WEIGHTS if "structured" in parts else [1.0]
    rows = []
    for weight in weights:
        scale = sparse.diags(np.where(is_text, weight, 1.0))
        fit_weighted, valid_weighted = x_fit @ scale, x_valid @ scale
        for alpha in ALPHAS:
            ridge = Ridge(alpha=alpha).fit(fit_weighted, fit_part[TARGET])
            rows.append({
                "ngram_range": str(ngram), "min_df": min_df, "text_weight": weight,
                "alpha": alpha, "fold": fold,
                "mae": mean_absolute_error(valid_part[TARGET], ridge.predict(valid_weighted)),
            })
    return rows


def fit_text_models(train: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """모델 A~D를 학습 상품의 5겹 교차 검증으로 조정한 뒤 학습 상품 전체로 재학습합니다."""
    folds = list(KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(train))
    fitted, summary, cv_tables = {}, [], []
    for name, parts in TEXT_MODELS.items():
        start = perf_counter()
        jobs = [
            delayed(score_fold)(parts, ngram, min_df, fold, train, fit_index, valid_index)
            for ngram, min_df in product(NGRAM_RANGES, MIN_DFS)
            for fold, (fit_index, valid_index) in enumerate(folds)
        ]
        scores = pd.DataFrame([row for rows in Parallel(n_jobs=-1)(jobs) for row in rows])
        keys = ["ngram_range", "min_df", "text_weight", "alpha"]
        cv_table = scores.groupby(keys, sort=False).mae.agg(cv_mae="mean", cv_mae_std="std").reset_index()
        cv_table = cv_table.sort_values("cv_mae", kind="stable").assign(model=name)
        best = cv_table.iloc[0]
        ngram = tuple(int(v) for v in best.ngram_range.strip("()").split(","))
        model = make_model(parts, ngram, int(best.min_df), float(best.text_weight), float(best.alpha))
        fitted[name] = model.fit(train, train[TARGET])
        weight = f"{best.text_weight:g}" if "structured" in parts else "-"
        summary.append({
            "model": name, "cv_mae": best.cv_mae,
            "best_params": f"alpha={best.alpha:.4g}, ngram_range={best.ngram_range}, "
                           f"min_df={best.min_df}, text_weight={weight}",
            "fit_seconds": perf_counter() - start,
        })
        cv_tables.append(cv_table)
    return fitted, pd.DataFrame(summary), pd.concat(cv_tables, ignore_index=True)


def top_terms(best: Pipeline, model: str, n: int = 20) -> pd.DataFrame:
    """후속 저평점 비율을 높이는(+)·낮추는(−) 방향의 상위 텍스트 특성을 뽑습니다."""
    names = best.named_steps["features"].get_feature_names_out()
    coef = best.named_steps["ridge"].coef_
    table = pd.DataFrame({"feature": names, "coefficient": coef})
    table = table.loc[~table.feature.str.startswith("structured__")]
    table[["source", "term"]] = table.feature.str.split("__", n=1, expand=True)
    # 텍스트 블록 가중치를 곱해 TF-IDF 1단위당 실제 예측 기여도로 맞춥니다.
    weights = best.named_steps["features"].transformer_weights
    table["coefficient"] *= table.source.map(weights)
    up = table.nlargest(n, "coefficient").assign(direction="higher_future_low_ratio")
    down = table.nsmallest(n, "coefficient").assign(direction="lower_future_low_ratio")
    result = pd.concat([up, down]).assign(model=model)
    result["rank"] = result.groupby("direction").cumcount() + 1
    return result[["model", "direction", "rank", "source", "term", "coefficient"]]


def paired_bootstrap(predictions: pd.DataFrame, models: list[str], references: list[str]) -> pd.DataFrame:
    """같은 평가 상품을 복원추출해 (기준 MAE − 비교 모델 MAE)의 95% 구간을 구합니다."""
    rng = np.random.default_rng(RANDOM_STATE)
    index = rng.integers(0, len(predictions), size=(BOOTSTRAP_ROUNDS, len(predictions)))
    rows = []
    for reference in references:
        ref_error = predictions[f"{reference}_absolute_error"].to_numpy()
        for model in models:
            error = predictions[f"{model}_absolute_error"].to_numpy()
            diff = ref_error[index].mean(axis=1) - error[index].mean(axis=1)
            rows.append({
                "reference": reference, "model": model,
                "mae_reduction": ref_error.mean() - error.mean(),
                "ci_low": np.quantile(diff, 0.025), "ci_high": np.quantile(diff, 0.975),
                "share_improved": (diff > 0).mean(),
            })
    return pd.DataFrame(rows)


def tfidf_toy_example() -> pd.DataFrame:
    """보고서용 TF-IDF 예시: 부정 표현과 bigram이 어떻게 가중되는지 보여줍니다."""
    docs = [
        "great scent and great price",
        "does not work and broke after a week",
        "great product but does not last",
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([clean_text(doc) for doc in docs])
    table = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out(),
                         index=[f"doc{i + 1}" for i in range(len(docs))])
    keep = ["great", "not", "does not", "not work", "broke", "scent", "last", "and"]
    return table[keep].round(3)


def save_figure(metrics: pd.DataFrame, bootstrap: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """3주차 그림과 같은 스타일로 오차·개선 구간·예측 산점도를 저장합니다."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "axes.linewidth": .7, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    navy, light, accent = "#263E56", "#9CAAB7", "#A16B5D"
    labels = {
        "baseline_linear": "Mean-rating\nbaseline", "structured_ridge": "Structured\nRidge",
        "A_text_only": "A: Text\nonly", "B_structured_title": "B: Struct.\n+ title",
        "C_structured_body": "C: Struct.\n+ body", "D_structured_title_body": "D: Struct.\n+ title + body",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5), gridspec_kw={"width_ratios": [1.6, 1.1, 1]})

    main = metrics.loc[metrics.setting.eq("main")]
    x = np.arange(len(main))
    width = .36
    axes[0].bar(x - width / 2, main.mae, width, label="MAE", color=navy)
    axes[0].bar(x + width / 2, main.rmse, width, label="RMSE", color=light)
    axes[0].set_xticks(x, [labels[m] for m in main.model], fontsize=7.5)
    axes[0].set(title="(a) Holdout error", ylabel="Error", ylim=(0, main.rmse.max() * 1.25))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, fontsize=8)

    versus = bootstrap.loc[bootstrap.reference.eq("structured_ridge")].iloc[::-1]
    y = np.arange(len(versus))
    axes[1].errorbar(versus.mae_reduction, y,
                     xerr=[versus.mae_reduction - versus.ci_low, versus.ci_high - versus.mae_reduction],
                     fmt="o", color=accent, ecolor=accent, elinewidth=1.1, capsize=3, markersize=4)
    axes[1].axvline(0, color="#777777", linewidth=.8, linestyle="--")
    axes[1].set_yticks(y, [labels[m].replace("\n", " ") for m in versus.model], fontsize=8)
    axes[1].set(title="(b) MAE reduction vs. Structured", xlabel="MAE reduction (95% bootstrap CI)")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1, decimals=1))

    axes[2].scatter(predictions[TARGET], predictions["D_structured_title_body_prediction"], s=9,
                    alpha=.28, color=accent, edgecolors="none", rasterized=True)
    axes[2].plot([0, 1], [0, 1], color="#777777", linewidth=.8, linestyle="--")
    axes[2].set(title="(c) D: Struct. + title + body", xlabel="Observed future low-rating share",
                ylabel="Predicted share", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
    axes[2].xaxis.set_major_formatter(PercentFormatter(1))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))

    for ax in axes:
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(title, loc="left", pad=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    axes[0].grid(axis="y", color="#E7E7E7", linewidth=.5)
    axes[1].grid(axis="x", color="#E7E7E7", linewidth=.5)
    axes[2].grid(axis="y", color="#E7E7E7", linewidth=.5)
    fig.tight_layout(w_pad=1.8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"week4_model_comparison.{extension}", dpi=400,
                    bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def main() -> None:
    start = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    products = week3.load_products().set_index("parent_asin")
    initial = load_initial_reviews()
    documents = build_documents(initial)
    assert documents.index.equals(products.index.sort_values()), "문서와 상품 목록이 일치해야 합니다."
    data = products.join(documents).reset_index()

    # 3주차와 동일한 상품 분할을 사용하고, 저장된 분할과 재현 결과가 같은지 확인합니다.
    split = pd.read_csv(SPLIT, dtype={"parent_asin": "string"}).set_index("parent_asin").split
    week3_train, week3_test = week3.split_products(data)
    assert set(week3_test.parent_asin) == set(split[split.eq("test")].index)
    train = data.loc[data.parent_asin.map(split).eq("train")].sort_values("parent_asin")
    test = data.loc[data.parent_asin.map(split).eq("test")].sort_values("parent_asin")
    assert set(train.parent_asin).isdisjoint(test.parent_asin)
    assert len(train) + len(test) == len(data)

    # 3주차 기준 모델 두 개를 그대로 다시 학습하고 기록된 성능과 같은지 확인합니다.
    week3_metrics, predictions, _, _ = week3.fit_models(train, test)
    recorded = pd.read_csv(WEEK3_METRICS).set_index("model")
    for row in week3_metrics.itertuples():
        assert np.isclose(row.mae, recorded.loc[row.model, "mae"], atol=1e-9), "3주차 결과 재현 실패"
    predictions = predictions.rename(columns={
        "baseline_prediction": "baseline_linear_prediction",
        "structured_prediction": "structured_ridge_prediction",
        "baseline_absolute_error": "baseline_linear_absolute_error",
        "structured_absolute_error": "structured_ridge_absolute_error",
    }).set_index("parent_asin")

    fitted, search_summary, cv_results = fit_text_models(train)
    rows = [{**row, "setting": "main"} for row in week3_metrics.to_dict("records")]
    for name, model in fitted.items():
        raw = model.predict(test)
        metrics, clipped = week3.evaluate(name, test[TARGET], raw)
        rows.append({**metrics, "setting": "main"})
        predictions[f"{name}_prediction"] = pd.Series(clipped, index=test.parent_asin)
        predictions[f"{name}_absolute_error"] = (predictions[TARGET] - predictions[f"{name}_prediction"]).abs()

    # 민감도 점검: "Five Stars" 같은 자동 생성 제목을 지우고 모델 D를 같은 설정으로 재학습합니다.
    star_share = initial.title.str.match(STAR_TITLE).mean()
    no_star = data.drop(columns=["title_doc", "body_doc"]).set_index("parent_asin").join(
        build_documents(initial, drop_star_titles=True)).reset_index()
    no_star_train = no_star.set_index("parent_asin").loc[train.parent_asin].reset_index()
    no_star_test = no_star.set_index("parent_asin").loc[test.parent_asin].reset_index()
    sensitivity = clone(fitted["D_structured_title_body"]).fit(
        no_star_train, no_star_train[TARGET])
    metrics, _ = week3.evaluate("D_without_star_titles", no_star_test[TARGET], sensitivity.predict(no_star_test))
    rows.append({**metrics, "setting": "sensitivity"})

    metrics = pd.DataFrame(rows).merge(search_summary, on="model", how="left")
    predictions = predictions.reset_index()
    assert predictions.filter(like="_prediction").apply(lambda c: c.between(0, 1).all()).all()

    bootstrap = paired_bootstrap(predictions, list(TEXT_MODELS),
                                 ["baseline_linear", "structured_ridge"])
    terms = pd.concat([top_terms(fitted["A_text_only"], "A_text_only"),
                       top_terms(fitted["D_structured_title_body"], "D_structured_title_body")])

    metrics.to_csv(OUTPUT / "week4_metrics.csv", index=False)
    predictions.to_csv(OUTPUT / "week4_predictions.csv", index=False)
    bootstrap.to_csv(OUTPUT / "week4_bootstrap.csv", index=False)
    terms.to_csv(OUTPUT / "week4_top_terms.csv", index=False)
    cv_results.to_csv(OUTPUT / "week4_cv_results.csv", index=False)
    save_figure(metrics, bootstrap, predictions)

    vocab = fitted["D_structured_title_body"].named_steps["features"]
    # 정제 효과가 드러나도록 <br> 태그 주변을 예시로 보여줍니다.
    sample_text = initial.loc[initial.text.str.contains("<br", regex=False)].iloc[0].text
    tag_at = sample_text.find("<br")
    sample = sample_text[max(0, tag_at - 60): tag_at + 80]
    with (OUTPUT / "week4_output.txt").open("w", encoding="utf-8") as output, redirect_stdout(output):
        fmt = lambda value: f"{value:.6f}"  # noqa: E731
        print("# 4주차 텍스트 전처리·TF-IDF·제목·본문 결합 모델")
        print("텍스트 입력:", WINDOWS.relative_to(ROOT), "(window == 'initial'만 사용)")
        print(f"분석 상품 수: {len(data):,} / 초기 리뷰 수: {len(initial):,}")
        print(f"3주차와 동일한 상품 분할: 학습 {len(train):,}개 / 평가 {len(test):,}개")
        print("3주차 기준 모델 재현 검사: PASS")

        print("\n[TF-IDF 예시] (ngram_range=(1, 2), smooth_idf, L2 정규화)")
        print(tfidf_toy_example().to_string())

        print("\n[텍스트 전처리]")
        print("HTML 엔티티 해제 → 태그 제거 → 소문자 → n't를 not으로 분리 → URL·숫자 토큰화 → 공백 정리")
        print("불용어 목록 미사용(not/no 보존), 흔한 단어는 max_df=0.9로 제외")
        print("<br> 포함 본문 비율:", f"{initial.text.str.contains('<br', regex=False).mean():.2%}")
        print("정제 전 예시:", sample.replace("\n", " "))
        print("정제 후 예시:", clean_text(sample))
        print(f"상품별 본문 문서 평균 단어 수: {data.body_doc.str.split().str.len().mean():.1f}")
        print(f"상품별 제목 문서 평균 단어 수: {data.title_doc.str.split().str.len().mean():.1f}")
        print(f"'N Stars' 형식 자동 제목 비율: {star_share:.2%}")
        for part in ["title", "body"]:
            size = len(vocab.named_transformers_[part].vocabulary_)
            print(f"모델 D {part} 어휘 크기(학습 상품에서만 산출): {size:,}")

        print(f"\n[하이퍼파라미터] 학습 상품 {CV_FOLDS}겹 교차 검증, 평가 데이터는 최종 1회만 사용")
        print(search_summary.to_string(index=False, float_format=fmt))

        print("\n[평가 결과] 예측값을 0~1로 제한")
        print(metrics[["model", "setting", "mae", "rmse", "r2", "raw_below_zero", "raw_above_one"]]
              .to_string(index=False, float_format=fmt))
        base_mae = metrics.set_index("model").mae
        for name in TEXT_MODELS:
            print(f"{name}: 기준 모델 대비 MAE {1 - base_mae[name] / base_mae['baseline_linear']:.2%}, "
                  f"정형 Ridge 대비 {1 - base_mae[name] / base_mae['structured_ridge']:.2%}")

        print(f"\n[대응 부트스트랩 {BOOTSTRAP_ROUNDS}회] MAE 감소량(양수 = 개선)과 95% 구간")
        print(bootstrap.to_string(index=False, float_format=fmt))

        print("\n[모델 A 상위 텍스트 특성] 후속 저평점 비율을 높이는 방향")
        a_terms = terms.loc[terms.model.eq("A_text_only")]
        print(a_terms.loc[a_terms.direction.str.startswith("higher")].head(15)
              [["source", "term", "coefficient"]].to_string(index=False, float_format=fmt))
        print("\n[모델 A 상위 텍스트 특성] 후속 저평점 비율을 낮추는 방향")
        print(a_terms.loc[a_terms.direction.str.startswith("lower")].head(15)
              [["source", "term", "coefficient"]].to_string(index=False, float_format=fmt))

        print("\n상품 분할·초기 구간 텍스트만 사용·어휘 학습 데이터 한정·예측 범위 검사: PASS")
        print(f"실행 시간: {perf_counter() - start:.1f}초")
    print(OUTPUT / "week4_output.txt")


if __name__ == "__main__":
    main()
