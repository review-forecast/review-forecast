# 2주차: 데이터 품질 점검과 초기·후속 리뷰 구성
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2,<4"]
# ///
# How to run: python scripts/week2_all_beauty_analysis.py
# 또는 uv run scripts/week2_all_beauty_analysis.py

from __future__ import annotations

import hashlib
import json
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
from typing import Final

import pandas as pd  # 기존 작업자 스타일을 따르라는 사용자 요청

ROOT: Final = Path(__file__).resolve().parents[1]
DATA_PATH: Final = ROOT / "data/All_Beauty.jsonl.gz"
OUTPUT: Final = ROOT / "reports"
COLUMNS: Final = ["parent_asin", "user_id", "asin", "rating", "title", "text",
                  "timestamp", "verified_purchase", "helpful_vote"]
REVIEW_KEY: Final = ["parent_asin", "user_id", "timestamp", "rating", "title", "text"]


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def load_reviews(path: Path) -> pd.DataFrame:
    """압축 원본을 나누어 읽고 전체 행 중복 판정용 해시를 보존합니다."""
    parts = []
    with pd.read_json(path, lines=True, chunksize=25000, convert_dates=False) as reader:
        for chunk in reader:
            # images처럼 리스트인 열도 포함한 전체 행 기준입니다.
            hashes = [hashlib.sha256(json.dumps(row, sort_keys=True,
                       ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
                      for row in chunk.to_dict("records")]
            part = chunk[COLUMNS].copy()
            part["source_row"] = chunk.index + 1
            part["full_row_hash"] = hashes
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """필수 값이 유효한 행에서 완전 중복만 제거합니다."""
    section("1. 결측치·빈 문자열·중복 점검")
    print("원본 리뷰 수:", len(df))
    print("원본 상품 수:", df.parent_asin.nunique())
    print("원본 리뷰 30개 이상 상품 수:", int(df.groupby("parent_asin").size().ge(30).sum()))
    print("\n필드별 결측치:\n", df[COLUMNS].isna().sum().to_string())
    for column in ["parent_asin", "user_id", "title", "text"]:
        print(f"{column} 빈 문자열·공백 수:", int(df[column].fillna("").str.strip().eq("").sum()))
    print("전체 필드가 동일한 중복 행 수(첫 행 제외):", int(df.full_row_hash.duplicated().sum()))
    print("상품·작성자·시각·별점·제목·본문 동일 중복 행 수:", int(df.duplicated(REVIEW_KEY).sum()))
    print("상품·작성자 반복 행 수:", int(df.duplicated(["parent_asin", "user_id"]).sum()))
    numeric_time = pd.to_numeric(df.timestamp, errors="coerce")
    time = pd.to_datetime(numeric_time, unit="ms", errors="coerce", utc=True)
    rating = pd.to_numeric(df.rating, errors="coerce")
    valid = df.parent_asin.fillna("").str.strip().ne("") & rating.isin(range(1, 6)) & time.notna()
    print("필수 식별자·별점·시각 문제로 제외한 행 수:", int((~valid).sum()))
    cleaned = df.loc[valid].drop_duplicates("full_row_hash", keep="first").copy()
    cleaned["rating"] = rating.loc[cleaned.index]
    cleaned["timestamp"] = numeric_time.loc[cleaned.index]
    cleaned["review_datetime_utc"] = time.loc[cleaned.index]
    cleaned["title"] = cleaned.title.fillna("")
    cleaned["text"] = cleaned.text.fillna("")
    print("처리 기준: 필수 값 문제 및 완전 중복만 제거; 빈 텍스트와 반복 작성자는 유지")
    print("정제 후 리뷰 수:", len(cleaned))
    print("정제 후 상품 수:", cleaned.parent_asin.nunique())
    print("정제 후 리뷰 30개 이상 상품 수:", int(cleaned.groupby("parent_asin").size().ge(30).sum()))
    count_audit = df.groupby("parent_asin").size().rename("raw_review_count").to_frame()
    count_audit["clean_review_count"] = cleaned.groupby("parent_asin").size().reindex(count_audit.index, fill_value=0)
    count_audit.loc[count_audit.raw_review_count.ge(30) & count_audit.clean_review_count.lt(30)].to_csv(
        OUTPUT / "week2_excluded_products.csv")
    return cleaned


def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    """동일 시각은 원본 행 순서로 정렬하며 경계 동시각 상품을 표시합니다."""
    ordered = df.sort_values(["parent_asin", "timestamp", "source_row"], kind="stable").copy()
    ordered["review_count"] = ordered.groupby("parent_asin").rating.transform("size")
    ordered["review_rank"] = ordered.groupby("parent_asin").cumcount() + 1
    windows = ordered.loc[ordered.review_count.ge(30) & ordered.review_rank.le(30)].copy()
    windows["window"] = "followup"
    windows.loc[windows.review_rank.le(10), "window"] = "initial"
    windows["low_rating"] = windows.rating.le(2)
    windows["text_length"] = windows.text.str.len()
    windows["title_length"] = windows.title.str.len()
    windows["text_present"] = windows.text.str.strip().ne("")
    return windows


def summarize_products(windows: pd.DataFrame) -> pd.DataFrame:
    """説明変数は初期10件だけから算出し、後続情報を別名で保持します。"""
    initial = windows.loc[windows.window.eq("initial")]
    followup = windows.loc[windows.window.eq("followup")]
    products = initial.groupby("parent_asin").agg(
        review_count=("review_count", "first"), initial_mean_rating=("rating", "mean"),
        initial_rating_std=("rating", "std"), initial_low_ratio=("low_rating", "mean"),
        initial_mean_text_length=("text_length", "mean"),
        initial_mean_title_length=("title_length", "mean"),
        initial_text_present_ratio=("text_present", "mean"),
        initial_verified_ratio=("verified_purchase", "mean"),
        initial_start_timestamp=("timestamp", "min"), initial_end_timestamp=("timestamp", "max"),
    )
    for rating in range(1, 6):
        products[f"initial_rating_{rating}_ratio"] = initial.rating.eq(rating).groupby(initial.parent_asin).mean()
    targets = followup.groupby("parent_asin").agg(
        followup_low_count=("low_rating", "sum"), followup_low_ratio=("low_rating", "mean"),
        followup_mean_rating=("rating", "mean"), followup_start_timestamp=("timestamp", "min"),
        followup_end_timestamp=("timestamp", "max"),
    )
    products = products.join(targets)
    products["boundary_same_timestamp"] = products.initial_end_timestamp.eq(products.followup_start_timestamp)
    return products


def analyze(products: pd.DataFrame, windows: pd.DataFrame) -> None:
    section("2. 초기 10개·후속 20개 리뷰 구성")
    print("분석 상품 수:", len(products))
    print("초기·후속 리뷰 수:\n", windows.window.value_counts().to_string())
    print("초기·후속 경계 시각이 같은 상품 수:", int(products.boundary_same_timestamp.sum()))
    print("동일 시각은 원본 행 순서. 엄격한 시간순 평가에서는 경계 동시각 상품을 별도로 처리해야 합니다.")
    print("신청서의 3,648개 상품과 차이:", len(products) - 3648)
    section("3. 상품별 탐색적 분석")
    features = ["initial_mean_rating", "initial_rating_std", "initial_low_ratio",
                "initial_mean_text_length", "initial_verified_ratio", "followup_low_ratio"]
    print("\n상품별 기술 통계:\n", products[features].describe().to_string())
    print("후속 저평점 비율이 0인 상품 수:", int(products.followup_low_ratio.eq(0).sum()))
    correlations = products[features].corr()["followup_low_ratio"].drop("followup_low_ratio")
    print("\n후속 저평점 비율과 Pearson 상관계수:\n", correlations.to_string())
    correlations.rename("pearson_r").to_csv(OUTPUT / "week2_correlations.csv")
    # 초기 평균이 정확히 같은 그룹에서 후속 비율 차이가 큰 3쌍을 선택합니다.
    spans = products.groupby("initial_mean_rating").followup_low_ratio.agg(["min", "max", "size"])
    spans["gap"] = spans["max"] - spans["min"]
    spans = spans.loc[spans["size"].ge(2)].sort_values(["gap", "size"], ascending=False).head(3)
    examples = []
    for mean in spans.index:
        group = products.loc[products.initial_mean_rating.eq(mean)].sort_values(["followup_low_ratio", "parent_asin"])
        examples.extend([group.index[0], group.index[-1]])
    print("\n초기 평균은 같지만 후속 평가가 다른 상품 사례:\n", products.loc[examples, features].to_string())
    products.loc[examples].to_csv(OUTPUT / "week2_case_products.csv")
    windows.loc[windows.parent_asin.isin(examples)].drop(columns="full_row_hash").to_csv(
        OUTPUT / "week2_case_reviews.csv", index=False)
    print("사례는 차이가 큰 상품을 의도적으로 선택한 탐색용이며 대표성이나 예측 성능을 입증하지 않습니다.")


def main() -> None:
    start = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "week2_output.txt").open("w", encoding="utf-8") as output, redirect_stdout(output):
        print("분석 파일:", DATA_PATH)
        reviews = clean_reviews(load_reviews(DATA_PATH))
        windows = build_windows(reviews)
        products = summarize_products(windows)
        counts = windows.groupby(["parent_asin", "window"]).size().unstack()
        assert counts.initial.eq(10).all() and counts.followup.eq(20).all()
        assert windows.source_row.is_unique
        assert products.initial_end_timestamp.le(products.followup_start_timestamp).all()
        assert products.followup_low_ratio.eq(products.followup_low_count / 20).all()
        assert products.followup_low_ratio.between(0, 1).all()
        products.to_csv(OUTPUT / "week2_products.csv")
        windows.drop(columns="full_row_hash").to_csv(OUTPUT / "week2_review_windows.csv.gz", index=False)
        analyze(products, windows)
        print(f"\n구간별 건수·시간순·목표값 정합성 검사: PASS / 실행 시간: {perf_counter() - start:.1f}초")
    print(OUTPUT / "week2_output.txt")


if __name__ == "__main__":
    main()
