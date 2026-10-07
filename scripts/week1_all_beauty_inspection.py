# 1주차: All_Beauty 리뷰 구조와 기초 통계 확인

from pathlib import Path
import sys

import pandas as pd


DATA_PATH = "../data/All_Beauty.jsonl.gz"

EXPECTED_COLUMNS = [
    "parent_asin", "rating", "title", "text", "timestamp",
    "verified_purchase", "helpful_vote",
]


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def load_reviews(path: Path) -> pd.DataFrame:
    """Parquet 또는 JSONL(압축 JSONL 포함)을 읽습니다."""
    if not path.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}\nDATA_PATH를 수정하세요.")
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if name.endswith((".jsonl", ".jsonl.gz", ".jsonl.bz2", ".jsonl.xz")):
        return pd.read_json(path, lines=True, compression="infer", convert_dates=False)
    raise ValueError("지원 형식: .parquet, .pq, .jsonl, .jsonl.gz/.bz2/.xz")


def inspect_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """구조·통계를 출력하고 상품별 시간순으로 정렬한 데이터프레임을 반환합니다."""
    section("1. 실제 파일의 데이터 구조")
    print("행·열 크기(shape):", df.shape)
    print("열 이름(columns):", df.columns.tolist())
    print("\n자료형(dtypes):\n", df.dtypes.to_string())
    print("\n처음 5개 리뷰(head):\n", df.head().to_string(index=False))

    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    print("\n주요 필드 중 누락된 열:", missing or "없음")
    required = [column for column in ["parent_asin", "rating", "timestamp"] if column not in df.columns]
    if required:
        raise ValueError(f"분석에 필요한 열이 없습니다: {required}. 원본 리뷰 파일인지 확인하세요.")
    print("\n주요 필드의 결측치 수:\n", df[[c for c in EXPECTED_COLUMNS if c in df]].isna().sum().to_string())
    if df.empty:
        print("빈 파일이므로 기초 통계와 정렬을 생략합니다.")
        return df.copy()

    section("2. 별점의 기초 통계와 분포")
    ratings = pd.to_numeric(df["rating"], errors="coerce")
    valid = ratings.isin([1, 2, 3, 4, 5])
    print("결측 또는 1~5 정수 별점이 아닌 행 수:", int((~valid).sum()))
    print("\n유효 별점의 기술 통계(describe):\n", ratings[valid].describe().to_string())
    print("표본 분산:", ratings[valid].var())
    counts = ratings[valid].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    distribution = pd.DataFrame({"리뷰 수": counts, "비율": counts / valid.sum() if valid.any() else float("nan")})
    distribution.index.name = "별점"
    print("\n별점별 리뷰 수와 비율:\n", distribution.to_string())
    print("전체 유효 리뷰 중 저평점(1~2점) 비율:", ratings[valid].le(2).mean())

    section("3. 상품별 리뷰 수")
    product_counts = df.groupby("parent_asin", dropna=True).size().sort_values(ascending=False)
    print("상품 식별자 결측 행 수:", int(df["parent_asin"].isna().sum()))
    print("상품 수:", len(product_counts))
    print("\n상품별 리뷰 수 기술 통계:\n", product_counts.describe().to_string())
    print("\n리뷰가 많은 상위 10개 상품:\n", product_counts.head(10).to_string())
    print("리뷰 30개 이상인 상품 수:", int(product_counts.ge(30).sum()))

    section("4. 상품별 작성 시각순 정렬")
    sorted_df = df.copy()
    numeric_time = pd.to_numeric(df["timestamp"], errors="coerce")
    sorted_df["review_datetime_utc"] = pd.to_datetime(numeric_time, unit="ms", errors="coerce", utc=True)
    print("시각 변환 실패 또는 결측 행 수:", int(sorted_df["review_datetime_utc"].isna().sum()))
    # 같은 시각은 원본 순서를 유지합니다.
    keys = pd.DataFrame({"product": df["parent_asin"].to_numpy(), "time": numeric_time.to_numpy()})
    positions = keys.sort_values(["product", "time"], kind="stable", na_position="last").index
    sorted_df = sorted_df.iloc[positions].reset_index(drop=True)
    view_columns = [c for c in EXPECTED_COLUMNS if c in sorted_df] + ["review_datetime_utc"]
    print("\n정렬한 데이터의 처음 10행:\n", sorted_df[view_columns].head(10).to_string(index=False))
    print("\n1주차 확인 완료: 구조, 별점 분포, 상품별 리뷰 수, 시간순 정렬을 확인했습니다.")
    print("다음 단계: 결측·중복 점검 후 상품별 초기 10개 및 후속 20개 리뷰 구성")
    return sorted_df


def main() -> None:
    path = Path(DATA_PATH).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    print("분석 파일:", path)
    df = load_reviews(path)
    sorted_df = inspect_reviews(df)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"\n실행 안내: {exc}", file=sys.stderr)
        sys.exit(1)
