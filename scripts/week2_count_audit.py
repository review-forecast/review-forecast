# 2주차: 신청서 상품 수의 산출 조건 비교
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2,<4"]
# ///
# How to run: python scripts/week2_count_audit.py
# 또는 uv run scripts/week2_count_audit.py
from __future__ import annotations

from contextlib import redirect_stdout
from time import perf_counter

import pandas as pd  # 기존 작업자 스타일을 따르라는 사용자 요청
from week2_all_beauty_analysis import DATA_PATH, OUTPUT, REVIEW_KEY, load_reviews


def main() -> None:
    """정제 조건별 건수와 현재 분석 상품 집합과의 차이를 저장합니다."""
    start = perf_counter()
    df = load_reviews(DATA_PATH)
    baseline = set(pd.read_csv(OUTPUT / "week2_products.csv").parent_asin)
    variants = {
        "raw": df.index,
        "full_row_duplicate_removed": df.drop_duplicates("full_row_hash").index,
        "review_key_duplicate_removed": df.drop_duplicates(REVIEW_KEY).index,
        "parent_user_duplicate_removed": df.drop_duplicates(["parent_asin", "user_id"]).index,
        "asin_user_duplicate_removed": df.drop_duplicates(["asin", "user_id"]).index,
        "parent_user_time_duplicate_removed": df.drop_duplicates(["parent_asin", "user_id", "timestamp"]).index,
        "parent_user_text_duplicate_removed": df.drop_duplicates(["parent_asin", "user_id", "text"]).index,
        "text_present": df.index[df.text.fillna("").str.strip().ne("")],
        "verified_only": df.index[df.verified_purchase.eq(True)],
    }
    rows = []
    differences = []
    for name, positions in variants.items():
        subset = df.loc[positions]
        for product_key in ["parent_asin", "asin"]:
            counts = subset.groupby(product_key).size()
            for minimum in [30, 31]:
                ids = set(counts.index[counts.ge(minimum)])
                rows.append({"rule": name, "product_key": product_key,
                             "minimum_reviews": minimum, "reviews": len(subset),
                             "eligible_products": len(ids), "difference_from_3648": len(ids)-3648})
                if product_key == "parent_asin" and minimum == 30:
                    for product in sorted(ids ^ baseline):
                        differences.append({"rule": name, "parent_asin": product,
                                            "in_current_analysis": product in baseline,
                                            "in_variant": product in ids})
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "week2_count_audit.csv", index=False)
    pd.DataFrame(differences).to_csv(OUTPUT / "week2_count_audit_differences.csv", index=False)
    with (OUTPUT / "week2_count_audit_output.txt").open("w", encoding="utf-8") as output, redirect_stdout(output):
        print(result.to_string(index=False))
        print("\n3,648개와 일치하는 조건:\n", result.loc[result.eligible_products.eq(3648)].to_string(index=False))
        print("일치하더라도 당시 신청서의 계산 기준이었다는 증거는 아닙니다.")
        print(f"실행 시간: {perf_counter()-start:.1f}초")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
