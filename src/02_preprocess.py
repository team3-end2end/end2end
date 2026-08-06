"""Step 2 — 전처리: 원본을 정제해 cleaned.parquet 을 만든다.

데이터를 바꾸는 유일한 단계다. 시각화·통계 분석은 하지 않는다.
Step 3~6 전체가 이 스크립트의 산출물을 입력으로 받는다.

산출물
    data/processed/cleaned.parquet      Step 3·4·5·6 의 입력
    outputs/results/preprocess.json     Step 7(report) 이 읽는 수치
    outputs/results/preprocess.md       사람이 이 단계만 따로 확인하는 용도

설계 메모
    - 결측을 채우지 않는다. Flex Fare 행의 5개 컬럼은 결함이 아니라 '해당 없음'이므로
      임의 대체하면 없던 값을 만들어내는 셈이 된다. 비어 있는 채로 넘기고,
      Step 6 은 features.json 화이트리스트로 해당 컬럼을 아예 쓰지 않는다.
    - 이상값 제거는 '세는 조건'과 '지우는 조건'을 서로의 정확한 부정으로 맞췄다.
      (예: 지울 때 distance >= 200, 남길 때 distance < 200) 경계값이 새는 것을 막는다.
    - 요금에 상한을 두지 않은 것은 의도한 비대칭이다. 거리 200마일·소요 180분은
      물리적으로 불가능에 가깝지만, 고액 요금은 장거리·정액요금으로 실제 발생한다.
"""

import json
import os
import urllib.request

import pandas as pd

RAW_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet"
RAW_PATH = "data/raw/yellow_tripdata_2026-05.parquet"
OUT_PARQUET = "data/processed/cleaned.parquet"
OUT_JSON = "outputs/results/preprocess.json"
OUT_MD = "outputs/results/preprocess.md"

# 분석 대상 클래스. 3(무료)·4(분쟁)은 합산 0.85%로 학습이 불가능해 제외한다.
PAYMENT_LABELS = {0: "Flex Fare", 1: "신용카드", 2: "현금"}
TARGET_CODES = list(PAYMENT_LABELS)

# 이상값 제거 기준
MIN_FARE = 0.0
MAX_DISTANCE = 200.0
MAX_DURATION_MIN = 180.0

# Flex Fare 행에서만 비어 있는 컬럼. 결측이 아니라 '해당 없음'이다.
NA_BY_DESIGN = ["passenger_count", "RatecodeID", "store_and_fwd_flag",
                "congestion_surcharge", "Airport_fee"]


def fetch_raw() -> pd.DataFrame:
    """원본을 로딩한다. 없으면 TLC 공식 배포처에서 내려받는다(약 66MB)."""
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    if not os.path.exists(RAW_PATH):
        print(f"[다운로드] {RAW_PATH} ...")
        urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    else:
        print(f"[기존 파일] {RAW_PATH}")
    return pd.read_parquet(RAW_PATH)


def diagnose(df: pd.DataFrame) -> dict:
    """정제 전 현황을 파악한다. 중복 0건을 출력하는 것도 '처리했다'는 증빙이다."""
    print("\n[1] 원본 현황")
    n_dup = int(df.duplicated().sum())
    miss = df.isnull().sum()
    miss = miss[miss > 0]

    print(f"    형태          : {df.shape[0]:,}행 × {df.shape[1]}열")
    print(f"    완전 중복 행  : {n_dup:,}건")
    print(f"    결측 컬럼     : {len(miss)}개 (각 {miss.iloc[0]:,}건)" if len(miss) else "    결측 없음")

    # 결측 구조 교차검증 — 무작위 결측이 아니라 특정 클래스의 구조적 특성임을 보인다
    same_rows = bool(((df.payment_type == 0) == df.passenger_count.isna()).all())
    print(f"    결측 행 == payment_type 0(Flex Fare) 행 : {same_rows}")
    print("      → 무작위 결측(MCAR)이 아니라 '해당 없음'. 대체하지 않고 그대로 둔다.")

    return {"raw_shape": list(df.shape), "duplicates": n_dup,
            "missing_cols": {c: int(v) for c, v in miss.items()},
            "missing_matches_flexfare": same_rows}


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """분석 대상 선별 → 이상값 제거 → 파생변수 생성."""
    print("\n[2] 정제")
    before = len(df)

    # 완전 중복 제거 (0건이어도 처리 경로는 남긴다)
    df = df.drop_duplicates()

    # 분석 대상 클래스만 남긴다
    df = df[df.payment_type.isin(TARGET_CODES)].copy()
    n_class = before - len(df)
    print(f"    payment_type ∈ {TARGET_CODES} 외 제외 : {n_class:,}건")

    # 소요시간은 이상값 판정에 쓰이므로 필터 전에 만든다
    df["trip_duration_min"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60

    # 규칙별 해당 건수를 각각 센다 (PR·리포트의 근거로 사용)
    # 세는 조건과 지우는 조건은 서로의 정확한 부정이다
    rules = {
        "fare_amount <= 0": int((df.fare_amount <= MIN_FARE).sum()),
        "trip_distance <= 0": int((df.trip_distance <= 0).sum()),
        f"trip_distance >= {MAX_DISTANCE:.0f}": int((df.trip_distance >= MAX_DISTANCE).sum()),
        "trip_duration_min <= 0": int((df.trip_duration_min <= 0).sum()),
        f"trip_duration_min >= {MAX_DURATION_MIN:.0f}": int(
            (df.trip_duration_min >= MAX_DURATION_MIN).sum()),
    }
    for k, v in rules.items():
        print(f"    {k:32s} : {v:>9,}건")

    df = df[(df.fare_amount > MIN_FARE)
            & (df.trip_distance > 0) & (df.trip_distance < MAX_DISTANCE)
            & (df.trip_duration_min > 0) & (df.trip_duration_min < MAX_DURATION_MIN)]

    # 파생변수. is_weekend 는 사람이 읽는 요약용이며 day_of_week 에서 완전히 유도되므로
    # 모델 피처로는 쓰지 않는다(Step 3 의 화이트리스트에서 제외).
    df = df.copy()
    df["pickup_hour"] = df.tpep_pickup_datetime.dt.hour
    df["day_of_week"] = df.tpep_pickup_datetime.dt.dayofweek
    df["is_weekend"] = (df.day_of_week >= 5).astype(int)
    df["payment_label"] = df.payment_type.map(PAYMENT_LABELS)

    dist = df.payment_label.value_counts(normalize=True).mul(100).round(2)
    print(f"\n    정제 후 : {len(df):,}행 (보존율 {len(df) / before:.2%})")
    print("    클래스 분포 :", ", ".join(f"{k} {v}%" for k, v in dist.items()))

    return df, {"rules": rules, "excluded_by_class": n_class,
                "clean_shape": list(df.shape), "retain_rate": round(len(df) / before, 4),
                "class_pct": dist.to_dict()}


def verify(df: pd.DataFrame):
    """정제가 의도대로 됐는지 단언한다. 실패하면 여기서 멈춘다."""
    print("\n[3] 검증")

    assert set(df.payment_type.unique()) <= set(TARGET_CODES), "대상 외 클래스가 남아 있음"
    print(f"    payment_type ⊆ {TARGET_CODES} : 통과")

    # 남은 결측이 전부 Flex Fare × 5컬럼인지 — 결함이 아니라 '해당 없음'임을 확인
    n_flex = int((df.payment_type == 0).sum())
    expected = n_flex * len(NA_BY_DESIGN)
    actual = int(df[NA_BY_DESIGN].isnull().sum().sum())
    assert actual == expected, f"결측 {actual}개 ≠ Flex Fare {n_flex}행 × {len(NA_BY_DESIGN)}컬럼"
    print(f"    남은 결측 {actual:,}개 = Flex Fare {n_flex:,}행 × {len(NA_BY_DESIGN)}컬럼 : 통과")

    other = [c for c in df.columns if c not in NA_BY_DESIGN]
    assert df[other].isnull().sum().sum() == 0, "설계 외 컬럼에 결측이 남아 있음"
    print("    그 밖의 컬럼에 결측 없음 : 통과")


def write_results(diag: dict, prep: dict, df: pd.DataFrame):
    """JSON(기계용)과 MD(사람용)를 한 쌍으로 남긴다. Step 7이 JSON을 조립한다."""
    payload = {"step": "02_preprocess", **diag, **prep,
               "output": OUT_PARQUET,
               "na_by_design_cols": NA_BY_DESIGN}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    rules_md = "\n".join(f"| `{k}` | {v:,} |" for k, v in prep["rules"].items())
    cls_md = "\n".join(f"| {k} | {v}% |" for k, v in prep["class_pct"].items())
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(f"""# Step 2 — 전처리

원본 {diag['raw_shape'][0]:,}행 × {diag['raw_shape'][1]}열 → 정제 후 **{prep['clean_shape'][0]:,}행 × {prep['clean_shape'][1]}열**
(보존율 {prep['retain_rate']:.2%})

## 제거 규칙별 해당 건수

| 규칙 | 건수 |
|---|---|
{rules_md}

`payment_type ∉ {TARGET_CODES}` 제외: {prep['excluded_by_class']:,}건 (무료·분쟁, 학습 불가)

## 클래스 분포

| 결제수단 | 비율 |
|---|---|
{cls_md}

## 결측 처리

결측 {sum(diag['missing_cols'].values()):,}건은 전부 `payment_type=0`(Flex Fare) 행에서 발생하며
(교차검증: {diag['missing_matches_flexfare']}), 결함이 아니라 **해당 없음**이다.
미터기 세부 요금 필드가 애초에 존재하지 않는 요금제이므로 **대체하지 않고 그대로 둔다.**
해당 컬럼: {', '.join(f'`{c}`' for c in NA_BY_DESIGN)}
""")
    print(f"\n[4] 산출물\n    {OUT_PARQUET}\n    {OUT_JSON}\n    {OUT_MD}")


def main():
    raw = fetch_raw()
    diag = diagnose(raw)
    df, prep = clean(raw)
    verify(df)
    df.to_parquet(OUT_PARQUET, index=False)
    write_results(diag, prep, df)
    print("\n완료 — Step 3·4·5·6 은 data/processed/cleaned.parquet 을 읽으면 됩니다.")


if __name__ == "__main__":
    main()
