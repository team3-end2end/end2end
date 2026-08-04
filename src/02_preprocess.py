"""
Step 2 — 전처리

원본 409만 행을 분석 가능한 형태로 정제해 `cleaned.parquet`을 만든다.
**데이터를 바꾸는 유일한 단계**이며, 탐색·통계·시각화는 하지 않는다(Step 3·4·5 담당).

정제 순서(이 순서를 지켜야 하는 이유는 각 함수 주석 참조):
  1) 분석 대상 클래스 선별  — payment_type ∈ {0(Flex Fare), 1(신용카드), 2(현금)}
  2) 이상값 제거            — 규칙별 해당 건수를 각각 기록
  3) 파생변수 생성          — 소요시간·시각·요일·주말 여부·라벨
  4) 검증 후 저장

Step 1의 결론에 따라 무거운 처리는 Polars로 수행하고, 저장 직전에 Pandas로 변환한다
(Step 3 이후의 seaborn·scipy·scikit-learn이 Pandas 객체를 입력으로 받으므로).

산출물:
  - data/processed/cleaned.parquet        정제 결과 (Step 3·4·5·6의 입력)
  - outputs/results/preprocess.json       수치 결과 (Step 7의 report.md 생성용)
  - outputs/results/preprocess.md         사람이 읽는 요약
"""

import json
from pathlib import Path

import polars as pl

# --- 경로 상수 ---
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "yellow_tripdata_2026-05.parquet"
CLEAN_PATH = ROOT / "data" / "processed" / "cleaned.parquet"
RESULT_DIR = ROOT / "outputs" / "results"

# --- 분석 설정 ---
# payment_type 코드 → 라벨. 3(무료)·4(분쟁)는 합산 0.85%로 학습이 불가능해 제외한다.
PAYMENT_LABELS = {0: "Flex Fare", 1: "신용카드", 2: "현금"}
TARGET_CODES = list(PAYMENT_LABELS)

# 타깃 누수 컬럼 — 정제 결과 파일에는 남기되(Step 3에서 누수 근거를 제시해야 하므로),
# Step 6의 모델 피처로는 절대 사용하지 않는다. 화이트리스트는 Step 3이 확정한다.
#   앞 2개: 현금 팁은 기록되지 않아 결제수단을 그대로 노출
#   뒤 5개: Flex Fare에서만 비어 있어 "비어 있으면 Flex Fare"를 외우게 만듦
LEAKAGE_COLS = [
    "tip_amount",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
    "RatecodeID",
    "passenger_count",
    "store_and_fwd_flag",
]

# 이상값 제거 기준. 팀 리포트와 동일한 기준을 사용해 수치를 비교 가능하게 유지한다.
#
# 요금에만 상한이 없는 것은 의도한 비대칭이다. 거리 200마일·소요시간 180분은 물리적으로
# 불가능에 가까운 기록이지만, 고액 요금은 장거리·정액요금으로 실제 발생할 수 있다.
# 꼬리도 얇다 — 99.99% 분위 $286.90, $500 초과 24건.
MIN_FARE = 0.0
MAX_DISTANCE = 200.0
MAX_DURATION_MIN = 180.0


def load_raw() -> pl.DataFrame:
    """원본 parquet을 Polars로 읽는다."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"원본 데이터가 없습니다: {RAW_PATH}\n"
            "README.md의 curl 명령으로 data/raw/ 에 내려받으세요."
        )
    return pl.read_parquet(RAW_PATH)


def profile_raw(df: pl.DataFrame) -> dict:
    """정제하기 전에 원본 현황을 파악한다(아직 아무것도 고치지 않는다).

    결측치·중복은 이 데이터에서 실제로 처리할 것이 없지만,
    "확인한 결과 없더라"를 출력으로 남기는 것 자체가 과제 요구사항이다.
    """
    nulls = {k: v for k, v in df.null_count().to_dicts()[0].items() if v > 0}
    n_dup = int(df.is_duplicated().sum())

    dist = (
        df.group_by("payment_type")
        .agg(pl.len().alias("건수"))
        .sort("payment_type")
        .to_dicts()
    )

    # 결측이 payment_type=0에만 몰려 있는지 교차 검증한다.
    # 이것이 무작위 결측이 아니라 'Flex Fare는 미터기 세부 필드가 해당 없음'임을 보이는 근거다.
    # 결측 컬럼 전부를 검사한다 — 한 컬럼만 보고 "전부 확인했다"고 쓰면 문서가 사실과 어긋난다.
    null_in_flex = {
        col: int(
            df.filter(pl.col(col).is_null()).select((pl.col("payment_type") == 0).sum()).item()
        )
        for col in nulls
    }

    return {
        "shape": [df.height, df.width],
        "null_counts": nulls,
        "null_in_flex_fare": null_in_flex,
        "null_all_in_flex_fare": bool(nulls) and all(null_in_flex[c] == nulls[c] for c in nulls),
        "duplicated_rows": n_dup,
        "payment_type_distribution": dist,
    }


def select_target_classes(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """1단계 — 분석 대상 클래스만 남긴다.

    이 작업을 가장 먼저 하는 이유: 학습에 쓰지 않을 3(무료)·4(분쟁)가 섞인 채로
    평균·상관계수를 내면 그 수치가 무엇의 통계인지 애매해진다.
    """
    out = df.filter(pl.col("payment_type").is_in(TARGET_CODES))
    return out, df.height - out.height


def clean(df: pl.DataFrame) -> tuple[pl.DataFrame, list[dict]]:
    """2단계 — 물리적으로 불가능한 기록을 제거하고, 규칙별 해당 건수를 기록한다.

    "이상치 제거함"이 아니라 "어떤 규칙으로 몇 건"을 남겨야
    나중에 결과가 이상할 때 어느 필터가 원인인지 추적할 수 있다.

    소요시간은 필터 조건이면서 동시에 파생변수이므로 여기서 먼저 계산한다.
    (거꾸로 파생변수를 나중에 만들면 하차<승차인 행에서 음수 소요시간이 생겨 다시 걸러야 한다)
    """
    df = df.with_columns(
        (
            (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_seconds()
            / 60
        ).alias("trip_duration_min")
    )

    # 규칙별 '해당 건수'. 조건끼리 겹칠 수 있으므로 합계 ≠ 실제 제거 건수다.
    #
    # 각 규칙은 아래 keep 조건의 정확한 부정(negation)이어야 한다.
    # 세는 기준과 지우는 기준이 어긋나면 경계값 행이 제거되고도 어느 규칙에도 잡히지 않는다.
    rules = [
        ("요금 ≤ 0", pl.col("fare_amount") <= MIN_FARE),
        ("이동거리 ≤ 0", pl.col("trip_distance") <= 0),
        (f"이동거리 ≥ {MAX_DISTANCE:.0f}마일", pl.col("trip_distance") >= MAX_DISTANCE),
        ("소요시간 ≤ 0분", pl.col("trip_duration_min") <= 0),
        (f"소요시간 ≥ {MAX_DURATION_MIN:.0f}분", pl.col("trip_duration_min") >= MAX_DURATION_MIN),
    ]

    n_before = df.height
    rule_counts = []
    for name, cond in rules:
        hit = int(df.select(cond.sum()).item())
        rule_counts.append(
            {"규칙": name, "해당건수": hit, "비율%": round(hit / n_before * 100, 3)}
        )

    keep = (
        (pl.col("fare_amount") > MIN_FARE)
        & (pl.col("trip_distance") > 0)
        & (pl.col("trip_distance") < MAX_DISTANCE)
        & (pl.col("trip_duration_min") > 0)
        & (pl.col("trip_duration_min") < MAX_DURATION_MIN)
    )
    return df.filter(keep), rule_counts


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """3단계 — 예측에 쓸 파생변수와 사람이 읽을 라벨을 만든다.

    원본에는 승·하차 시각만 있어 '몇 시에 탔는가', '주말인가' 같은 패턴을 볼 수 없다.
    trip_duration_min은 필터에 필요해 clean()에서 이미 생성했다.
    """
    return df.with_columns(
        pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
        # Polars의 weekday는 월요일=1 … 일요일=7
        pl.col("tpep_pickup_datetime").dt.weekday().alias("day_of_week"),
        (pl.col("tpep_pickup_datetime").dt.weekday() >= 6).alias("is_weekend"),
        pl.col("payment_type")
        .replace_strict(PAYMENT_LABELS, return_dtype=pl.String)
        .alias("payment_label"),
    )


def class_distribution(df: pl.DataFrame) -> list[dict]:
    """정제가 제대로 됐는지 확인하기 위한 최소 집계.

    본격적인 분포 해석과 베이스라인 산정은 Step 3(`03_eda.py`)에서 한다.
    """
    return (
        df.group_by("payment_label")
        .agg(pl.len().alias("건수"))
        .with_columns((pl.col("건수") / df.height * 100).round(2).alias("비율%"))
        .sort("건수", descending=True)
        .to_dicts()
    )


def write_markdown(r: dict) -> None:
    """전처리 결과를 사람이 읽는 마크다운으로 저장한다."""

    def table(rows: list[dict]) -> str:
        if not rows:
            return "(없음)"
        cols = list(rows[0])
        head = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        body = "\n".join(
            "| " + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in row.values()) + " |"
            for row in rows
        )
        return f"{head}\n{sep}\n{body}"

    md = f"""# Step 2 — 전처리 결과

- 원본: {r['raw']['shape'][0]:,}행 × {r['raw']['shape'][1]}컬럼
- 정제 후: **{r['cleaned']['shape'][0]:,}행 × {r['cleaned']['shape'][1]}컬럼** (원본의 {r['retention_pct']}%)

> 이 단계는 데이터를 바꾸는 유일한 단계다. 탐색·통계·시각화는 Step 3·4·5에서 한다.

## 1. 정제 전 현황

중복 행은 **{r['raw']['duplicated_rows']}건**으로 제거할 것이 없다.

결측이 있는 컬럼은 아래 {len(r['raw']['null_counts'])}개이며, 모두 정확히 같은 건수다.

{table([{'컬럼': k, '결측건수': v} for k, v in r['raw']['null_counts'].items()])}

**{len(r['raw']['null_counts'])}개 컬럼 전부에 대해** 결측 행이 모두 `payment_type=0`(Flex Fare)인지
교차 검증한 결과: **{'일치함' if r['raw']['null_all_in_flex_fare'] else '불일치'}.**
따라서 이는 무작위 결측이 아니라, Flex Fare가 앱 기반 사전 확정요금이라
미터기 세부 요금 필드가 **애초에 해당 없음**을 의미한다. 채우거나 지울 대상이 아니다.

## 2. 정제 내역

### 2-1. 분석 대상 선별
`payment_type` 3(무료)·4(분쟁)를 제외했다. 합산 {r['excluded_minor_classes']:,}건으로
표본이 지나치게 적어 신뢰할 수 있는 분류가 불가능하다.

### 2-2. 이상값 제거 (규칙별 해당 건수)

{table(r['rule_counts'])}

> 규칙끼리 겹칠 수 있으므로 위 합계와 실제 제거 건수는 다르다.
> 대상 선별 후 {r['after_select']:,}행 → 이상값 제거 후 **{r['cleaned']['shape'][0]:,}행**

### 2-3. 생성한 파생변수

| 변수 | 내용 | 만든 이유 |
|---|---|---|
| `trip_duration_min` | 하차 − 승차 (분) | 원본에 소요시간이 없다 |
| `pickup_hour` | 승차 시각의 '시' (0~23) | 시간대별 패턴을 보기 위해 |
| `day_of_week` | 요일 (월=1 … 일=7) | 평일과 주말의 이동 성격이 다르다 |
| `is_weekend` | 주말 여부 | 사람이 읽는 요약용. **모델 피처로는 쓰지 않는다** — `day_of_week`에서 완전히 유도되므로 정보 중복 (Step 3의 `features.json` 참조) |
| `payment_label` | 결제수단 이름 | 0/1/2 코드를 읽을 수 있게 |

## 3. 정제 후 클래스 분포 (검증용)

{table(r['class_distribution'])}

> 여기서는 정제가 제대로 됐는지 확인만 한다. 분포 해석과 베이스라인 산정은 Step 3에서 한다.

## 4. 저장 파일에 남은 결측 — 결함이 아님

`cleaned.parquet`에는 결측이 **{r['remaining_nulls']['total']:,}개** 남아 있다.
이는 Flex Fare {r['remaining_nulls']['flex_fare_rows']:,}행 × {len(r['remaining_nulls']['columns'])}개 컬럼과
정확히 일치하며, 그 외 행의 결측은 0이다.

채우거나 지우지 않고 그대로 두는 이유는 **값이 없는 것이 아니라 항목 자체가 해당 없기 때문**이다.
평균으로 채우면 존재하지 않는 미터기 요금을 지어내는 셈이 된다.

### 모델 피처에서 제외할 누수 컬럼 {len(r['leakage_cols'])}개

`{'`, `'.join(r['leakage_cols'])}`

앞의 두 개는 팁 정보라 결제수단을 그대로 노출하고, 나머지는 Flex Fare에서만 비어 있어
"비어 있으면 Flex Fare"라는 규칙을 외우게 만든다.
**Step 3이 이를 근거로 피처 화이트리스트(`features.json`)를 확정하고, Step 6은 그 목록만 사용한다.**

## 산출물

- `data/processed/cleaned.parquet` — Step 3·4·5·6의 입력
- `outputs/results/preprocess.json` — Step 7의 report.md 생성용 수치
"""
    (RESULT_DIR / "preprocess.md").write_text(md, encoding="utf-8")


def main() -> None:
    for d in (CLEAN_PATH.parent, RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1) 원본 로딩 및 현황 파악 ---
    raw = load_raw()
    print(f"원본 로딩: {raw.height:,}행 × {raw.width}컬럼")
    prof = profile_raw(raw)

    print("\n[정제 전 현황]")
    print(f"  중복 행: {prof['duplicated_rows']}건")
    print(f"  결측 컬럼: {len(prof['null_counts'])}개")
    for col, cnt in prof["null_counts"].items():
        print(f"    {col:<24}{cnt:>10,}건")
    print(f"  결측 행이 전부 payment_type=0(Flex Fare)인가: "
          f"{'예 — 무작위 결측이 아님' if prof['null_all_in_flex_fare'] else '아니오'}")

    print("\n  payment_type 분포:")
    for row in prof["payment_type_distribution"]:
        label = PAYMENT_LABELS.get(row["payment_type"], f"{row['payment_type']} (제외 대상)")
        print(f"    {row['payment_type']} {label:<12}{row['건수']:>10,}건")

    # --- 2) 분석 대상 선별 ---
    df, n_excluded = select_target_classes(raw)
    print(f"\n[1단계] 분석 대상 선별 (0·1·2만): {df.height:,}행  (제외 {n_excluded:,}건)")
    after_select = df.height

    # --- 3) 이상값 제거 ---
    df, rule_counts = clean(df)
    print("\n[2단계] 이상값 제거 — 규칙별 해당 건수 (겹칠 수 있음)")
    for rc in rule_counts:
        print(f"    {rc['규칙']:<20}{rc['해당건수']:>10,}건  ({rc['비율%']}%)")
    print(f"  → {after_select:,}행 → {df.height:,}행")

    # --- 4) 파생변수 생성 ---
    df = add_features(df)
    print("\n[3단계] 파생변수 생성: trip_duration_min, pickup_hour, day_of_week, "
          "is_weekend, payment_label")

    # --- 5) 검증 ---
    codes = sorted(df["payment_type"].unique().to_list())
    assert codes == TARGET_CODES, f"정제 후 payment_type이 {codes} — {TARGET_CODES}여야 함"
    dist = class_distribution(df)

    print(f"\n[4단계] 검증 — payment_type = {codes} 확인")
    print("  클래스 분포 (해석은 Step 3에서):")
    for d in dist:
        print(f"    {d['payment_label']:<12}{d['건수']:>10,}건  ({d['비율%']}%)")

    # --- 6) 저장 (Step 1 결론대로 Pandas로 변환해 저장) ---
    df.to_pandas().to_parquet(CLEAN_PATH, index=False)

    # 저장된 파일에 남은 결측을 설명한다.
    # Flex Fare 행의 5개 컬럼은 '해당 없음'이므로 채우지 않고 그대로 둔다.
    # Step 3·4·6에서 이 결측을 결함으로 오해하지 않도록 근거를 함께 출력한다.
    remaining_nulls = {k: v for k, v in df.null_count().to_dicts()[0].items() if v > 0}
    n_flex = int(df.select((pl.col("payment_type") == 0).sum()).item())
    nulls_explained = all(v == n_flex for v in remaining_nulls.values())
    assert nulls_explained, (
        f"예상치 못한 결측: {remaining_nulls} — Flex Fare 행 수({n_flex:,})와 일치해야 함"
    )
    print(f"\n[저장 파일의 결측] {sum(remaining_nulls.values()):,}개 "
          f"= Flex Fare {n_flex:,}행 × {len(remaining_nulls)}개 컬럼")
    print("  → 결함이 아니라 '해당 없음'. 채우지 않고 두되 모델 피처에서는 제외한다.")

    result = {
        "raw": prof,
        "excluded_minor_classes": n_excluded,
        "after_select": after_select,
        "rule_counts": rule_counts,
        "cleaned": {"shape": [df.height, df.width]},
        "retention_pct": round(df.height / raw.height * 100, 1),
        "class_distribution": dist,
        "leakage_cols": LEAKAGE_COLS,
        "remaining_nulls": {
            "total": sum(remaining_nulls.values()),
            "columns": remaining_nulls,
            "flex_fare_rows": n_flex,
            "explained_as_not_applicable": nulls_explained,
        },
    }
    (RESULT_DIR / "preprocess.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(result)

    print("\n저장 완료:")
    print(f"  {CLEAN_PATH.relative_to(ROOT)}  ({df.height:,}행, 원본의 {result['retention_pct']}%)")
    print(f"  {(RESULT_DIR / 'preprocess.json').relative_to(ROOT)}")
    print(f"  {(RESULT_DIR / 'preprocess.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
