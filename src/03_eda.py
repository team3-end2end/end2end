"""
Step 3 — EDA (탐색적 데이터 분석)

`cleaned.parquet`을 **읽기만** 하며 데이터를 바꾸지 않는다. 차트도 그리지 않는다(Step 5 담당).

차트를 먼저 그리지 않는 이유: 그림부터 보면 해석이 그림에 끌려간다.
실제로 이상치 제거 전에는 Flex Fare의 평균 거리가 신용카드의 약 3배로 보였으나,
제거 후에는 11% 차이였다. 숫자를 먼저 확인해야 그림을 올바로 읽을 수 있다.

수행 항목:
  1) 클래스 분포와 베이스라인 성능    4) 지역 패턴 (교차표)
  2) 결제수단별 기술통계              5) 타깃 누수 진단
  3) 시간 패턴 (시간대·요일)          6) 피처 화이트리스트 확정

전체 기술통계(2-3 담당)와 상관계수(2-2 담당)는 이 파일에서 다루지 않는다.
한때 여기 있었으나 분업 경계를 넘은 것이어서 원 담당자에게 반환했다(PR #1 리뷰).

산출물:
  - outputs/results/eda.json / eda.md
  - outputs/results/features.json   Step 6이 읽어 쓸 피처 목록 (누수 차단 장치)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- 경로 상수 ---
ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "processed" / "cleaned.parquet"
RESULT_DIR = ROOT / "outputs" / "results"
PREPROCESS_JSON = RESULT_DIR / "preprocess.json"

# --- 분석 설정 ---
# 누수 위험이 없는 수치형 피처. 상관분석 대상이기도 하다.
NUMERIC_FEATURES = ["trip_distance", "fare_amount", "trip_duration_min"]

# 범주형 피처. PULocationID 등은 숫자로 저장돼 있지만 지역 코드이므로
# 연속형으로 취급해 상관계수를 계산하면 통계적으로 무의미하다. 교차표로 본다.
# 여기 들어가는 컬럼은 반드시 main()에서 타깃과 교차표로 검증한 뒤 확정한다.
CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID", "pickup_hour", "day_of_week"]

# 검증 결과 누수로 판정되어 제외한 범주형 컬럼.
CATEGORICAL_LEAKAGE = {
    "VendorID": "VendorID=6인 6,614행이 예외 없이 전부 Flex Fare — 원핫 시 정답을 그대로 노출",
}

TARGET = "payment_label"
TOP_N_ZONES = 10


def load_cleaned() -> pd.DataFrame:
    """Step 2의 산출물을 읽는다. 없으면 명확한 에러로 멈춘다."""
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"정제 데이터가 없습니다: {CLEAN_PATH}\n먼저 `python src/02_preprocess.py`를 실행하세요."
        )
    return pd.read_parquet(CLEAN_PATH)


def load_leakage_cols() -> list[str]:
    """누수 컬럼 목록을 Step 2의 산출물에서 읽는다.

    코드에 다시 적지 않고 파일에서 읽는 이유: 두 곳에 적으면 한쪽만 수정되어
    서로 다른 목록을 쓰게 된다. preprocess.json이 단일 기준이다.
    """
    if not PREPROCESS_JSON.exists():
        raise FileNotFoundError(
            f"{PREPROCESS_JSON}가 없습니다. 먼저 `python src/02_preprocess.py`를 실행하세요."
        )
    return json.loads(PREPROCESS_JSON.read_text(encoding="utf-8"))["leakage_cols"]


def class_distribution_and_baseline(df: pd.DataFrame) -> dict:
    """1) 클래스 분포와 베이스라인 성능.

    '최빈 클래스만 찍는' 모델의 성능을 미리 계산해둔다.
    Step 6의 모델이 이 수치를 넘지 못하면 학습한 의미가 없으므로, 판단 기준선이 된다.
    """
    counts = df[TARGET].value_counts()
    ratios = (counts / len(df) * 100).round(2)

    # 최빈 클래스만 예측할 때: 그 클래스는 recall=1, precision=비율 → F1 계산 가능.
    # 나머지 클래스는 하나도 맞히지 못하므로 F1=0. macro F1은 세 값의 평균이다.
    p = counts.max() / len(df)
    majority_f1 = 2 * p * 1.0 / (p + 1.0)
    macro_f1 = majority_f1 / df[TARGET].nunique()

    return {
        "distribution": [
            {"결제수단": k, "건수": int(counts[k]), "비율%": float(ratios[k])} for k in counts.index
        ],
        "n_classes": int(df[TARGET].nunique()),
        "baseline_majority_class": str(counts.idxmax()),
        "baseline_accuracy": round(float(p), 4),
        "baseline_macro_f1": round(float(macro_f1), 4),
    }


def describe_by_class(df: pd.DataFrame) -> list[dict]:
    """2) 결제수단별 기술통계 — EDA의 핵심인 그룹 간 비교."""
    g = df.groupby(TARGET)
    out = []
    for label, sub in g:
        row = {"결제수단": label, "건수": len(sub)}
        for col in NUMERIC_FEATURES:
            row[f"{col}_평균"] = round(float(sub[col].mean()), 2)
            row[f"{col}_표준편차"] = round(float(sub[col].std()), 2)
        out.append(row)
    return sorted(out, key=lambda r: -r["건수"])


def composition_by(df: pd.DataFrame, col: str) -> list[dict]:
    """그룹별 결제수단 구성비(%)를 구한다.

    건수가 아니라 비율로 보는 이유: 시간대·요일마다 전체 운행량이 다르므로
    건수로 보면 '새벽에 현금이 적다'가 아니라 '새벽에 운행이 적다'만 드러난다.
    """
    ct = pd.crosstab(df[col], df[TARGET], normalize="index") * 100
    return [
        {col: (int(idx) if isinstance(idx, (int, np.integer)) else idx),
         **{c: round(float(row[c]), 2) for c in ct.columns}}
        for idx, row in ct.iterrows()
    ]


def zone_composition(df: pd.DataFrame, col: str = "PULocationID") -> list[dict]:
    """5) 지역 패턴 — 상위 구역의 결제수단 구성비.

    지역 ID는 임의로 부여된 코드이므로 상관계수 대상이 아니다. 교차표로 본다.
    """
    top = df[col].value_counts().head(TOP_N_ZONES)
    sub = df[df[col].isin(top.index)]
    ct = pd.crosstab(sub[col], sub[TARGET], normalize="index") * 100
    return [
        {col: int(z), "건수": int(top[z]),
         **{c: round(float(ct.loc[z, c]), 2) for c in ct.columns}}
        for z in top.index
    ]


def leakage_diagnostics(df: pd.DataFrame) -> dict:
    """5) 타깃 누수 진단 — 이 프로젝트의 핵심 발견.

    (a) 팁: 현금은 구조적으로 팁이 기록되지 않아 tip_amount가 결제수단을 노출한다.
    (b) 요금 끝자리: Flex Fare는 사전 확정요금이라 미터기 요금과 소수점 분포가 다르다.
        누수는 아니지만 Flex Fare를 거의 식별하는 강한 신호이므로 함께 기록한다.
    (c) VendorID: 기록 경로를 나타내는 변수라 특정 값이 결제수단을 그대로 노출한다.
    """
    # 소수 1자리로 반올림하면 현금의 0.01%가 0.0%로 보여 "현금은 팁이 전혀 없다"는
    # 잘못된 서술을 유도한다. 건수를 함께 남기고 2자리까지 기록한다.
    has_tip = df["tip_amount"] > 0
    tip_n = has_tip.groupby(df[TARGET]).sum()
    tip = has_tip.groupby(df[TARGET]).mean().mul(100).round(2)

    # 부동소수점 오차를 피하기 위해 센트 단위 정수로 변환해 비교한다.
    cents = (df["fare_amount"] * 100).round().astype("int64")
    half = df.assign(반달러배수=cents % 50 == 0).groupby(TARGET)["반달러배수"].mean().mul(100).round(2)

    return {
        "tip_positive_pct": [
            {"결제수단": k, "팁>0 비율%": float(v), "팁>0 건수": int(tip_n[k])}
            for k, v in tip.items()
        ],
        "fare_half_dollar_pct": [
            {"결제수단": k, "$0.50 배수 비율%": float(v)} for k, v in half.sort_values().items()
        ],
        # 건수가 필요하므로 composition_by 대신 zone_composition을 쓴다.
        # VendorID는 값이 3개뿐이라 상위 N개 제한에 걸리지 않는다.
        "vendor_composition": zone_composition(df, "VendorID"),
    }


def build_feature_whitelist(leakage_cols: list[str]) -> dict:
    """8) 피처 화이트리스트 확정.

    Step 6은 이 파일만 읽어 피처를 구성한다. 코드에 컬럼명을 직접 적지 않으면
    누수 컬럼이 실수로 섞일 경로 자체가 사라진다.
    """
    whitelist = {
        "target": TARGET,
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "excluded_leakage": leakage_cols + list(CATEGORICAL_LEAKAGE),
        "excluded_reason": {
            "tip_amount": "현금 팁은 미기록 — 결제수단을 그대로 노출",
            "total_amount": "tip_amount를 포함하므로 동일한 누수",
            "congestion_surcharge": "Flex Fare에서만 결측 — 결측 여부가 정답을 노출",
            "Airport_fee": "위와 동일",
            "RatecodeID": "위와 동일",
            "passenger_count": "위와 동일",
            "store_and_fwd_flag": "위와 동일",
            **CATEGORICAL_LEAKAGE,
            "is_weekend": "day_of_week에서 완전히 유도되므로 정보 중복 (요일별 고유값 1개)",
        },
    }
    # 누수 컬럼이 화이트리스트에 섞이지 않았는지 검증한다.
    used = set(whitelist["numeric"]) | set(whitelist["categorical"])
    overlap = used & set(whitelist["excluded_leakage"])
    assert not overlap, f"누수 컬럼이 피처에 포함됨: {overlap}"
    return whitelist


def write_markdown(r: dict, w: dict) -> None:
    """EDA 결과를 사람이 읽는 마크다운으로 저장한다."""

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

    cls = r["class_and_baseline"]

    md = f"""# Step 3 — EDA 결과

- 대상: `data/processed/cleaned.parquet` ({r['n_rows']:,}행)
- 이 단계는 데이터를 **읽기만** 한다. 차트는 Step 5에서 그린다.

## 1. 클래스 분포와 베이스라인

{table(cls['distribution'])}

**베이스라인**: 무조건 최빈 클래스(`{cls['baseline_majority_class']}`)만 예측하는 모델의
정확도는 **{cls['baseline_accuracy']}**, macro F1은 **{cls['baseline_macro_f1']}**이다.

Step 6의 모델은 이 두 수치와 비교해 평가한다. 특히 **정확도만 보면 안 된다** —
아무것도 학습하지 않아도 {cls['baseline_accuracy']:.0%}가 나오기 때문이다.
소수 클래스를 살리려 `class_weight="balanced"`를 쓰면 정확도는 오히려 떨어질 수 있으며,
그때 판단 기준은 macro F1이다.

## 2. 결제수단별 기술통계

{table(r['describe_by_class'])}

## 3. 시간 패턴

### 시간대별 결제수단 구성비 (%)

{table(r['hourly_composition'])}

### 요일별 결제수단 구성비 (%) — 월=1 … 일=7

{table(r['weekday_composition'])}

## 4. 지역 패턴 — 상위 {TOP_N_ZONES}개 구역 구성비 (%)

### 승차 구역 (`PULocationID`)

{table(r['zone_composition'])}

### 하차 구역 (`DOLocationID`)

{table(r['dropoff_zone_composition'])}

> 지역 ID는 임의로 부여된 코드다. 숫자로 저장돼 있다고 해서 상관계수를 계산하면
> 통계적으로 무의미하므로, 교차표로 확인했다.
>
> 두 컬럼 모두 구역마다 구성비가 완만하게 달라질 뿐 특정 구역이 결제수단을 확정하지 않는다.
> **정상 신호이므로 피처로 사용한다.** (같은 검사에서 탈락한 `VendorID`는 5절 (c) 참조)

## 5. 타깃 누수 진단 — 이 프로젝트의 핵심 발견

### (a) 팁 지급 비율

{table(r['leakage']['tip_positive_pct'])}

**현금은 구조적으로 팁이 기록되지 않고**(시스템을 거치지 않으므로),
**Flex Fare는 앱에서 별도로 팁이 붙어 9%대가 기록된다.** 즉 "카드만 팁이 있다"가 아니라
"현금만 팁이 없다"가 정확한 서술이다.

어느 쪽이든 `tip_amount`를 피처로 쓰면 모델은 "팁이 0이면 현금"을 외우게 되며,
이는 예측이 아니라 정답을 베끼는 것이다. `total_amount`도 팁을 포함하므로 같은 이유로 제외한다.

덧붙여 **"팁이 있는데 카드가 아니면 Flex Fare"**라는 식별 경로도 함께 생긴다.

### (b) 요금 끝자리 패턴

{table(r['leakage']['fare_half_dollar_pct'])}

미터기 요금은 정해진 단위로 올라가지만 Flex Fare는 사전에 계산된 값이라 끝자리가 고르지 않다.
**요금의 소수점 패턴만으로 Flex Fare가 상당 부분 구분된다.**
`fare_amount`는 정당한 피처이므로 제외하지 않지만, Step 6에서 Flex Fare 성능이 높게 나온다면
이 패턴에 의존했을 가능성을 함께 검토해야 한다.

### (c) `VendorID` — 검증 결과 누수로 판정

{table(r['leakage']['vendor_composition'])}

**`VendorID == 6`인 행은 예외 없이 전부 Flex Fare다.** 원핫 인코딩하면 `VendorID_6` 컬럼이
해당 행의 정답을 그대로 알려준다. 전체의 0.2% 미만이라 지표를 크게 흔들지는 않지만,
누수 차단 설계가 막으려던 바로 그 경로이므로 **범주형 피처에서 제외한다.**

1과 2 사이의 Flex Fare 비율 차이(약 9%p)도 작지 않다. Flex Fare만 5개 컬럼이 100% 결측이라는
사실과 합치면, `VendorID`는 운행 특성이 아니라 **기록 경로**를 나타내는 변수로 보는 것이 자연스럽다.

> 화이트리스트에 들어가는 범주형 컬럼은 이렇게 타깃과 교차표를 그려 확인한 뒤 확정한다.
> "코드에 컬럼명을 적지 않으면 누수가 섞일 경로가 없다"는 설계는
> 화이트리스트 자체가 검증됐을 때만 성립하기 때문이다.

## 6. 피처 화이트리스트 (`features.json`)

Step 6은 이 목록만 읽어 피처를 구성한다. 코드에 컬럼명을 직접 적지 않으므로
누수 컬럼이 실수로 섞일 경로가 없다.

| 구분 | 컬럼 |
|---|---|
| 타깃 | `{w['target']}` |
| 수치형 ({len(w['numeric'])}개) | `{'`, `'.join(w['numeric'])}` |
| 범주형 ({len(w['categorical'])}개) | `{'`, `'.join(w['categorical'])}` |
| **제외 (누수 {len(w['excluded_leakage'])}개)** | `{'`, `'.join(w['excluded_leakage'])}` |

## 이 문서가 다루지 않는 것

- **전체 기술통계**(평균·표준편차·분위수) — 2-3 담당
- **상관계수** — 2-2 담당 (subset·파생변수 추출 근거로 함께 다룰 것)

한때 이 파일에서 함께 생성했으나 분업 경계를 넘은 것이어서 원 담당자에게 반환했다(PR #1 리뷰).
두 항목 모두 `features.json`에는 쓰이지 않으므로 여기서 빠져도 Step 6의 입력은 그대로다.

## 산출물

- `outputs/results/eda.json` — Step 7의 report.md 생성용 수치
- `outputs/results/features.json` — Step 6의 피처 목록
"""
    (RESULT_DIR / "eda.md").write_text(md, encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_cleaned()
    leakage_cols = load_leakage_cols()
    print(f"정제 데이터 로딩: {len(df):,}행 × {df.shape[1]}컬럼\n")

    # --- 1) 클래스 분포와 베이스라인 ---
    cls = class_distribution_and_baseline(df)
    print("[1] 클래스 분포")
    for d in cls["distribution"]:
        print(f"    {d['결제수단']:<12}{d['건수']:>10,}건  ({d['비율%']}%)")
    print(f"  베이스라인(최빈 클래스만 예측): 정확도 {cls['baseline_accuracy']}, "
          f"macro F1 {cls['baseline_macro_f1']}")

    # --- 2) 결제수단별 기술통계 ---
    by_class = describe_by_class(df)
    print("\n[2] 결제수단별 기술통계 (평균)")
    print(f"    {'결제수단':<12}{'거리':>10}{'요금':>10}{'소요시간':>10}")
    for d in by_class:
        print(f"    {d['결제수단']:<12}{d['trip_distance_평균']:>10}"
              f"{d['fare_amount_평균']:>10}{d['trip_duration_min_평균']:>10}")

    # --- 3) 시간 패턴 ---
    hourly = composition_by(df, "pickup_hour")
    weekday = composition_by(df, "day_of_week")
    print(f"\n[3] 시간 패턴 — 시간대 {len(hourly)}개 구간, 요일 {len(weekday)}개 집계")
    cash_by_hour = {h["pickup_hour"]: h.get("현금", 0) for h in hourly}
    hi = max(cash_by_hour, key=cash_by_hour.get)
    lo = min(cash_by_hour, key=cash_by_hour.get)
    print(f"    현금 비율이 가장 높은 시각: {hi}시 ({cash_by_hour[hi]}%) / "
          f"가장 낮은 시각: {lo}시 ({cash_by_hour[lo]}%)")

    # --- 4) 지역 패턴 ---
    zones = zone_composition(df, "PULocationID")
    dropoff_zones = zone_composition(df, "DOLocationID")
    print(f"\n[4] 지역 패턴 — 승·하차 상위 {TOP_N_ZONES}개 구역")
    print(f"    {'승차구역':<8}{'건수':>12}{'신용카드':>10}{'Flex Fare':>12}{'현금':>8}")
    for z in zones[:5]:
        print(f"    {z['PULocationID']:<8}{z['건수']:>12,}{z.get('신용카드', 0):>10}"
              f"{z.get('Flex Fare', 0):>12}{z.get('현금', 0):>8}")
    print(f"    ... (전체 {len(zones)}개는 eda.md 참조)")
    print(f"    하차구역 상위 {TOP_N_ZONES}개도 함께 확인 — DOLocationID를 피처로 쓰는 근거")

    # --- 5) 누수 진단 ---
    leak = leakage_diagnostics(df)
    print("\n[5] 타깃 누수 진단")
    print("    팁 > 0 비율:")
    for d in leak["tip_positive_pct"]:
        print(f"      {d['결제수단']:<12}{d['팁>0 비율%']:>8}%  ({d['팁>0 건수']:,}건)")
    print("    요금이 $0.50 배수인 비율:")
    for d in leak["fare_half_dollar_pct"]:
        print(f"      {d['결제수단']:<12}{d['$0.50 배수 비율%']:>8}%")
    print("    VendorID별 결제수단 구성비:")
    for d in leak["vendor_composition"]:
        print(f"      {d['VendorID']:<12}{d['건수']:>10,}건  신용카드 {d.get('신용카드', 0):>6}%  "
              f"Flex Fare {d.get('Flex Fare', 0):>6}%  현금 {d.get('현금', 0):>6}%")

    # --- 6) 피처 화이트리스트 ---
    whitelist = build_feature_whitelist(leakage_cols)
    print(f"\n[6] 피처 화이트리스트 확정 — 수치형 {len(whitelist['numeric'])}개 + "
          f"범주형 {len(whitelist['categorical'])}개, "
          f"누수 {len(whitelist['excluded_leakage'])}개 제외")
    print(f"    범주형 제외: {', '.join(CATEGORICAL_LEAKAGE)} (5절 (c) 참조)")
    print("    검증: 화이트리스트에 누수 컬럼 없음 (assert 통과)")

    result = {
        "n_rows": len(df),
        "class_and_baseline": cls,
        "describe_by_class": by_class,
        "hourly_composition": hourly,
        "weekday_composition": weekday,
        "zone_composition": zones,
        "dropoff_zone_composition": dropoff_zones,
        "leakage": leak,
    }
    (RESULT_DIR / "eda.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "features.json").write_text(
        json.dumps(whitelist, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(result, whitelist)

    print("\n저장 완료:")
    for name in ("eda.json", "features.json", "eda.md"):
        print(f"  {(RESULT_DIR / name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
