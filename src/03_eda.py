"""
Step 3 — EDA (탐색적 데이터 분석)

`cleaned.parquet`을 **읽기만** 하며 데이터를 바꾸지 않는다. 차트도 그리지 않는다(Step 5 담당).

차트를 먼저 그리지 않는 이유: 그림부터 보면 해석이 그림에 끌려간다.
실제로 이상치 제거 전에는 Flex Fare의 평균 거리가 신용카드의 약 3배로 보였으나,
제거 후에는 11% 차이였다. 숫자를 먼저 확인해야 그림을 올바로 읽을 수 있다.

수행 항목:
  1) 클래스 분포와 베이스라인 성능    5) 지역 패턴 (교차표)
  2) 전체 기술통계                    6) 상관계수 (수치형끼리만)
  3) 결제수단별 기술통계              7) 타깃 누수 진단
  4) 시간 패턴 (시간대·요일)          8) 피처 화이트리스트 확정

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

# 범주형 피처. PULocationID 등은 숫자로 저장돼 있지만 지역·업체 코드이므로
# 연속형으로 취급해 상관계수를 계산하면 통계적으로 무의미하다. 교차표로 본다.
CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID", "pickup_hour", "day_of_week", "VendorID"]

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


def describe_overall(df: pd.DataFrame) -> list[dict]:
    """2) 전체 기술통계 — 평균·표준편차·분위수 (과제 요구사항)."""
    d = df[NUMERIC_FEATURES].describe().T
    return [
        {
            "변수": idx,
            "평균": round(row["mean"], 2),
            "표준편차": round(row["std"], 2),
            "최솟값": round(row["min"], 2),
            "25%": round(row["25%"], 2),
            "중앙값": round(row["50%"], 2),
            "75%": round(row["75%"], 2),
            "최댓값": round(row["max"], 2),
        }
        for idx, row in d.iterrows()
    ]


def describe_by_class(df: pd.DataFrame) -> list[dict]:
    """3) 결제수단별 기술통계 — EDA의 핵심인 그룹 간 비교."""
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


def zone_composition(df: pd.DataFrame) -> list[dict]:
    """5) 지역 패턴 — 픽업 상위 구역의 결제수단 구성비.

    PULocationID는 임의로 부여된 지역 코드이므로 상관계수 대상이 아니다. 교차표로 본다.
    """
    top = df["PULocationID"].value_counts().head(TOP_N_ZONES)
    sub = df[df["PULocationID"].isin(top.index)]
    ct = pd.crosstab(sub["PULocationID"], sub[TARGET], normalize="index") * 100
    return [
        {"PULocationID": int(z), "픽업건수": int(top[z]),
         **{c: round(float(ct.loc[z, c]), 2) for c in ct.columns}}
        for z in top.index
    ]


def correlation(df: pd.DataFrame) -> dict:
    """6) 상관계수 — 수치형 변수끼리만 (Pearson).

    타깃은 3개 범주라 상관계수로 관계를 볼 수 없으므로, 타깃과의 관계는
    describe_by_class()와 composition_by()의 그룹 비교로 대신한다.
    """
    corr = df[NUMERIC_FEATURES].corr().round(3)
    return {
        "columns": NUMERIC_FEATURES,
        "matrix": [
            {"변수": idx, **{c: float(row[c]) for c in corr.columns}}
            for idx, row in corr.iterrows()
        ],
        "max_offdiag": round(float(corr.where(~np.eye(len(corr), dtype=bool)).max().max()), 3),
    }


def leakage_diagnostics(df: pd.DataFrame) -> dict:
    """7) 타깃 누수 진단 — 이 프로젝트의 핵심 발견.

    (a) 팁: 현금 팁은 기록되지 않아 tip_amount가 결제수단을 그대로 노출한다.
    (b) 요금 끝자리: Flex Fare는 사전 확정요금이라 미터기 요금과 소수점 분포가 다르다.
        누수는 아니지만 Flex Fare를 거의 식별하는 강한 신호이므로 함께 기록한다.
    """
    tip = (
        df.assign(팁있음=df["tip_amount"] > 0)
        .groupby(TARGET)["팁있음"]
        .mean()
        .mul(100)
        .round(1)
    )
    # 부동소수점 오차를 피하기 위해 센트 단위 정수로 변환해 비교한다.
    cents = (df["fare_amount"] * 100).round().astype("int64")
    half = df.assign(반달러배수=cents % 50 == 0).groupby(TARGET)["반달러배수"].mean().mul(100).round(1)

    return {
        "tip_positive_pct": [{"결제수단": k, "팁>0 비율%": float(v)} for k, v in tip.items()],
        "fare_half_dollar_pct": [
            {"결제수단": k, "$0.50 배수 비율%": float(v)} for k, v in half.sort_values().items()
        ],
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
        "excluded_leakage": leakage_cols,
        "excluded_reason": {
            "tip_amount": "현금 팁은 미기록 — 결제수단을 그대로 노출",
            "total_amount": "tip_amount를 포함하므로 동일한 누수",
            "congestion_surcharge": "Flex Fare에서만 결측 — 결측 여부가 정답을 노출",
            "Airport_fee": "위와 동일",
            "RatecodeID": "위와 동일",
            "passenger_count": "위와 동일",
            "store_and_fwd_flag": "위와 동일",
        },
    }
    # 누수 컬럼이 화이트리스트에 섞이지 않았는지 검증한다.
    used = set(whitelist["numeric"]) | set(whitelist["categorical"])
    overlap = used & set(leakage_cols)
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
    corr = r["correlation"]

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

## 2. 전체 기술통계

{table(r['describe_overall'])}

## 3. 결제수단별 기술통계

{table(r['describe_by_class'])}

## 4. 시간 패턴

### 시간대별 결제수단 구성비 (%)

{table(r['hourly_composition'])}

### 요일별 결제수단 구성비 (%) — 월=1 … 일=7

{table(r['weekday_composition'])}

## 5. 지역 패턴 — 픽업 상위 {TOP_N_ZONES}개 구역 구성비 (%)

{table(r['zone_composition'])}

> `PULocationID`는 임의로 부여된 지역 코드다. 숫자로 저장돼 있다고 해서 상관계수를 계산하면
> 통계적으로 무의미하므로, 교차표로 확인했다.

## 6. 상관계수 (수치형 변수끼리만)

{table(corr['matrix'])}

대각선을 뺀 최댓값이 **{corr['max_offdiag']}**로, 세 변수가 강하게 묶여 있다.
거리가 길수록 요금과 소요시간이 함께 는다는 자연스러운 관계지만,
**셋을 모두 넣으면 같은 정보가 중복된다**는 뜻이기도 하다.
Step 6에서 다중공선성의 영향을 덜 받는 트리 기반 모델을 쓰는 근거가 된다.

타깃(`{TARGET}`)은 3개 범주라 상관계수로 관계를 볼 수 없다.
타깃과의 관계는 위 3·4·5절의 그룹 비교로 대신했다.

## 7. 타깃 누수 진단 — 이 프로젝트의 핵심 발견

### (a) 팁 지급 비율

{table(r['leakage']['tip_positive_pct'])}

카드 결제만 팁이 기록된다. 즉 `tip_amount`를 피처로 쓰면 모델은
"팁이 있으면 카드"를 외우게 되며, 이는 예측이 아니라 정답을 베끼는 것이다.
`total_amount`도 팁을 포함하므로 같은 이유로 제외한다.

### (b) 요금 끝자리 패턴

{table(r['leakage']['fare_half_dollar_pct'])}

미터기 요금은 정해진 단위로 올라가지만 Flex Fare는 사전에 계산된 값이라 끝자리가 고르지 않다.
**요금의 소수점 패턴만으로 Flex Fare가 상당 부분 구분된다.**
`fare_amount`는 정당한 피처이므로 제외하지 않지만, Step 6에서 Flex Fare 성능이 높게 나온다면
이 패턴에 의존했을 가능성을 함께 검토해야 한다.

## 8. 피처 화이트리스트 (`features.json`)

Step 6은 이 목록만 읽어 피처를 구성한다. 코드에 컬럼명을 직접 적지 않으므로
누수 컬럼이 실수로 섞일 경로가 없다.

| 구분 | 컬럼 |
|---|---|
| 타깃 | `{w['target']}` |
| 수치형 ({len(w['numeric'])}개) | `{'`, `'.join(w['numeric'])}` |
| 범주형 ({len(w['categorical'])}개) | `{'`, `'.join(w['categorical'])}` |
| **제외 (누수 {len(w['excluded_leakage'])}개)** | `{'`, `'.join(w['excluded_leakage'])}` |

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

    # --- 2) 전체 기술통계 ---
    overall = describe_overall(df)
    print("\n[2] 전체 기술통계")
    print(f"    {'변수':<20}{'평균':>10}{'표준편차':>10}{'중앙값':>10}{'75%':>10}")
    for d in overall:
        print(f"    {d['변수']:<20}{d['평균']:>10}{d['표준편차']:>10}{d['중앙값']:>10}{d['75%']:>10}")

    # --- 3) 결제수단별 기술통계 ---
    by_class = describe_by_class(df)
    print("\n[3] 결제수단별 기술통계 (평균)")
    print(f"    {'결제수단':<12}{'거리':>10}{'요금':>10}{'소요시간':>10}")
    for d in by_class:
        print(f"    {d['결제수단']:<12}{d['trip_distance_평균']:>10}"
              f"{d['fare_amount_평균']:>10}{d['trip_duration_min_평균']:>10}")

    # --- 4) 시간 패턴 ---
    hourly = composition_by(df, "pickup_hour")
    weekday = composition_by(df, "day_of_week")
    print(f"\n[4] 시간 패턴 — 시간대 {len(hourly)}개 구간, 요일 {len(weekday)}개 집계")
    cash_by_hour = {h["pickup_hour"]: h.get("현금", 0) for h in hourly}
    hi = max(cash_by_hour, key=cash_by_hour.get)
    lo = min(cash_by_hour, key=cash_by_hour.get)
    print(f"    현금 비율이 가장 높은 시각: {hi}시 ({cash_by_hour[hi]}%) / "
          f"가장 낮은 시각: {lo}시 ({cash_by_hour[lo]}%)")

    # --- 5) 지역 패턴 ---
    zones = zone_composition(df)
    print(f"\n[5] 지역 패턴 — 픽업 상위 {TOP_N_ZONES}개 구역")
    print(f"    {'구역':<8}{'픽업건수':>12}{'신용카드':>10}{'Flex Fare':>12}{'현금':>8}")
    for z in zones[:5]:
        print(f"    {z['PULocationID']:<8}{z['픽업건수']:>12,}{z.get('신용카드', 0):>10}"
              f"{z.get('Flex Fare', 0):>12}{z.get('현금', 0):>8}")
    print(f"    ... (전체 {len(zones)}개는 eda.md 참조)")

    # --- 6) 상관계수 ---
    corr = correlation(df)
    print("\n[6] 상관계수 (수치형끼리만)")
    for row in corr["matrix"]:
        vals = "  ".join(f"{row[c]:>6}" for c in corr["columns"])
        print(f"    {row['변수']:<20}{vals}")
    print(f"    대각선 제외 최댓값: {corr['max_offdiag']} → 정보 중복, 트리 기반 모델 사용 근거")

    # --- 7) 누수 진단 ---
    leak = leakage_diagnostics(df)
    print("\n[7] 타깃 누수 진단")
    print("    팁 > 0 비율:")
    for d in leak["tip_positive_pct"]:
        print(f"      {d['결제수단']:<12}{d['팁>0 비율%']:>8}%")
    print("    요금이 $0.50 배수인 비율:")
    for d in leak["fare_half_dollar_pct"]:
        print(f"      {d['결제수단']:<12}{d['$0.50 배수 비율%']:>8}%")

    # --- 8) 피처 화이트리스트 ---
    whitelist = build_feature_whitelist(leakage_cols)
    print(f"\n[8] 피처 화이트리스트 확정 — 수치형 {len(whitelist['numeric'])}개 + "
          f"범주형 {len(whitelist['categorical'])}개, 누수 {len(leakage_cols)}개 제외")
    print("    검증: 화이트리스트에 누수 컬럼 없음 (assert 통과)")

    result = {
        "n_rows": len(df),
        "class_and_baseline": cls,
        "describe_overall": overall,
        "describe_by_class": by_class,
        "hourly_composition": hourly,
        "weekday_composition": weekday,
        "zone_composition": zones,
        "correlation": corr,
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
