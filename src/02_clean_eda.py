"""
Step 2 — 데이터 정제 + EDA + 시각화

원본 409만 행을 분석 가능한 형태로 정제하고, 결제수단별 특성을 탐색한다.

정제 순서(이 순서를 지켜야 하는 이유는 각 함수 주석 참조):
  1) 분석 대상 클래스 선별  — payment_type ∈ {0(Flex Fare), 1(신용카드), 2(현금)}
  2) 이상값 제거            — 규칙별 제거 건수를 각각 기록
  3) 파생변수 생성          — 소요시간·시간대·요일·주말 여부·라벨
  4) 검증 후 저장

Step 1의 결론에 따라 무거운 처리는 Polars로 수행하고, 저장 직전에 Pandas로 변환한다.
(seaborn·scipy·scikit-learn이 Pandas 객체를 입력으로 받으므로)

산출물:
  - data/processed/cleaned.parquet        정제 결과 (Step 3·4의 입력)
  - outputs/figures/seaborn_*.png         Seaborn 정적 차트
  - outputs/figures/plotly_*.html         Plotly 인터랙티브 차트
  - outputs/results/eda.json              수치 결과 (Step 5의 report.md 생성용)
  - outputs/results/eda.md                사람이 읽는 요약
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # GUI 없이 파일로만 저장

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.express as px
import polars as pl
import seaborn as sns

# --- 경로 상수 ---
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "yellow_tripdata_2026-05.parquet"
CLEAN_PATH = ROOT / "data" / "processed" / "cleaned.parquet"
FIG_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"

# --- 분석 설정 ---
# payment_type 코드 → 라벨. 3(무료)·4(분쟁)는 합산 0.85%로 학습이 불가능해 제외한다.
PAYMENT_LABELS = {0: "Flex Fare", 1: "신용카드", 2: "현금"}
TARGET_CODES = list(PAYMENT_LABELS)

# 타깃 누수 컬럼 — 정제 결과 파일에는 남기되(누수 근거를 Step 3에서 제시해야 하므로),
# Step 4의 모델 피처로는 절대 사용하지 않는다.
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
MIN_FARE = 0.0
MAX_DISTANCE = 200.0
MAX_DURATION_MIN = 180.0

PLOT_SAMPLE_N = 200_000  # 차트용 표본 (389만 행 전부 그리면 느리고 가독성도 나쁘다)
RANDOM_STATE = 42


def setup_korean_font() -> str:
    """차트의 한글이 깨지지 않도록 폰트를 설정한다.

    OS마다 설치된 한글 폰트가 다르므로 후보를 순서대로 시도한다.
    (macOS: AppleGothic / Windows: Malgun Gothic / Linux: NanumGothic)
    """
    installed = {f.name for f in fm.fontManager.ttflist}
    for candidate in ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]:
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
            return candidate
    print("[경고] 한글 폰트를 찾지 못했습니다. 차트의 한글이 깨질 수 있습니다.")
    return "(없음)"


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
    null_col = next(iter(nulls), None)
    null_in_flex = (
        int(df.filter(pl.col(null_col).is_null()).select((pl.col("payment_type") == 0).sum()).item())
        if null_col
        else 0
    )
    n_null_rows = nulls.get(null_col, 0)

    return {
        "shape": [df.height, df.width],
        "null_counts": nulls,
        "null_all_in_flex_fare": bool(n_null_rows and null_in_flex == n_null_rows),
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
    """2단계 — 물리적으로 불가능한 기록을 제거하고, 규칙별 제거 건수를 기록한다.

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
    rules = [
        ("요금 ≤ 0", pl.col("fare_amount") <= MIN_FARE),
        ("이동거리 ≤ 0", pl.col("trip_distance") <= 0),
        (f"이동거리 > {MAX_DISTANCE:.0f}마일", pl.col("trip_distance") > MAX_DISTANCE),
        ("소요시간 ≤ 0분", pl.col("trip_duration_min") <= 0),
        (f"소요시간 > {MAX_DURATION_MIN:.0f}분", pl.col("trip_duration_min") > MAX_DURATION_MIN),
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
    """결제수단별 건수와 비율을 구한다."""
    return (
        df.group_by("payment_label")
        .agg(pl.len().alias("건수"))
        .with_columns((pl.col("건수") / df.height * 100).round(2).alias("비율%"))
        .sort("건수", descending=True)
        .to_dicts()
    )


def describe_by_payment(df: pl.DataFrame) -> list[dict]:
    """결제수단별 기술통계 — EDA의 핵심인 '그룹 간 비교'."""
    return (
        df.group_by("payment_label")
        .agg(
            pl.len().alias("건수"),
            pl.col("trip_distance").mean().round(2).alias("거리평균"),
            pl.col("trip_distance").std().round(2).alias("거리표준편차"),
            pl.col("fare_amount").mean().round(2).alias("요금평균"),
            pl.col("fare_amount").std().round(2).alias("요금표준편차"),
            pl.col("trip_duration_min").mean().round(2).alias("소요시간평균"),
            pl.col("trip_duration_min").std().round(2).alias("소요시간표준편차"),
        )
        .sort("건수", descending=True)
        .to_dicts()
    )


def fare_ending_pattern(df: pl.DataFrame) -> list[dict]:
    """Flex Fare 판별 신호 검증 — 요금 끝자리가 $0.50 배수인 비율.

    Flex Fare는 사전 확정요금이라 미터기 요금과 소수점 분포가 다르다.
    부동소수점 오차를 피하기 위해 센트 단위 정수로 변환해 비교한다.
    """
    return (
        df.with_columns(
            ((pl.col("fare_amount") * 100).round(0).cast(pl.Int64) % 50 == 0).alias("is_half")
        )
        .group_by("payment_label")
        .agg((pl.col("is_half").mean() * 100).round(1).alias("$0.50배수비율%"))
        .sort("$0.50배수비율%")
        .to_dicts()
    )


def make_seaborn_chart(df: pl.DataFrame) -> Path:
    """Seaborn 정적 차트 — 결제수단별 이동거리 분포 (그룹 비교).

    극단값 때문에 상자가 눌려 보이지 않도록 이상치 표시를 끄고 y축을 제한한다.
    """
    sample = df.sample(n=min(PLOT_SAMPLE_N, df.height), seed=RANDOM_STATE).to_pandas()
    order = sample["payment_label"].value_counts().index.tolist()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=sample, x="payment_label", y="trip_distance", order=order,
                showfliers=False, ax=ax)
    ax.set_title("결제수단별 이동거리 분포", fontsize=15, pad=14)
    ax.set_xlabel("결제수단")
    ax.set_ylabel("이동거리 (mile)")
    ax.set_ylim(0, 12)
    fig.text(0.99, 0.01, f"표본 {len(sample):,}건 · 이상치 표시 제외",
             ha="right", fontsize=9, color="gray")
    fig.tight_layout()

    path = FIG_DIR / "seaborn_distance_by_payment.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_plotly_chart(df: pl.DataFrame) -> Path:
    """Plotly 인터랙티브 차트 — 시간대별 결제수단 구성비.

    건수가 아니라 비율로 그린다. 시간대마다 전체 운행량이 크게 다르므로
    건수로 그리면 '새벽엔 현금이 적다'가 아니라 '새벽엔 운행이 적다'만 보인다.
    """
    hourly = (
        df.group_by(["pickup_hour", "payment_label"])
        .agg(pl.len().alias("건수"))
        .with_columns(
            (pl.col("건수") / pl.col("건수").sum().over("pickup_hour") * 100)
            .round(2)
            .alias("비율")
        )
        .sort(["pickup_hour", "payment_label"])
        .to_pandas()
    )

    fig = px.bar(
        hourly, x="pickup_hour", y="비율", color="payment_label",
        title="시간대별 결제수단 구성비",
        labels={"pickup_hour": "승차 시각 (시)", "비율": "구성비 (%)", "payment_label": "결제수단"},
        hover_data={"건수": ":,"},
    )
    fig.update_layout(barmode="stack", xaxis=dict(dtick=1), yaxis_range=[0, 100],
                      font=dict(family="AppleGothic, Malgun Gothic, sans-serif"))

    path = FIG_DIR / "plotly_hourly_payment_mix.html"
    fig.write_html(path)
    return path


def write_markdown(r: dict) -> None:
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

    md = f"""# Step 2 — 정제 및 EDA 결과

- 원본: {r['raw']['shape'][0]:,}행 × {r['raw']['shape'][1]}컬럼
- 정제 후: **{r['cleaned']['shape'][0]:,}행** (원본의 {r['retention_pct']}%)

## 1. 정제 전 현황

중복 행은 **{r['raw']['duplicated_rows']}건**으로 제거할 것이 없다.

결측이 있는 컬럼은 아래 5개이며, 모두 정확히 같은 건수다.

{table([{'컬럼': k, '결측건수': v} for k, v in r['raw']['null_counts'].items()])}

**이 결측 행이 전부 `payment_type=0`(Flex Fare)인지 교차 검증한 결과: {'일치함' if r['raw']['null_all_in_flex_fare'] else '불일치'}.**
따라서 이는 무작위 결측이 아니라, Flex Fare가 앱 기반 사전 확정요금이라
미터기 세부 요금 필드가 **애초에 해당 없음**을 의미한다. 채우거나 지울 대상이 아니다.
다만 이 컬럼들은 Flex Fare에서만 비어 있어 정답을 노출하므로 **모델 피처에서는 제외**한다.

## 2. 정제 내역

### 2-1. 분석 대상 선별
`payment_type` 3(무료)·4(분쟁)를 제외했다. 합산 {r['excluded_minor_classes']:,}건으로
표본이 지나치게 적어 신뢰할 수 있는 분류가 불가능하다.

### 2-2. 이상값 제거 (규칙별 해당 건수)

{table(r['rule_counts'])}

> 규칙끼리 겹칠 수 있으므로 위 합계와 실제 제거 건수는 다르다.
> 대상 선별 후 {r['after_select']:,}행 → 이상값 제거 후 **{r['cleaned']['shape'][0]:,}행**

### 2-3. 생성한 파생변수
`trip_duration_min`(소요시간, 분), `pickup_hour`(승차 시각), `day_of_week`(요일, 월=1),
`is_weekend`(주말 여부), `payment_label`(결제수단 이름)

## 3. 정제 후 클래스 분포

{table(r['class_distribution'])}

최소 클래스가 9% 수준이므로 `class_weight="balanced"`로 다룰 수 있다.
다만 최빈 클래스만 예측하는 베이스라인의 정확도가 약 {r['majority_baseline_acc']}이므로,
**정확도만으로 모델을 평가해서는 안 되며 macro F1을 함께 봐야 한다.**

## 4. 결제수단별 기술통계

{table(r['describe_by_payment'])}

## 5. Flex Fare 판별 신호 — 요금 끝자리 패턴

{table(r['fare_ending_pattern'])}

미터기 요금은 정해진 단위로 올라가지만 Flex Fare는 사전에 계산된 값이라 끝자리가 고르지 않다.
**요금의 소수점 패턴만으로 Flex Fare가 상당 부분 구분된다**는 뜻이다.
누수는 아니지만(요금은 정당한 피처), Step 4에서 Flex Fare 분류 성능이 높게 나온다면
이 패턴에 의존했을 가능성을 함께 검토해야 한다.

## 6. 시각화

| 차트 | 파일 | 내용 |
|---|---|---|
| Seaborn (정적) | `{r['figures']['seaborn']}` | 결제수단별 이동거리 분포 (boxplot) |
| Plotly (인터랙티브) | `{r['figures']['plotly']}` | 시간대별 결제수단 구성비 (stacked bar) |

Plotly 차트를 비율로 그린 이유: 시간대마다 전체 운행량이 크게 다르므로 건수로 그리면
"새벽에 현금이 적다"가 아니라 "새벽에 운행이 적다"만 보이게 된다.

## 7. 저장 파일에 남은 결측 — 결함이 아님

`cleaned.parquet`에는 결측이 **{r['remaining_nulls']['total']:,}개** 남아 있다.
이는 Flex Fare {r['remaining_nulls']['flex_fare_rows']:,}행 × {len(r['remaining_nulls']['columns'])}개 컬럼과
정확히 일치하며, 그 외 행의 결측은 0이다.

채우거나 지우지 않고 그대로 두는 이유는 **값이 없는 것이 아니라 항목 자체가 해당 없기 때문**이다.
평균으로 채우면 존재하지 않는 미터기 요금을 지어내는 셈이 된다.

### 모델 피처에서 제외할 누수 컬럼 {len(r['leakage_cols'])}개

`{'`, `'.join(r['leakage_cols'])}`

앞의 두 개는 팁 정보라 결제수단을 그대로 노출하고, 나머지는 Flex Fare에서만 비어 있어
"비어 있으면 Flex Fare"라는 규칙을 외우게 만든다. **Step 4는 반드시 피처 화이트리스트 방식으로
구성해 이 컬럼들이 실수로 포함되지 않게 할 것.**

## 산출물

- `data/processed/cleaned.parquet` — Step 3·4의 입력
- `outputs/results/eda.json` — Step 5의 report.md 생성용 수치
"""
    (RESULT_DIR / "eda.md").write_text(md, encoding="utf-8")


def main() -> None:
    for d in (CLEAN_PATH.parent, FIG_DIR, RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    font = setup_korean_font()
    print(f"한글 폰트: {font}\n")

    # --- 1) 원본 로딩 및 현황 파악 ---
    raw = load_raw()
    print(f"원본 로딩: {raw.height:,}행 × {raw.width}컬럼")
    prof = profile_raw(raw)

    print(f"\n[정제 전 현황]")
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
    print(f"\n[2단계] 이상값 제거 — 규칙별 해당 건수 (겹칠 수 있음)")
    for rc in rule_counts:
        print(f"    {rc['규칙']:<20}{rc['해당건수']:>10,}건  ({rc['비율%']}%)")
    print(f"  → {after_select:,}행 → {df.height:,}행")

    # --- 4) 파생변수 생성 ---
    df = add_features(df)
    print(f"\n[3단계] 파생변수 생성: trip_duration_min, pickup_hour, day_of_week, "
          f"is_weekend, payment_label")

    # --- 5) 검증 ---
    codes = sorted(df["payment_type"].unique().to_list())
    assert codes == TARGET_CODES, f"정제 후 payment_type이 {codes} — {TARGET_CODES}여야 함"
    dist = class_distribution(df)
    majority_acc = round(max(d["비율%"] for d in dist) / 100, 4)

    print(f"\n[4단계] 검증 — payment_type = {codes} 확인")
    print("  클래스 분포:")
    for d in dist:
        print(f"    {d['payment_label']:<12}{d['건수']:>10,}건  ({d['비율%']}%)")
    print(f"  최빈 클래스만 예측하는 베이스라인 정확도: {majority_acc}")

    # --- 6) EDA ---
    desc = describe_by_payment(df)
    print("\n[결제수단별 기술통계]")
    print(f"  {'결제수단':<12}{'거리평균':>10}{'요금평균':>10}{'소요시간평균':>14}")
    for d in desc:
        print(f"  {d['payment_label']:<12}{d['거리평균']:>10}{d['요금평균']:>10}{d['소요시간평균']:>14}")

    endings = fare_ending_pattern(df)
    print("\n[요금 끝자리가 $0.50 배수인 비율]")
    for e in endings:
        print(f"  {e['payment_label']:<12}{e['$0.50배수비율%']:>8}%")

    # --- 7) 시각화 ---
    seaborn_path = make_seaborn_chart(df)
    plotly_path = make_plotly_chart(df)
    print(f"\n[시각화]\n  {seaborn_path.relative_to(ROOT)}\n  {plotly_path.relative_to(ROOT)}")

    # --- 8) 저장 (Step 1 결론대로 Pandas로 변환해 저장) ---
    df.to_pandas().to_parquet(CLEAN_PATH, index=False)

    # 저장된 파일에 남은 결측을 설명한다.
    # Flex Fare 행의 5개 컬럼은 '해당 없음'이므로 채우지 않고 그대로 둔다.
    # Step 3·4에서 이 결측을 결함으로 오해하지 않도록 근거를 함께 출력한다.
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
        "majority_baseline_acc": majority_acc,
        "describe_by_payment": desc,
        "fare_ending_pattern": endings,
        "figures": {
            "seaborn": str(seaborn_path.relative_to(ROOT)),
            "plotly": str(plotly_path.relative_to(ROOT)),
        },
        "leakage_cols": LEAKAGE_COLS,
        "remaining_nulls": {
            "total": sum(remaining_nulls.values()),
            "columns": remaining_nulls,
            "flex_fare_rows": n_flex,
            "explained_as_not_applicable": nulls_explained,
        },
    }
    (RESULT_DIR / "eda.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(result)

    print(f"\n저장 완료:")
    print(f"  {CLEAN_PATH.relative_to(ROOT)}  ({df.height:,}행, 원본의 {result['retention_pct']}%)")
    print(f"  {(RESULT_DIR / 'eda.json').relative_to(ROOT)}")
    print(f"  {(RESULT_DIR / 'eda.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
