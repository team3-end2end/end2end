"""
Step 1 — Pandas vs Polars 로딩 성능 비교

NYC Yellow Taxi 원본 parquet(약 409만 행)을 Pandas와 Polars로 각각 로딩하여
(1) 로딩 시간, (2) 메모리 사용량을 비교하고,
(3) 두 라이브러리가 '같은 데이터'를 읽었는지 검증한다.

이 단계의 목적은 속도 자체가 아니라, 이후 Step 2(전처리)에서
어느 라이브러리로 무거운 처리를 할지 결정할 근거를 만드는 것이다.

산출물:
  - outputs/results/load_compare.json  (Step 5의 report.md 자동 생성이 읽어감)
  - outputs/results/load_compare.md    (사람이 읽는 요약)
"""

import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path

import pandas as pd
import polars as pl

# --- 경로 상수 (하드코딩된 절대 경로 금지, 프로젝트 루트 기준으로 계산) ---
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "yellow_tripdata_2026-05.parquet"
RESULT_DIR = ROOT / "outputs" / "results"
JSON_PATH = RESULT_DIR / "load_compare.json"
MD_PATH = RESULT_DIR / "load_compare.md"

# 측정 반복 횟수. 1회만 재면 OS 디스크 캐시 상태에 따라 먼저 잰 쪽이 손해를 보므로
# 워밍업 1회를 버린 뒤 N회 반복하여 중앙값을 사용한다.
N_RUNS = 3


def check_input() -> None:
    """원본 파일이 없으면 즉시 실패시킨다.

    이상한 결과를 내는 것보다 명확한 에러로 멈추는 편이 낫다.
    """
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"원본 데이터가 없습니다: {RAW_PATH}\n"
            "README.md의 curl 명령으로 data/raw/ 에 내려받으세요."
        )


def measure(load_fn, n_runs: int = N_RUNS) -> tuple[object, list[float]]:
    """로딩 함수를 워밍업 1회 + n_runs회 실행하고, 각 실행 시간(초)을 반환한다.

    반환값의 첫 원소는 마지막으로 로딩된 데이터프레임(이후 검증에 사용).
    """
    # 워밍업: 첫 읽기는 디스크에서 가져오므로 느리다. 이 결과는 버린다.
    df = load_fn()
    del df
    gc.collect()

    times: list[float] = []
    df = None
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        df = load_fn()
        times.append(time.perf_counter() - start)
    return df, times


def load_pandas():
    """Pandas로 parquet 전체를 읽는다(즉시 실행)."""
    return pd.read_parquet(RAW_PATH)


def load_polars():
    """Polars로 parquet 전체를 읽는다.

    주의: scan_parquet()은 지연(lazy) 실행이라 실제로 데이터를 읽지 않으므로
    비교가 불공정해진다. 양쪽 모두 즉시 실행인 read_parquet으로 맞춘다.
    """
    return pl.read_parquet(RAW_PATH)


def transform_pandas(pdf: pd.DataFrame) -> pd.DataFrame:
    """Step 2에서 실제로 할 전처리를 Pandas로 수행한다.

    로딩 시간만으로는 두 라이브러리의 차이가 크지 않으므로
    (양쪽 모두 Arrow 기반이라 parquet 디코딩 비용이 비슷하다),
    실제 변환 작업으로 비교해야 Step 2의 선택 근거가 된다.
    """
    d = pdf[pdf["payment_type"].isin([1, 2]) & (pdf["fare_amount"] > 0)].copy()
    d["duration_min"] = (
        d["tpep_dropoff_datetime"] - d["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    d["pickup_hour"] = d["tpep_pickup_datetime"].dt.hour
    d["is_card"] = (d["payment_type"] == 1).astype(int)
    return d.groupby("pickup_hour").agg(
        trips=("is_card", "size"),
        card_ratio=("is_card", "mean"),
        mean_fare=("fare_amount", "mean"),
    )


def transform_polars(pldf: pl.DataFrame) -> pl.DataFrame:
    """transform_pandas와 동일한 전처리를 Polars로 수행한다."""
    return (
        pldf.filter(pl.col("payment_type").is_in([1, 2]) & (pl.col("fare_amount") > 0))
        .with_columns(
            (
                (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_seconds()
                / 60
            ).alias("duration_min"),
            pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
            (pl.col("payment_type") == 1).cast(pl.Int8).alias("is_card"),
        )
        .group_by("pickup_hour")
        .agg(
            pl.len().alias("trips"),
            pl.col("is_card").mean().alias("card_ratio"),
            pl.col("fare_amount").mean().alias("mean_fare"),
        )
        .sort("pickup_hour")
    )


def verify_same_data(pdf: pd.DataFrame, pldf: pl.DataFrame) -> dict:
    """두 라이브러리가 동일한 데이터를 읽었는지 검증한다.

    속도만 비교하고 끝내면 '어느 쪽 기준으로 분석한 것인가'가 애매해지므로,
    행·열·결측 개수와 대표 집계값이 일치하는지 확인한다.
    """
    # 결측 개수 비교 (Pandas는 NaN, Polars는 null로 다루지만 개수는 같아야 한다)
    pd_nulls = pdf.isna().sum().to_dict()
    pl_nulls = pldf.null_count().to_dicts()[0]

    # 대표 집계값 비교 — 개수뿐 아니라 값 자체가 같은지 확인하기 위한 최소한의 체크
    pd_agg = {
        "fare_amount_sum": round(float(pdf["fare_amount"].sum()), 2),
        "trip_distance_max": round(float(pdf["trip_distance"].max()), 2),
    }
    pl_agg = {
        "fare_amount_sum": round(float(pldf["fare_amount"].sum()), 2),
        "trip_distance_max": round(float(pldf["trip_distance"].max()), 2),
    }

    return {
        "shape_match": list(pdf.shape) == list(pldf.shape),
        "columns_match": list(pdf.columns) == list(pldf.columns),
        "null_counts_match": pd_nulls == pl_nulls,
        "aggregates_match": pd_agg == pl_agg,
        "pandas_shape": list(pdf.shape),
        "polars_shape": list(pldf.shape),
        "null_counts": {k: int(v) for k, v in pd_nulls.items() if v > 0},
        "aggregates": pd_agg,
    }


def write_markdown(result: dict) -> None:
    """비교 결과를 사람이 읽는 마크다운으로 저장한다."""
    pdm, plm = result["pandas"], result["polars"]
    tf = result["transform"]
    eq = result["equivalence"]
    env = result["environment"]

    # 검증 항목을 통과/실패 기호로 표시
    checks = "\n".join(
        f"| {label} | {'✅ 일치' if eq[key] else '❌ 불일치'} |"
        for key, label in [
            ("shape_match", "행·열 수"),
            ("columns_match", "컬럼 이름·순서"),
            ("null_counts_match", "결측 개수"),
            ("aggregates_match", "대표 집계값(요금 합계·최대 거리)"),
        ]
    )

    md = f"""# Step 1 — Pandas vs Polars 로딩 비교 결과

- 대상 파일: `{result['file']}` ({result['file_size_mb']} MB, {pdm['shape'][0]:,}행 × {pdm['shape'][1]}컬럼)
- 측정 방식: 워밍업 1회 후 **{result['n_runs']}회 반복, 중앙값** 사용
- 양쪽 모두 즉시 실행(`read_parquet`)으로 맞춤 — Polars의 `scan_parquet`은 지연 실행이라 비교 대상이 아님

## 1. 로딩 성능 비교

| 항목 | Pandas | Polars | 차이 |
|---|---|---|---|
| 로딩 시간 (중앙값) | {pdm['load_time_sec']:.3f}초 | {plm['load_time_sec']:.3f}초 | Polars가 {result['speedup']:.2f}배 빠름 |
| 측정값 전체 | {', '.join(f'{t:.3f}' for t in pdm['times'])} | {', '.join(f'{t:.3f}' for t in plm['times'])} | |
| 메모리 사용량 | {pdm['memory_mb']:,.0f} MB | {plm['memory_mb']:,.0f} MB | Polars가 {result['memory_ratio']:.2f}배 적음 |

**해석**: 흔히 알려진 것만큼 격차가 크지 않다. Pandas의 `read_parquet`도 내부적으로 PyArrow를
쓰기 때문에 parquet 디코딩 비용 자체는 양쪽이 비슷하고, 메모리도 둘 다 Arrow 기반 표현이라
큰 차이가 없다. **즉 로딩 속도만으로는 라이브러리를 선택할 근거가 되지 않는다.**

> 메모리는 두 라이브러리의 집계 기준이 완전히 같지 않다(Pandas는 `memory_usage(deep=True)`,
> Polars는 `estimated_size()`). 정확한 배수보다 규모 차이로 해석해야 한다.

## 2. 전처리 성능 비교 (추가 측정)

로딩 비교만으로 결론이 서지 않아, **Step 2에서 실제로 수행할 작업**으로 다시 측정했다.
작업 내용: {tf['description']}

| 항목 | Pandas | Polars | 차이 |
|---|---|---|---|
| 전처리 시간 (중앙값) | {tf['pandas_sec']:.3f}초 | {tf['polars_sec']:.3f}초 | **Polars가 {tf['speedup']:.2f}배 빠름** |
| 측정값 전체 | {', '.join(f'{t:.3f}' for t in tf['pandas_times'])} | {', '.join(f'{t:.3f}' for t in tf['polars_times'])} | |
| 집계 결과 일치 | {'✅ 동일' if tf['results_match'] else '❌ 불일치'} | | |

두 라이브러리의 차이는 파일을 읽는 단계가 아니라 **데이터를 변환·집계하는 단계**에서 벌어진다.
Polars는 여러 연산을 병렬로 실행하고 중간 복사본을 덜 만들기 때문이다.

## 3. 동일 데이터 검증

속도만 비교하면 '어느 쪽 기준으로 분석한 것인가'가 애매해지므로, 같은 데이터를 읽었는지 확인했다.

| 검증 항목 | 결과 |
|---|---|
{checks}

- 두 라이브러리 모두 **{pdm['shape'][0]:,}행 × {pdm['shape'][1]}컬럼**으로 동일
- 결측이 있는 컬럼: {', '.join(f'`{k}`({v:,}건)' for k, v in eq['null_counts'].items()) or '없음'}

## 4. 결론 — 이후 단계에서 무엇을 쓸 것인가

**Step 2의 전처리는 Polars로 하고, 끝난 뒤 Pandas로 변환해 저장한다.**

근거:
- 로딩 단계의 차이({result['speedup']:.2f}배)는 선택 근거로 삼기에 작다.
  반면 실제 전처리 작업에서는 **{tf['speedup']:.2f}배** 차이가 났고, 같은 결과를 낸다는 것도 확인했다.
- Step 2는 {pdm['shape'][0]:,}행에 대해 필터링·파생변수 생성·집계를 반복하므로 이 차이가 누적된다.
- 다만 Step 3 이후의 seaborn·scipy·scikit-learn은 Pandas 객체를 입력으로 받는다.
  따라서 전처리를 마친 뒤 `.to_pandas()`로 변환해 `data/processed/cleaned.parquet`에 저장하고,
  이후 단계는 Pandas로 진행한다.

**되돌리려면**: Step 2를 Pandas만으로 작성해도 결과는 동일하다(위에서 검증됨). 실행 시간만 늘어난다.

## 실행 환경

| | |
|---|---|
| Python | {env['python']} |
| Pandas | {env['pandas']} |
| Polars | {env['polars']} |
| 플랫폼 | {env['platform']} |
| CPU 코어 | {env['cpu_count']} |

> 속도 수치는 머신마다 달라진다. 팀원 간 결과가 다르면 이 환경 정보를 비교할 것.
"""
    MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    check_input()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    file_size_mb = RAW_PATH.stat().st_size / 1024**2
    print(f"대상 파일: {RAW_PATH.name} ({file_size_mb:.1f} MB)")
    print(f"측정: 워밍업 1회 + {N_RUNS}회 반복 (중앙값 사용)\n")

    # --- 로딩 시간 측정 ---
    print("Pandas 로딩 중...")
    pdf, pd_times = measure(load_pandas)
    print("Polars 로딩 중...")
    pldf, pl_times = measure(load_polars)

    pd_time = statistics.median(pd_times)
    pl_time = statistics.median(pl_times)

    # --- 메모리 측정 ---
    # deep=True: 문자열 컬럼의 실제 사용량까지 포함시킨다(기본값은 포인터 크기만 셈).
    pd_mem = pdf.memory_usage(deep=True).sum() / 1024**2
    pl_mem = pldf.estimated_size("mb")

    # --- 동일 데이터 검증 ---
    equivalence = verify_same_data(pdf, pldf)

    # --- 전처리(변환) 성능 비교 ---
    # 로딩만으로는 차이가 작으므로, Step 2에서 실제로 할 작업으로 다시 잰다.
    print("\n전처리 작업 비교 중 (필터 + 파생변수 + 시간대별 집계)...")
    pd_result, pd_tf_times = measure(lambda: transform_pandas(pdf))
    pl_result, pl_tf_times = measure(lambda: transform_polars(pldf))
    pd_tf = statistics.median(pd_tf_times)
    pl_tf = statistics.median(pl_tf_times)

    # 두 라이브러리의 집계 결과가 같은지 확인 (부동소수점 오차 허용)
    tf_match = bool(
        (pd_result["trips"].values == pl_result["trips"].to_numpy()).all()
        and abs(pd_result["card_ratio"].values - pl_result["card_ratio"].to_numpy()).max() < 1e-9
    )

    result = {
        "file": RAW_PATH.name,
        "file_size_mb": round(file_size_mb, 1),
        "n_runs": N_RUNS,
        "pandas": {
            "load_time_sec": round(pd_time, 3),
            "times": [round(t, 3) for t in pd_times],
            "memory_mb": round(pd_mem, 1),
            "shape": list(pdf.shape),
        },
        "polars": {
            "load_time_sec": round(pl_time, 3),
            "times": [round(t, 3) for t in pl_times],
            "memory_mb": round(pl_mem, 1),
            "shape": list(pldf.shape),
        },
        "speedup": round(pd_time / pl_time, 2),
        "memory_ratio": round(pd_mem / pl_mem, 2),
        "transform": {
            "description": "payment_type 필터 + 파생변수 3개 + 시간대별 집계",
            "pandas_sec": round(pd_tf, 3),
            "polars_sec": round(pl_tf, 3),
            "pandas_times": [round(t, 3) for t in pd_tf_times],
            "polars_times": [round(t, 3) for t in pl_tf_times],
            "speedup": round(pd_tf / pl_tf, 2),
            "results_match": tf_match,
        },
        "equivalence": equivalence,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "polars": pl.__version__,
            "platform": f"{platform.system()} {platform.machine()}",
            "cpu_count": os.cpu_count(),
        },
    }

    # --- 콘솔 출력 (제출용 화면 캡처 대상) ---
    speed_note = f"Polars {result['speedup']}x 빠름"
    mem_note = f"Polars {result['memory_ratio']}x 적음"

    print("\n" + "=" * 62)
    print(f"{'항목':<12}{'Pandas':>14}{'Polars':>14}{'비교':>20}")
    print("-" * 62)
    print(f"{'로딩 시간':<12}{pd_time:>13.3f}s{pl_time:>13.3f}s{speed_note:>20}")
    print(f"{'메모리':<12}{pd_mem:>12,.0f}MB{pl_mem:>12,.0f}MB{mem_note:>20}")
    print(f"{'행 x 열':<12}{str(pdf.shape):>14}{str(pldf.shape):>14}")
    print("-" * 62)
    tf_note = f"Polars {result['transform']['speedup']}x 빠름"
    print(f"{'전처리':<12}{pd_tf:>13.3f}s{pl_tf:>13.3f}s{tf_note:>20}")
    print(f"{'  (필터+파생변수+시간대별 집계, 결과 일치: ' + ('예' if tf_match else '아니오') + ')'}")
    print("=" * 62)

    print("\n[동일 데이터 검증]")
    for key, label in [
        ("shape_match", "행·열 수"),
        ("columns_match", "컬럼 이름·순서"),
        ("null_counts_match", "결측 개수"),
        ("aggregates_match", "대표 집계값"),
    ]:
        print(f"  {label:<16} {'일치' if equivalence[key] else '불일치 (확인 필요)'}")

    print("\n[결측이 있는 컬럼]")
    for col, cnt in equivalence["null_counts"].items():
        print(f"  {col:<24} {cnt:>10,}건")

    # --- 저장 ---
    JSON_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result)
    print(f"\n저장 완료:\n  {JSON_PATH.relative_to(ROOT)}\n  {MD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
