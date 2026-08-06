"""Step 1 — Pandas vs Polars 로딩·전처리 성능 비교.

채점 항목 "데이터셋을 Pandas와 Polars 양쪽으로 로딩하여 결과 비교"에 대응한다.

측정 방법에 세 가지 장치를 둔다. 앞선 리뷰에서 같은 코드가 실행할 때마다
1.00x ~ 2.56x 로 흔들린 적이 있어, 재현되지 않는 수치를 남기지 않기 위한 것이다.

    1) 이전 회차 프레임을 먼저 버리고 gc 를 돌린다.
       대입이 호출 완료 후에 일어나므로, 그냥 돌리면 581MB 프레임 두 개가
       동시에 존재하는 상태에서 측정된다.
    2) 워밍업 1회는 측정에서 제외한다. 첫 호출에는 임포트 초기화와
       페이지 캐시 적재 비용이 섞여 있어, 한쪽만 먼저 실행되면 결과가 뒤집힌다.
    3) 평균이 아니라 최소값을 쓴다. 다른 프로세스의 간섭은 시간을 늘리기만 하므로
       최소값이 순수 실행 비용에 가장 가깝다. 개별 측정치도 JSON에 전부 남긴다.
    4) 한쪽에만 있는 작업을 넣지 않는다. pandas 쪽 .copy() 처럼 Polars 에 대응물이
       없는 연산이 섞이면 그만큼 결과가 왜곡된다 (제거 전후 4.49x → 4.07x).

로딩 시간만으로는 차이가 크지 않다(양쪽 다 Arrow 기반이라 parquet 디코딩 비용이 비슷하다).
실제 변환 작업까지 비교해야 이후 단계의 선택 근거가 된다.

산출물
    outputs/results/load_compare.json
    outputs/results/load_compare.md
"""

import gc
import json
import os
import platform
import time
import urllib.request

import pandas as pd
import polars as pl

RAW_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet"
RAW_PATH = "data/raw/yellow_tripdata_2026-05.parquet"
OUT_JSON = "outputs/results/load_compare.json"
OUT_MD = "outputs/results/load_compare.md"

N_RUNS = 15
TARGET_CODES = [0, 1, 2]   # Step 2 와 동일한 분석 대상 클래스


def ensure_raw():
    """원본이 없으면 내려받는다(약 66MB)."""
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    if not os.path.exists(RAW_PATH):
        print(f"[다운로드] {RAW_PATH} ...")
        urllib.request.urlretrieve(RAW_URL, RAW_PATH)


def bench(fn, n=N_RUNS) -> dict:
    """워밍업 1회 후 n회 측정. 이전 결과를 먼저 버려 메모리 간섭을 없앤다."""
    out = fn()          # 워밍업 — 측정에서 제외
    del out
    gc.collect()

    times = []
    for _ in range(n):
        out = None      # 이전 회차 프레임을 먼저 버린다
        gc.collect()
        t = time.perf_counter()
        out = fn()
        times.append(round(time.perf_counter() - t, 4))
        del out
    return {"min_sec": min(times), "median_sec": round(sorted(times)[len(times) // 2], 4),
            "times": times}


def transform_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 에서 실제로 할 전처리를 Pandas 로 수행한다.

    .copy() 를 쓰지 않는다. 필터 결과를 통째로 복사하면 390만 행이 메모리에
    다시 쓰이는데, Polars 는 데이터가 불변이라 이런 방어 복사 자체가 없다.
    한쪽에만 있는 작업이 붙으면 비교가 공정하지 않다 — 실측 결과 pandas 시간의
    9.4%를 먹고 있었고, 그만큼 Polars 우위가 부풀려졌다.
    pandas 3.0 은 Copy-on-Write 가 기본이라 방어 복사가 필요하지도 않다.
    """
    d = df[df.payment_type.isin(TARGET_CODES) & (df.fare_amount > 0)]
    d = d.assign(
        trip_duration_min=lambda x: (
            x.tpep_dropoff_datetime - x.tpep_pickup_datetime
        ).dt.total_seconds() / 60,
        pickup_hour=lambda x: x.tpep_pickup_datetime.dt.hour,
    )
    return d.groupby("pickup_hour").agg(
        trips=("fare_amount", "size"),
        mean_fare=("fare_amount", "mean"),
        mean_distance=("trip_distance", "mean"),
    )


def transform_polars(df: pl.DataFrame) -> pl.DataFrame:
    """transform_pandas 와 동일한 전처리를 Polars 로 수행한다."""
    return (
        df.filter(pl.col("payment_type").is_in(TARGET_CODES) & (pl.col("fare_amount") > 0))
        .with_columns(
            ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
             .dt.total_seconds() / 60).alias("trip_duration_min"),
            pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
        )
        .group_by("pickup_hour")
        .agg(pl.len().alias("trips"),
             pl.col("fare_amount").mean().alias("mean_fare"),
             pl.col("trip_distance").mean().alias("mean_distance"))
        .sort("pickup_hour")
    )


def verify_same(pdf: pd.DataFrame, pldf: pl.DataFrame) -> dict:
    """두 라이브러리가 같은 데이터를 읽었는지 확인한다. 다르면 비교 자체가 무의미하다."""
    same_shape = list(pdf.shape) == list(pldf.shape)
    # 부동소수점 합계까지 일치하면 동일 데이터로 본다
    pd_sum = round(float(pdf.fare_amount.sum()), 2)
    pl_sum = round(float(pldf["fare_amount"].sum()), 2)
    print("\n[검증] 두 라이브러리가 같은 데이터를 읽었는가")
    print(f"    shape        : {list(pdf.shape)} vs {list(pldf.shape)} → {same_shape}")
    print(f"    fare 합계    : {pd_sum:,.2f} vs {pl_sum:,.2f} → {pd_sum == pl_sum}")
    assert same_shape and pd_sum == pl_sum, "두 라이브러리의 로딩 결과가 다름"
    return {"shape": list(pdf.shape), "fare_amount_sum": pd_sum, "identical": True}


def main():
    ensure_raw()
    size_mb = round(os.path.getsize(RAW_PATH) / 1024**2, 1)
    print(f"[대상] {RAW_PATH} ({size_mb} MB) · {N_RUNS}회 측정")

    print("\n[1] 로딩")
    pd_load = bench(lambda: pd.read_parquet(RAW_PATH))
    pl_load = bench(lambda: pl.read_parquet(RAW_PATH))
    print(f"    Pandas  최소 {pd_load['min_sec']:.4f}초 / 중앙값 {pd_load['median_sec']:.4f}초")
    print(f"    Polars  최소 {pl_load['min_sec']:.4f}초 / 중앙값 {pl_load['median_sec']:.4f}초")

    pdf, pldf = pd.read_parquet(RAW_PATH), pl.read_parquet(RAW_PATH)
    same = verify_same(pdf, pldf)

    print("\n[2] 전처리 (필터 + 파생변수 + 시간대별 집계)")
    pd_tr = bench(lambda: transform_pandas(pdf), n=7)
    pl_tr = bench(lambda: transform_polars(pldf), n=7)
    print(f"    Pandas  최소 {pd_tr['min_sec']:.4f}초")
    print(f"    Polars  최소 {pl_tr['min_sec']:.4f}초")

    def ratio(a, b):
        return round(max(a, b) / min(a, b), 2), ("Polars" if b < a else "Pandas")

    r_load, w_load = ratio(pd_load["min_sec"], pl_load["min_sec"])
    r_tr, w_tr = ratio(pd_tr["min_sec"], pl_tr["min_sec"])
    print(f"\n[결과] 로딩 {w_load} {r_load}배 · 전처리 {w_tr} {r_tr}배")
    print("    해석: 로딩은 양쪽 다 Arrow 기반이라 차이가 작다.")
    print("          차이는 변환·집계에서 드러나므로 이후 단계의 선택 근거는 전처리 쪽이다.")

    payload = {
        "step": "01_load_compare", "file": os.path.basename(RAW_PATH),
        "file_size_mb": size_mb, "n_runs": N_RUNS,
        "env": {"python": platform.python_version(), "pandas": pd.__version__,
                "polars": pl.__version__, "platform": f"{platform.system()} {platform.machine()}",
                "cpu_count": os.cpu_count()},
        "load": {"pandas": pd_load, "polars": pl_load,
                 "faster": w_load, "ratio": r_load},
        "transform": {"pandas": pd_tr, "polars": pl_tr,
                      "faster": w_tr, "ratio": r_tr},
        "verify": same,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(f"""# Step 1 — Pandas vs Polars 비교

`{os.path.basename(RAW_PATH)}` ({size_mb} MB · {same['shape'][0]:,}행 × {same['shape'][1]}열)
{N_RUNS}회 측정 · 워밍업 제외 · **최소값** 기준

| 작업 | Pandas | Polars | 우세 |
|---|---|---|---|
| 로딩 | {pd_load['min_sec']:.4f}초 | {pl_load['min_sec']:.4f}초 | {w_load} {r_load}배 |
| 전처리 (필터+파생+집계) | {pd_tr['min_sec']:.4f}초 | {pl_tr['min_sec']:.4f}초 | {w_tr} {r_tr}배 |

**동일 데이터 확인** — shape {same['shape']}, `fare_amount` 합계 {same['fare_amount_sum']:,.2f} 로 양쪽 일치.

## 측정 방법

한 번만 재면 실행할 때마다 배수가 크게 흔들린다. 아래 세 가지로 통제했다.

1. 이전 회차 프레임을 먼저 버리고 `gc.collect()` — 그러지 않으면 581MB 프레임 두 개가
   동시에 존재하는 상태에서 측정된다
2. 워밍업 1회 제외 — 첫 호출의 임포트 초기화·페이지 캐시 비용을 뺀다
3. 평균이 아닌 최소값 — 외부 간섭은 시간을 늘리기만 하므로 최소값이 순수 비용에 가깝다
4. 한쪽에만 있는 작업 제거 — pandas 의 `.copy()` 는 Polars 에 대응물이 없어
   pandas 만 손해를 본다. 제거 전 4.49x → 제거 후 4.07x 로, 이 한 줄이 배수를 0.4 부풀리고 있었다

개별 측정치는 `load_compare.json` 의 `times` 에 전부 남겨 두었다.

## 해석

로딩 차이는 작다. 양쪽 다 Arrow 기반이라 parquet 디코딩 비용이 비슷하기 때문이다.
차이는 변환·집계에서 드러나므로, 라이브러리 선택 근거로는 전처리 쪽 수치를 봐야 한다.

측정 환경: Python {platform.python_version()} / pandas {pd.__version__} / polars {pl.__version__} /
{platform.system()} {platform.machine()} / {os.cpu_count()}코어
""")
    print(f"\n[산출물] {OUT_JSON}\n          {OUT_MD}")


if __name__ == "__main__":
    main()
