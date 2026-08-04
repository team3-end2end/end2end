"""실험 1 — 속도 측정의 안정성 + .copy() 핸디캡 A/B 테스트"""
import gc, statistics, time
from pathlib import Path
import pandas as pd, polars as pl

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "yellow_tripdata_2026-05.parquet"
N = 15  # 원본은 3회. 편차를 보기 위해 15회로 늘린다.


def bench(fn, n=N):
    fn(); gc.collect()          # 워밍업 (원본 measure()와 동일)
    ts = []
    for _ in range(n):
        gc.collect()
        s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
    return ts


def stat(ts):
    return dict(median=statistics.median(ts), mn=min(ts), mx=max(ts),
                mean=statistics.mean(ts), sd=statistics.stdev(ts))


print(f"=== 로딩 {N}회 ===")
pd_l = bench(lambda: pd.read_parquet(RAW))
pl_l = bench(lambda: pl.read_parquet(RAW))
sp, sl = stat(pd_l), stat(pl_l)
print(f"pandas  median {sp['median']:.3f}  min {sp['mn']:.3f}  max {sp['mx']:.3f}  sd {sp['sd']:.3f}")
print(f"polars  median {sl['median']:.3f}  min {sl['mn']:.3f}  max {sl['mx']:.3f}  sd {sl['sd']:.3f}")
print(f"배수: median기준 {sp['median']/sl['median']:.2f}x | min기준 {sp['mn']/sl['mn']:.2f}x | max기준 {sp['mx']/sl['mx']:.2f}x")

# 3회 표본을 반복 추출해 원본 방식이 어떤 값을 낼 수 있었는지 본다
import random
random.seed(42)
ratios = []
for _ in range(2000):
    a = statistics.median(random.sample(pd_l, 3))
    b = statistics.median(random.sample(pl_l, 3))
    ratios.append(a / b)
ratios.sort()
print(f"→ 3회 표본(원본 방식)으로 나올 수 있는 배수 범위: "
      f"{ratios[0]:.2f}x ~ {ratios[-1]:.2f}x (5~95%: {ratios[100]:.2f}~{ratios[1900]:.2f})")

# ---------------------------------------------------------------- transform
pdf = pd.read_parquet(RAW)
pldf = pl.read_parquet(RAW)


def tf_pandas_orig(d0):
    """현재 코드 그대로 (01_load_compare.py:92 의 .copy() 포함)"""
    d = d0[d0["payment_type"].isin([1, 2]) & (d0["fare_amount"] > 0)].copy()
    d["duration_min"] = (d["tpep_dropoff_datetime"] - d["tpep_pickup_datetime"]).dt.total_seconds() / 60
    d["pickup_hour"] = d["tpep_pickup_datetime"].dt.hour
    d["is_card"] = (d["payment_type"] == 1).astype(int)
    return d.groupby("pickup_hour").agg(trips=("is_card", "size"),
                                        card_ratio=("is_card", "mean"),
                                        mean_fare=("fare_amount", "mean"))


def tf_pandas_nocopy(d0):
    """.copy() 만 제거. 나머지는 동일."""
    d = d0[d0["payment_type"].isin([1, 2]) & (d0["fare_amount"] > 0)]
    d = d.assign(duration_min=(d["tpep_dropoff_datetime"] - d["tpep_pickup_datetime"]).dt.total_seconds() / 60,
                 pickup_hour=d["tpep_pickup_datetime"].dt.hour,
                 is_card=(d["payment_type"] == 1).astype(int))
    return d.groupby("pickup_hour").agg(trips=("is_card", "size"),
                                        card_ratio=("is_card", "mean"),
                                        mean_fare=("fare_amount", "mean"))


def tf_polars(d0):
    return (d0.filter(pl.col("payment_type").is_in([1, 2]) & (pl.col("fare_amount") > 0))
            .with_columns(((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_seconds() / 60).alias("duration_min"),
                          pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
                          (pl.col("payment_type") == 1).cast(pl.Int8).alias("is_card"))
            .group_by("pickup_hour")
            .agg(pl.len().alias("trips"), pl.col("is_card").mean().alias("card_ratio"),
                 pl.col("fare_amount").mean().alias("mean_fare"))
            .sort("pickup_hour"))


# 두 pandas 구현이 같은 결과를 내는지 먼저 확인
a, b = tf_pandas_orig(pdf), tf_pandas_nocopy(pdf)
assert (a["trips"].values == b["trips"].values).all()
assert abs(a["card_ratio"].values - b["card_ratio"].values).max() < 1e-12
print("\n두 pandas 구현 결과 동일: 확인됨")

print(f"\n=== 전처리 {N}회 ===")
o = stat(bench(lambda: tf_pandas_orig(pdf)))
nc = stat(bench(lambda: tf_pandas_nocopy(pdf)))
p = stat(bench(lambda: tf_polars(pldf)))
print(f"pandas (.copy() 있음, 현재 코드)  median {o['median']:.3f}  sd {o['sd']:.3f}")
print(f"pandas (.copy() 제거)            median {nc['median']:.3f}  sd {nc['sd']:.3f}")
print(f"polars                           median {p['median']:.3f}  sd {p['sd']:.3f}")
print(f"\n배수(현재 코드):   {o['median']/p['median']:.2f}x")
print(f"배수(.copy() 제거): {nc['median']/p['median']:.2f}x")
print(f"→ .copy() 한 줄이 pandas 시간의 {(o['median']-nc['median'])/o['median']*100:.0f}% 를 차지")
