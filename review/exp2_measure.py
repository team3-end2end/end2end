"""실험 2 — 원본 measure()의 측정값이 튀는 원인 찾기

가설: 01_load_compare.py:63-67 의 루프가
    for _ in range(n_runs):
        gc.collect()
        start = ...
        df = load_fn()        # ← 이전 df가 아직 살아 있는 상태에서 새 df를 만든다
        times.append(...)
이므로, 2회차부터는 581MB 프레임 2개가 동시에 존재한다(피크 ~1.2GB).
즉 1회차와 2·3회차의 측정 조건이 다르다.
"""
import gc, statistics, time
from pathlib import Path
import pandas as pd, polars as pl

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/yellow_tripdata_2026-05.parquet"
REPEAT = 8   # 같은 프로토콜을 8번 반복해 회차별 분포를 본다


def measure_original(load_fn, n_runs=3):
    """01_load_compare.py 의 measure() 를 그대로 옮긴 것"""
    df = load_fn(); del df; gc.collect()
    times = []
    df = None
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        df = load_fn()            # 이전 df 가 살아 있음
        times.append(time.perf_counter() - start)
    return df, times


def measure_fixed(load_fn, n_runs=3):
    """이전 결과를 먼저 버리고 재는 버전 (그 외 동일)"""
    df = load_fn(); del df; gc.collect()
    times = []
    df = None
    for _ in range(n_runs):
        df = None                 # ← 이전 프레임을 먼저 해제
        gc.collect()
        start = time.perf_counter()
        df = load_fn()
        times.append(time.perf_counter() - start)
    return df, times


for name, fn in [("원본 measure()", measure_original), ("이전 df 해제 후 측정", measure_fixed)]:
    print(f"=== {name} — 3회 프로토콜을 {REPEAT}번 반복 ===")
    per_run = [[], [], []]
    ratios = []
    for _ in range(REPEAT):
        _, pt = fn(lambda: pd.read_parquet(RAW))
        _, lt = fn(lambda: pl.read_parquet(RAW))
        for i in range(3):
            per_run[i].append(pt[i])
        ratios.append(statistics.median(pt) / statistics.median(lt))
    for i in range(3):
        v = per_run[i]
        print(f"  pandas {i+1}회차: median {statistics.median(v):.3f}  "
              f"min {min(v):.3f}  max {max(v):.3f}")
    print(f"  → 산출된 speedup 범위: {min(ratios):.2f}x ~ {max(ratios):.2f}x "
          f"(median {statistics.median(ratios):.2f}x)\n")
