"""실험 3 — 리뷰에서 '확인 필요'로 남긴 데이터 관련 주장 검증"""
import json
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/yellow_tripdata_2026-05.parquet"
CLEAN = ROOT / "data/processed/cleaned.parquet"

df = pd.read_parquet(CLEAN)
print(f"cleaned: {len(df):,}행\n")

# ── 1. 이상값 규칙 경계 (카운트는 >200/>180, 필터는 <200/<180) ──────────────
raw = pl.read_parquet(RAW)
sel = raw.filter(pl.col("payment_type").is_in([0, 1, 2])).with_columns(
    ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_seconds() / 60).alias("dur"))
n_dist_eq = int(sel.select((pl.col("trip_distance") == 200.0).sum()).item())
n_dur_eq = int(sel.select((pl.col("dur") == 180.0).sum()).item())
print("── 1. 규칙 경계에서 누락되는 행 ──")
print(f"  trip_distance == 200.0 정확히: {n_dist_eq}건")
print(f"  trip_duration_min == 180.0 정확히: {n_dur_eq}건")
print(f"  → 제거되지만 어떤 규칙 카운트에도 안 잡히는 행: {n_dist_eq + n_dur_eq}건")

# 규칙 합계 vs 실제 제거 수 대조
removed = sel.height - len(df)
print(f"  참고: 대상선별 후 {sel.height:,} → 정제 후 {len(df):,} = 실제 제거 {removed:,}건\n")

# ── 2. VendorID × 결제수단 (features.json에 있으나 EDA 미검토) ──────────────
print("── 2. VendorID × 결제수단 구성비 (%) ── ※ EDA에 없던 항목")
ct = pd.crosstab(df["VendorID"], df["payment_label"], normalize="index").mul(100).round(2)
cnt = df["VendorID"].value_counts()
for v in ct.index:
    print(f"  VendorID {v}: n={cnt[v]:>10,}  " +
          "  ".join(f"{c} {ct.loc[v, c]:>6.2f}%" for c in ct.columns))
# 역방향: 각 결제수단이 어느 vendor에서 오는가
print("\n  역방향 — 결제수단별 VendorID 구성비 (%)")
ct2 = pd.crosstab(df["payment_label"], df["VendorID"], normalize="index").mul(100).round(2)
for c in ct2.index:
    print(f"  {c:<10} " + "  ".join(f"V{v} {ct2.loc[c, v]:>6.2f}%" for v in ct2.columns))

# ── 3. DOLocationID (features.json에 있으나 EDA 미검토) ────────────────────
print("\n── 3. DOLocationID 상위 5개 구역 구성비 (%) ── ※ EDA에 없던 항목")
top_do = df["DOLocationID"].value_counts().head(5)
sub = df[df["DOLocationID"].isin(top_do.index)]
ct3 = pd.crosstab(sub["DOLocationID"], sub["payment_label"], normalize="index").mul(100).round(2)
for z in top_do.index:
    print(f"  DO {z:>4}: n={top_do[z]:>9,}  " + "  ".join(f"{c} {ct3.loc[z, c]:>6.2f}%" for c in ct3.columns))

# ── 4. is_weekend 가 day_of_week 대비 추가 정보가 있는가 ────────────────────
print("\n── 4. is_weekend (생성했으나 features.json에서 제외됨) ──")
print(f"  is_weekend 는 day_of_week 로부터 완전히 결정되는가: "
      f"{df.groupby('day_of_week')['is_weekend'].nunique().max() == 1}")

# ── 5. t-검정: plan.md Step 4 가 지정한 조합 (카드 vs 현금, fare_amount) ────
print("\n── 5. t-검정 — plan.md Step 4 지정 조합 (신용카드 vs 현금, fare_amount) ──")


def welch(a, b, la, lb):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    n1, n2 = len(a), len(b)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    print(f"  {la}(n={n1:,}, 평균 {a.mean():.4f}, sd {s1:.4f}) vs "
          f"{lb}(n={n2:,}, 평균 {b.mean():.4f}, sd {s2:.4f})")
    print(f"    차이 {a.mean()-b.mean():+.4f} | t = {t:.3f} | p = {p:.6g} | Cohen's d = {d:.4f}")
    return p, d


card = df.loc[df["payment_label"] == "신용카드", "fare_amount"]
cash = df.loc[df["payment_label"] == "현금", "fare_amount"]
flex = df.loc[df["payment_label"] == "Flex Fare", "fare_amount"]
welch(card, cash, "신용카드", "현금")
print("\n  비교용 — PDF 리포트가 실제로 한 조합 및 대안:")
welch(flex, card, "Flex Fare", "신용카드")
cardd = df.loc[df["payment_label"] == "신용카드", "trip_distance"]
flexd = df.loc[df["payment_label"] == "Flex Fare", "trip_distance"]
welch(flexd, cardd, "Flex Fare(거리)", "신용카드(거리)")

# ── 6. 팁 / $0.50 배수 — eda.json 값 재확인 ────────────────────────────────
print("\n── 6. 누수 진단 수치 재확인 ──")
tip = df.assign(t=df["tip_amount"] > 0).groupby("payment_label")["t"].mean().mul(100)
for k, v in tip.items():
    n = int((df["payment_label"] == k).sum() * v / 100)
    print(f"  팁>0  {k:<10} {v:>6.2f}%  ({n:,}건)")
cents = (df["fare_amount"] * 100).round().astype("int64")
half = df.assign(h=cents % 50 == 0).groupby("payment_label")["h"].mean().mul(100)
for k, v in half.items():
    print(f"  $0.50 배수  {k:<10} {v:>6.2f}%")

# ── 7. fare_amount 상한 부재의 영향 ────────────────────────────────────────
print("\n── 7. fare_amount 에만 상한이 없는 영향 ──")
f = df["fare_amount"]
for q in [0.99, 0.999, 0.9999]:
    print(f"  {q*100:>7.2f}% 분위: ${f.quantile(q):>8.2f}")
print(f"  최댓값: ${f.max():.2f} | $200 초과: {(f > 200).sum():,}건 | $500 초과: {(f > 500).sum():,}건")
