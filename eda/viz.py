"""EDA ③ 시각화 — figures/ 에 PNG 저장"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import missingno as msno
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#2a78d6"
RED = "#e34948"
GRAY = "#f0efec"
INK = "#0b0b0b"
MUTED = "#52514e"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", [BLUE, GRAY, RED])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#d8d7d2", "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "font.family": "AppleGothic", "axes.unicode_minus": False,
})

OUT = "eda/handoff/figures"
df = pd.read_parquet("data/raw/yellow_tripdata_2026-05.parquet")
s = df.sample(100_000, random_state=42)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=130)
    plt.close(fig)


# 1. 히스토그램 3단 비교 (원본 / 이상치 제거 / 로그)
for col, cap in [("trip_distance", 20), ("fare_amount", 100)]:
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    v = s[col]
    ax[0].hist(v, bins=60, color=BLUE)
    ax[0].set_title(f"원본  (max={df[col].max():,.0f})", color=INK)
    clean = v[(v > 0) & (v <= cap)]
    ax[1].hist(clean, bins=60, color=BLUE)
    ax[1].set_title(f"0 < x ≤ {cap} 로 절단", color=INK)
    ax[2].hist(np.log1p(clean), bins=60, color=BLUE)
    ax[2].set_title("log(1+x) 변환", color=INK)
    for a in ax:
        a.set_xlabel(col)
    fig.suptitle(f"{col} 분포", fontsize=12, fontweight="bold", color=INK)
    save(fig, f"hist_{col}")

# 2. 박스플롯 (절단 후)
fig, ax = plt.subplots(figsize=(7, 3.2))
cols = ["trip_distance", "fare_amount", "tip_amount"]
data = [s[c][(s[c] > 0) & (s[c] <= 100)] for c in cols]
bp = ax.boxplot(data, tick_labels=cols, orientation="horizontal",
                patch_artist=True, widths=0.5,
                flierprops=dict(marker="o", markersize=2, markerfacecolor=BLUE,
                                markeredgecolor="none", alpha=0.25))
for b in bp["boxes"]:
    b.set(facecolor=BLUE, edgecolor="#fcfcfb", linewidth=2)
for m in bp["medians"]:
    m.set(color="#fcfcfb", linewidth=2)
ax.set_title("이상치 확인 (0 < x ≤ 100 구간)", fontsize=12, fontweight="bold", color=INK)
save(fig, "box_outlier")

# 3. 결측 패턴
fig, ax = plt.subplots(figsize=(10, 3.5))
msno.matrix(s.sort_values("tpep_pickup_datetime"), ax=ax, sparkline=False,
            color=(0.16, 0.47, 0.84))
ax.set_title("결측 패턴 (시간순 정렬, 흰 줄 = 결측)", fontsize=12, fontweight="bold", color=INK)
save(fig, "missing_matrix")

# 4. 거리 vs 요금 산점도
fig, ax = plt.subplots(figsize=(5.2, 4))
t = s[(s.trip_distance > 0) & (s.trip_distance <= 30) &
      (s.fare_amount > 0) & (s.fare_amount <= 150)]
ax.scatter(t.trip_distance, t.fare_amount, s=4, color=BLUE, alpha=0.12,
           edgecolors="none")
ax.set_xlabel("trip_distance (mile)")
ax.set_ylabel("fare_amount ($)")
ax.set_title("거리 vs 요금", fontsize=12, fontweight="bold", color=INK)
save(fig, "scatter_distance_fare")

# 5. 상관 히트맵
num = ["trip_distance", "fare_amount", "extra", "tip_amount", "tolls_amount",
       "congestion_surcharge", "Airport_fee", "total_amount"]
c = df[num].corr()
fig, ax = plt.subplots(figsize=(6.5, 5.4))
im = ax.imshow(c, cmap=DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(num)), num, rotation=45, ha="right")
ax.set_yticks(range(len(num)), num)
for i in range(len(num)):
    for j in range(len(num)):
        ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color=INK)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="상관계수")
ax.set_title("요금 관련 변수 상관관계", fontsize=12, fontweight="bold", color=INK)
save(fig, "corr_heatmap")

print("saved")
print(c["total_amount"].sort_values(ascending=False).to_string())

# ── Target(payment_type) 관계 분석 ────────────────────────────────
LAB = {0: "앱선불(추정)", 1: "카드", 2: "현금", 3: "무료", 4: "분쟁"}
ORDER = ["카드", "현금", "앱선불(추정)"]
COLORS = {"카드": BLUE, "현금": "#eb6834", "앱선불(추정)": "#1baf7a"}
df["pt"] = df.payment_type.map(LAB)
s2 = s.copy()
s2["pt"] = s2.payment_type.map(LAB)

# 6. 결제수단별 요금·거리 분포
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for a, (col, cap, unit) in zip(ax, [("fare_amount", 60, "$"), ("trip_distance", 12, "mile")]):
    data = [s2[(s2.pt == g) & (s2[col] > 0) & (s2[col] <= cap)][col] for g in ORDER]
    bp = a.boxplot(data, tick_labels=ORDER, orientation="horizontal",
                   patch_artist=True, widths=0.55, showfliers=False)
    for b, g in zip(bp["boxes"], ORDER):
        b.set(facecolor=COLORS[g], edgecolor="#fcfcfb", linewidth=2)
    for mm in bp["medians"]:
        mm.set(color="#fcfcfb", linewidth=2)
    a.set_xlabel(f"{col} ({unit})")
fig.suptitle("결제수단별 요금·거리 분포", fontsize=12, fontweight="bold", color=INK)
save(fig, "target_box")

# 7. 사업자별 결제수단 비율
ct = pd.crosstab(df.VendorID, df.pt, normalize="index")[ORDER] * 100
fig, ax = plt.subplots(figsize=(7, 3.2))
left = np.zeros(len(ct))
for g in ORDER:
    ax.barh(ct.index.astype(str), ct[g], left=left, color=COLORS[g], label=g,
            height=0.6, edgecolor="#fcfcfb", linewidth=2)
    for i, (v, l) in enumerate(zip(ct[g], left)):
        if v > 6:
            ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                    fontsize=8, color="#fcfcfb")
    left += ct[g].values
ax.set_xlabel("비율 (%)")
ax.set_ylabel("VendorID")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42))
ax.set_title("사업자별 결제수단 구성", fontsize=12, fontweight="bold", color=INK)
save(fig, "target_vendor")

# 8. 지역별 현금 비율 분포
g = df.groupby("PULocationID")["payment_type"].agg(n="size", cash=lambda x: (x == 2).mean() * 100)
g = g[g.n >= 200]
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.hist(g.cash, bins=40, color=BLUE)
ax.axvline(9.1, color=RED, linewidth=2)
ax.text(10, ax.get_ylim()[1] * 0.85, "전체 평균 9.1%", color=RED, fontsize=9)
ax.set_xlabel("지역별 현금 결제 비율 (%)")
ax.set_ylabel("지역 수")
ax.set_title(f"지역마다 현금 비율이 다르다 (n={len(g)}개 지역)",
             fontsize=12, fontweight="bold", color=INK)
save(fig, "target_location")

print("target figures saved")
