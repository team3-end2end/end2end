"""
피처 서브셋 시각화 — feature_analysis.py PART 5 산출물을 그림으로 확인한다.

입력: data_analysis/outputs/  (없으면 feature_analysis.py 를 먼저 실행)
출력: data_analysis/outputs/figures/*.png

이 파일은 산출물(parquet)을 만들지 않는다. 이미 만들어진 것을 그림으로 보여줄 뿐이라,
서브셋만 필요하면 여기까지 안 돌려도 된다.

실행
────
    python data_analysis/evaluate.py             # 전부 (6장)
    python data_analysis/evaluate.py set         # 최종 세트 그림만 (2장)
    python data_analysis/evaluate.py why         # 설계 근거 그림만 (4장)
    python data_analysis/evaluate.py 1 3         # 번호로 지정
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

BASE = Path(__file__).resolve().parent
SUBSETS = BASE / "outputs" / "subsets"
TABLES = BASE / "outputs" / "tables"
FIGURES = BASE / "outputs" / "figures"

# 그림 표본. 300만 점을 그대로 찍으면 뭉쳐서 아무것도 안 보인다.
N_PLOT = 120_000
RS = 42

# 색: 카드/현금 고정. 색맹 대응을 위해 파랑–주황 조합.
C_CARD, C_CASH = "#4C78A8", "#F58518"
PALETTE = {"신용카드": C_CARD, "현금": C_CASH}

from feature_analysis import FEATURE_SETS, LOC_COLS      # 세트 정의는 한 곳에서만 관리

# 진단용 통합 파일. 타겟 인코딩(OOF)과 모든 파생 피처가 다 들어 있다.
# 학습에 쓰면 안 되고(누수), 그림 그리는 데만 쓴다.
DIAG = "all_features_diagnostic.parquet"


# ──────────────────────────────────────────────────────────────────────
def setup_style() -> None:
    """한글 폰트 + 공통 스타일. 폰트가 없으면 경고만 내고 진행한다."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"):
        if cand in installed:
            plt.rcParams["font.family"] = cand
            break
    else:
        print("⚠️  한글 폰트를 못 찾았습니다. 축 라벨이 깨질 수 있어요.")
    plt.rcParams.update({
        "axes.unicode_minus": False,      # 한글 폰트에서 마이너스가 깨지는 것 방지
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_palette([C_CARD, C_CASH])


def need(path: Path) -> bool:
    if path.exists():
        return True
    print(f"⚠️  {path} 없음 — `.venv/bin/python data_analysis/feature_analysis.py 5` 먼저 실행하세요.")
    return False


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    p = FIGURES / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  저장 → outputs/figures/{name}  ({p.stat().st_size / 1e3:.0f} KB)")


def load_diag(cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(SUBSETS / DIAG, columns=cols)


# ──────────────────────────────────────────────────────────────────────
# 1. 블록별 기여도 — 어떤 피처 묶음이 얼마나 버는가
# ──────────────────────────────────────────────────────────────────────
def fig1_block_contribution() -> None:
    if not (need(TABLES / "01_block_solo.csv") and need(TABLES / "02_block_cumulative.csv")):
        return
    solo = pd.read_csv(TABLES / "01_block_solo.csv")
    cum = pd.read_csv(TABLES / "02_block_cumulative.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    s = solo.sort_values("pseudoR2")
    bars = ax1.barh(s["블록"], s["pseudoR2"],
                    color=[C_CASH if v == s["pseudoR2"].max() else "#9BB7D4"
                           for v in s["pseudoR2"]])
    ax1.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax1.set_xlim(0, s["pseudoR2"].max() * 1.22)
    ax1.set_xlabel("pseudo-R²")
    ax1.set_title("블록 단독 성능", loc="left", fontweight="bold")

    x = range(len(cum))
    ax2.plot(x, cum["pseudoR2"], "o-", color=C_CARD, lw=2, ms=7)
    ax2.fill_between(x, 0, cum["pseudoR2"], color=C_CARD, alpha=0.12)
    for i, r in cum.iterrows():
        lbl = f"{r['pseudoR2']:.4f}"
        if not pd.isna(r.get("dR2", np.nan)):
            lbl += f"\n(+{r['dR2']:.4f})"
        ax2.annotate(lbl, (i, r["pseudoR2"]), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=8.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(cum["모델"], fontsize=9)
    ax2.set_ylim(0, cum["pseudoR2"].max() * 1.35)
    ax2.set_ylabel("pseudo-R²")
    ax2.set_title("블록 누적 — 지리(D)에서 계단이 생김", loc="left", fontweight="bold")

    fig.suptitle("피처 블록별 기여도 (카드 vs 현금)", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "01_block_contribution.png")


# ──────────────────────────────────────────────────────────────────────
# 2. 서브셋별 피처 분포 — 카드 vs 현금이 실제로 갈리는가
# ──────────────────────────────────────────────────────────────────────
def fig2_subset_distributions() -> None:
    if not need(SUBSETS / DIAG):
        return
    full = load_diag().sample(N_PLOT, random_state=RS)

    for key, spec in FEATURE_SETS.items():
        # 지역은 ID든 인코딩이든 값 자체가 커서 다른 피처와 축이 안 맞는다.
        # 지리는 06번 그림에서 따로 다루므로 여기서는 뺀다.
        feats = [c for c in spec["cols"] if c in full and c not in LOC_COLS]
        df = full

        n = len(feats)
        ncol = min(5, n)                       # 한 줄에 5개까지, 넘으면 접는다
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[n:]:
            ax.axis("off")                     # 빈 칸 숨김

        drew_kde = False
        for ax, f in zip(axes, feats):
            v = df[f]
            zero_share = float((v == 0).mean())

            if v.nunique() <= 2:
                # 이진 피처는 밀도 대신 값별 현금비율이 훨씬 잘 보인다
                rate = df.groupby(f)["is_cash"].mean() * 100
                ax.bar([str(i) for i in rate.index], rate.values, color=C_CASH, width=0.55)
                ax.axhline(df["is_cash"].mean() * 100, ls="--", lw=1, color="gray")
                ax.set_ylabel("현금 비율 (%)")
                ax.set_title(f, fontsize=10, fontweight="bold")

            elif zero_share > 0.3:
                # 상호작용 항은 게이트가 꺼지면 정확히 0이라 값의 70%가 0이다.
                # 0을 섞어 KDE를 그리면 스파이크만 남으므로,
                # 0인 쪽은 현금비율로, 0이 아닌 쪽만 분포로 나눠 본다.
                on = df[v != 0]
                rate = [df.loc[v == 0, "is_cash"].mean() * 100,
                        on["is_cash"].mean() * 100]
                ax.bar(["= 0", "≠ 0"], rate, color=[ "#9BB7D4", C_CASH], width=0.5)
                ax.axhline(df["is_cash"].mean() * 100, ls="--", lw=1, color="gray")
                for i, r in enumerate(rate):
                    ax.text(i, r, f"{r:.1f}%", ha="center", va="bottom", fontsize=9)
                ax.set_ylim(0, max(rate) * 1.25)
                ax.set_ylabel("현금 비율 (%)")
                ax.set_title(f"{f}\n(값의 {zero_share:.0%}가 0)", fontsize=9.5, fontweight="bold")

            else:
                lo, hi = v.quantile([0.01, 0.99])           # 꼬리 1%는 잘라야 형태가 보임
                for lab, g in df.groupby("payment_label"):
                    sns.kdeplot(g[f].clip(lo, hi), ax=ax, label=lab, fill=True,
                                alpha=0.35, lw=1.5, color=PALETTE[lab], warn_singular=False)
                ax.set_ylabel("밀도")
                ax.set_title(f, fontsize=10, fontweight="bold")
                drew_kde = True
            ax.set_xlabel("")

        if drew_kde:
            for ax in axes:
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=8, frameon=False)
                    break
        fig.suptitle(f"{key} 세트 ({spec['설명']}) — 결제수단별 분포  "
                     f"표본 {N_PLOT:,} · 지역 피처는 06번 그림 참고",
                     fontsize=12, fontweight="bold", x=0.01, ha="left")
        fig.tight_layout()
        save(fig, f"02_dist_{key}.png")


# ──────────────────────────────────────────────────────────────────────
# 3. 효과크기 — 개별 피처는 다 죽어 있다
# ──────────────────────────────────────────────────────────────────────
def fig3_effect_sizes() -> None:
    if not need(TABLES / "05_feature_effect.csv"):
        return
    t = pd.read_csv(TABLES / "05_feature_effect.csv").sort_values("값", key=abs)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [C_CARD if k == "수치형" else "#72B7B2" for k in t["종류"]]
    bars = ax.barh(t["feature"], t["값"].abs(), color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5)

    # Cohen's d 관례적 경계
    for x, lab in [(0.2, "작음 0.2"), (0.5, "중간 0.5"), (0.8, "큼 0.8")]:
        ax.axvline(x, ls="--", lw=1, color="#C44E52", alpha=0.6)
        ax.text(x, len(t) - 0.3, lab, fontsize=8, color="#C44E52", ha="center")

    ax.set_xlim(0, max(0.9, t["값"].abs().max() * 1.15))
    ax.set_xlabel("효과크기 |Cohen's d| (수치형) / Cramer's V (범주형)")
    ax.set_title("원본 피처 개별 효과크기 — 전부 '작음' 미만\n"
                 "조합·상호작용이 필요한 이유", loc="left", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_CARD),
               plt.Rectangle((0, 0), 1, 1, color="#72B7B2")]
    ax.legend(handles, ["수치형 (Cohen's d)", "범주형 (Cramer's V)"],
              fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "03_effect_sizes.png")


# ──────────────────────────────────────────────────────────────────────
# 4. 상호작용 — E 블록이 왜 필요한지 한 장으로
# ──────────────────────────────────────────────────────────────────────
def fig4_interaction() -> None:
    if not need(SUBSETS / DIAG):
        return
    df = pd.read_parquet(SUBSETS / DIAG,
                         columns=["log_fare", "log_dist", "is_airport", "is_flat70",
                                  "is_cash", "payment_label", "PULocationID"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # (좌) 공항 여부에 따라 요금↔현금 곡선의 높이·기울기가 완전히 달라진다
    df["fare_bin"] = pd.qcut(df["log_fare"], 12, duplicates="drop")
    for flag, lab, col in [(0, "비공항", C_CARD), (1, "공항", C_CASH)]:
        g = df[df.is_airport == flag].groupby("fare_bin", observed=True)
        ax1.plot(np.expm1(g["log_fare"].mean()), g["is_cash"].mean() * 100,
                 "o-", color=col, label=lab, lw=2, ms=5)
    ax1.set_xscale("log")
    ax1.set_xticks([5, 10, 20, 40, 70])
    ax1.set_xticks([], minor=True)                     # 로그축 minor 눈금이 라벨과 겹침
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("fare_amount ($, 로그 축)")
    ax1.set_ylabel("현금 비율 (%)")
    ax1.set_title("같은 요금이어도 공항이면 현금 비율이 딴판\n→ is_airport × log_fare 가 필요한 이유",
                  loc="left", fontweight="bold", fontsize=11)
    ax1.legend(fontsize=9, frameon=False)

    # (우) JFK 승차만 — 여기서 거리↔현금이 단조가 아니다.
    #     공항 전체로 그리면 LGA 등이 섞여 이 패턴이 뭉개진다.
    jfk = df[df.PULocationID == 132].copy()
    bins = [0, 2, 5, 10, 14, 18, 25, 300]
    jfk["dbin"] = pd.cut(np.expm1(jfk["log_dist"]), bins)
    g = jfk.groupby("dbin", observed=True)["is_cash"].mean() * 100
    # 마지막 구간은 25~300마일이라 중앙값(162)을 그대로 쓰면 x축이 늘어진다.
    # 등간격 위치에 찍고 라벨로 실제 구간을 표시한다.
    x = np.arange(len(g))
    ax2.plot(x, g.values, "o-", color=C_CASH, lw=2, ms=6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{bins[i]}~{bins[i+1]}" for i in range(len(g))], fontsize=8.5)

    edge = next(i for i, iv in enumerate(g.index) if iv.right == 10)   # 10마일 경계
    ax2.axvline(edge + 0.5, ls="--", lw=1.2, color="gray")
    ax2.annotate("10마일\n정액제 구간 시작", (edge + 0.5, g.max() * 0.55),
                 xytext=(edge + 0.75, g.max() * 0.60), fontsize=9, color="gray")
    ax2.annotate(f"정점 {g.max():.0f}%", (int(np.argmax(g.values)), g.max()),
                 textcoords="offset points", xytext=(6, 8), fontsize=9, color=C_CASH)
    ax2.set_ylim(0, g.max() * 1.2)
    ax2.set_xlabel("trip_distance (mile)")
    ax2.set_ylabel("현금 비율 (%)")
    ax2.set_title("JFK 승차: 거리↔현금이 단조가 아니다 (증가 후 급락)\n"
                  "→ is_flat70 × log_dist 가 필요한 이유",
                  loc="left", fontweight="bold", fontsize=11)

    fig.tight_layout()
    save(fig, "04_interaction.png")


# ──────────────────────────────────────────────────────────────────────
# 5. 추천 조합 계수
# ──────────────────────────────────────────────────────────────────────
def fig5_coefficients() -> None:
    if not need(TABLES / "04_coefficients.csv"):
        return
    t = pd.read_csv(TABLES / "04_coefficients.csv")
    t = t[t["feature"] != "const"].sort_values("coef")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = [C_CASH if c > 0 else C_CARD for c in t["coef"]]
    bars = ax.barh(t["feature"], t["coef"], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)

    lim = t["coef"].abs().max() * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("표준화 계수  (양수 → 현금 쪽, 음수 → 카드 쪽)")
    ax.set_title("추천 조합 로지스틱 회귀 계수\n상위 4개가 전부 공항·정액제 관련",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "05_coefficients.png")


# ──────────────────────────────────────────────────────────────────────
# 6. 지리 인코딩 — 왜 정수로 넣으면 안 되는가
# ──────────────────────────────────────────────────────────────────────
def fig6_geo_encoding() -> None:
    if not need(SUBSETS / DIAG):
        return
    df = pd.read_parquet(SUBSETS / DIAG)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    for col, lab, col_c in [("pu_te", "승차지역", C_CARD),
                            ("do_te", "하차지역", C_CASH),
                            ("route_te", "경로(PU×DO)", "#72B7B2")]:
        sns.kdeplot(df[col] * 100, ax=ax1, label=lab, fill=True, alpha=0.3,
                    lw=1.6, color=col_c, warn_singular=False)
    ax1.axvline(df["is_cash"].mean() * 100, ls="--", lw=1.2, color="gray")
    ax1.text(df["is_cash"].mean() * 100, ax1.get_ylim()[1] * 0.92,
             f" 전체 평균 {df['is_cash'].mean()*100:.1f}%", fontsize=8.5, color="gray")
    ax1.set_xlim(0, 45)                       # 45% 위는 꼬리라 형태만 가림
    ax1.set_xlabel("타겟 인코딩 값 = 해당 지역/경로의 현금 비율 (%)")
    ax1.set_ylabel("밀도")
    ax1.set_title("타겟 인코딩이 만드는 분산\n퍼져 있을수록 구분력이 있다",
                  loc="left", fontweight="bold", fontsize=11)
    ax1.legend(fontsize=9, frameon=False)

    s = df.sample(min(40_000, len(df)), random_state=RS)
    ax2.scatter(s["pu_te"] * 100, s["do_te"] * 100, s=3, alpha=0.06, color=C_CARD)
    ax2.set_xlabel("승차지역 현금비율 (%)")
    ax2.set_ylabel("하차지역 현금비율 (%)")
    r = df["pu_te"].corr(df["do_te"])
    ax2.set_title(f"승차 vs 하차 인코딩 (r={r:.3f})\n"
                  f"{'겹치지 않아 둘 다 넣을 가치가 있다' if abs(r) < 0.5 else '정보가 겹친다'}",
                  loc="left", fontweight="bold", fontsize=11)

    fig.tight_layout()
    save(fig, "06_geo_encoding.png")


# ──────────────────────────────────────────────────────────────────────
# 그림은 두 종류다.
#   set    — 최종 세트가 어떻게 생겼는지. 인계 대상이 바로 보는 것
#   why    — 세트를 왜 그렇게 짰는지에 대한 근거. 인계서 본문이 인용
FIGS = {
    "2": ("세트별 피처 분포", "set", fig2_subset_distributions),
    "5": ("interpretable 계수", "set", fig5_coefficients),
    "1": ("블록 기여도", "why", fig1_block_contribution),
    "3": ("원본 피처 효과크기", "why", fig3_effect_sizes),
    "4": ("상호작용 구조", "why", fig4_interaction),
    "6": ("지리 인코딩", "why", fig6_geo_encoding),
}


def main() -> None:
    args = [a.lower() for a in sys.argv[1:]]
    if not args:
        want = list(FIGS)
    elif "set" in args or "why" in args:
        want = [k for k, v in FIGS.items() if v[1] in args]
    else:
        want = [k for k in args if k in FIGS]
        if not want:
            sys.exit(f"알 수 없는 인자: {args}  (1~6 / set / why)")

    setup_style()
    print(f"출력 → {FIGURES}")
    for k in want:
        name, kind, fn = FIGS[k]
        print(f"\n[{k}] {name}  ({kind})")
        fn()


if __name__ == "__main__":
    main()
