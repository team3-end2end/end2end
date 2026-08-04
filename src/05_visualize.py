"""
Step 5 — 시각화

`cleaned.parquet`을 읽기만 하며 데이터를 바꾸지 않는다.
Step 3(EDA)에서 숫자로 확인한 내용을 그림으로 보여주는 단계다.

과제 요구사항: Seaborn 정적 차트 1개 이상 + Plotly 인터랙티브 차트 1개 이상,
두 차트 모두 제목·축 레이블 포함.

산출물:
  - outputs/figures/seaborn_distance_by_payment.png
  - outputs/figures/plotly_hourly_payment_mix.html
  - outputs/results/figures.json   차트 경로·설명 (Step 7의 report.md 생성용)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # GUI 없이 파일로만 저장

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

# --- 경로 상수 ---
ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "processed" / "cleaned.parquet"
FIG_DIR = ROOT / "outputs" / "figures"
RESULT_DIR = ROOT / "outputs" / "results"

# --- 설정 ---
PLOT_SAMPLE_N = 200_000  # 389만 행을 전부 그리면 느리고 가독성도 나쁘다
RANDOM_STATE = 42
TARGET = "payment_label"


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


def load_cleaned() -> pd.DataFrame:
    """Step 2의 산출물을 읽는다. 없으면 명확한 에러로 멈춘다."""
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"정제 데이터가 없습니다: {CLEAN_PATH}\n먼저 `python src/02_preprocess.py`를 실행하세요."
        )
    return pd.read_parquet(CLEAN_PATH)


def make_seaborn_chart(df: pd.DataFrame) -> Path:
    """Seaborn 정적 차트 — 결제수단별 이동거리 분포 (그룹 비교).

    극단값 때문에 상자가 눌려 보이지 않도록 이상치 표시를 끄고 y축을 제한한다.
    """
    sample = df.sample(n=min(PLOT_SAMPLE_N, len(df)), random_state=RANDOM_STATE)
    order = sample[TARGET].value_counts().index.tolist()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=sample, x=TARGET, y="trip_distance", order=order, showfliers=False, ax=ax)
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


def make_plotly_chart(df: pd.DataFrame) -> Path:
    """Plotly 인터랙티브 차트 — 시간대별 결제수단 구성비.

    건수가 아니라 비율로 그린다. 시간대마다 전체 운행량이 크게 다르므로
    건수로 그리면 '새벽엔 현금이 적다'가 아니라 '새벽엔 운행이 적다'만 보인다.
    """
    counts = df.groupby(["pickup_hour", TARGET]).size().rename("건수").reset_index()
    counts["비율"] = (
        counts["건수"] / counts.groupby("pickup_hour")["건수"].transform("sum") * 100
    ).round(2)

    fig = px.bar(
        counts, x="pickup_hour", y="비율", color=TARGET,
        title="시간대별 결제수단 구성비",
        labels={"pickup_hour": "승차 시각 (시)", "비율": "구성비 (%)", TARGET: "결제수단"},
        hover_data={"건수": ":,"},
    )
    fig.update_layout(
        barmode="stack",
        xaxis=dict(dtick=1),
        yaxis_range=[0, 100],
        font=dict(family="AppleGothic, Malgun Gothic, sans-serif"),
    )

    path = FIG_DIR / "plotly_hourly_payment_mix.html"
    fig.write_html(path)
    return path


def main() -> None:
    for d in (FIG_DIR, RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    font = setup_korean_font()
    print(f"한글 폰트: {font}")

    df = load_cleaned()
    print(f"정제 데이터 로딩: {len(df):,}행\n")

    seaborn_path = make_seaborn_chart(df)
    plotly_path = make_plotly_chart(df)

    result = {
        "figures": [
            {
                "종류": "Seaborn (정적)",
                "파일": str(seaborn_path.relative_to(ROOT)),
                "내용": "결제수단별 이동거리 분포 (boxplot)",
                "비고": f"표본 {min(PLOT_SAMPLE_N, len(df)):,}건, 이상치 표시 제외, y축 0~12마일",
            },
            {
                "종류": "Plotly (인터랙티브)",
                "파일": str(plotly_path.relative_to(ROOT)),
                "내용": "시간대별 결제수단 구성비 (stacked bar)",
                "비고": "건수가 아닌 비율 — 시간대별 운행량 차이를 보정하기 위함",
            },
        ],
        "korean_font": font,
    }
    (RESULT_DIR / "figures.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("저장 완료:")
    for f in result["figures"]:
        print(f"  {f['파일']}  — {f['내용']}")
    print(f"  {(RESULT_DIR / 'figures.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
