"""Day 2 종합실습 진입점 — 이 파일 하나로 전체 산출물이 생성된다.

기본 모드  : 저장된 학습 결과를 재사용해 통계 표·차트 HTML·보고서를 생성한다. (수 분)
--full 모드: Optuna 탐색부터 모델 학습까지 처음부터 전부 실행한 뒤 위를 수행한다. (~40분)

실행: python main.py [--full]
"""
import argparse
import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(title, cmd):
    """하위 단계를 실행하고 실패하면 즉시 멈춘다."""
    print(f"\n=== {title} ===")
    subprocess.run([sys.executable, *cmd], check=True, cwd=ROOT)


def check_outputs(paths):
    """산출물이 실제로 존재하고 비어 있지 않은지 확인한다."""
    print("\n=== 산출물 확인 ===")
    missing = []
    for p in paths:
        full = ROOT / p
        if full.exists() and full.stat().st_size > 0:
            print(f"  ok  {p} ({full.stat().st_size:,}B)")
        else:
            missing.append(p)
            print(f"  없음 {p}")
    if missing:
        raise SystemExit(f"산출물 누락: {missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="Optuna 탐색 + 모델 학습부터 전체 재실행 (~40분)")
    args = parser.parse_args()

    if args.full:
        # 모듈로 import해 실행하므로 GeoEncoder가 pipeline 소속으로 pickle된다
        # (python pipeline.py 직접 실행 시의 __main__ 소속 pickle 문제 회피)
        import pipeline
        pipeline.main()
    elif not glob.glob(str(ROOT / "outputs" / "model_*.joblib")):
        raise SystemExit(
            "저장된 모델(outputs/model_*.joblib)이 없습니다. "
            "python main.py --full 로 학습부터 실행하세요."
        )
    else:
        print("저장된 학습 결과를 재사용합니다. 처음부터 학습하려면: python main.py --full")

    run_step("통계 분석 표 생성 (t-test·효과크기 CSV)", ["data_analysis/feature_analysis.py"])
    run_step("Seaborn 정적 차트 생성 (png)", ["data_analysis/evaluate.py"])
    run_step("Plotly 차트 HTML 내보내기", ["data_analysis/export_plotly_html.py"])
    run_step("보고서 생성 (report.md / report.html)", ["-m", "reporting.generate", "--strict"])

    check_outputs([
        "report.md",
        "report.html",
        "outputs/results.csv",
        "data_analysis/outputs/tables/06_ttest_card_vs_cash.csv",
        "data_analysis/outputs/figures/02_dist_interpretable.png",
        "data_analysis/outputs/figures/06_geo_encoding.png",
        *[f"data_analysis/outputs/figures/plotly_{i:02d}.html" for i in range(1, 6)],
    ])
    print("\n완료. 보고서: report.md / report.html")


if __name__ == "__main__":
    main()
