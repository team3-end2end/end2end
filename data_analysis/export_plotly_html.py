"""process.ipynb에 저장된 Plotly 차트를 인터랙티브 HTML 파일로 내보낸다.

노트북 실행 결과(셀 출력에 저장된 plotly JSON)를 그대로 사용하므로
데이터를 다시 읽거나 셀을 재실행하지 않는다.

실행: python data_analysis/export_plotly_html.py
산출: data_analysis/outputs/figures/plotly_01.html ~ plotly_NN.html
"""
import json
from pathlib import Path

import plotly.io as pio

NOTEBOOK = Path(__file__).resolve().parent / "process.ipynb"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "figures"


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            fig_json = output.get("data", {}).get("application/vnd.plotly.v1+json")
            if not fig_json:
                continue
            count += 1
            fig = pio.from_json(json.dumps(fig_json))
            path = OUT_DIR / f"plotly_{count:02d}.html"
            fig.write_html(path)
            # 저장을 믿지 않는다 — 파일 존재와 크기를 확인한다
            size = path.stat().st_size
            if size == 0:
                raise RuntimeError(f"저장된 HTML이 비어 있습니다: {path}")
            title = (fig.layout.title.text or "(제목 없음)") if fig.layout.title else "(제목 없음)"
            print(f"[export] {path.name} ({size:,}B) — {title}")

    if count == 0:
        raise SystemExit("노트북에 Plotly 출력이 없습니다. process.ipynb를 먼저 실행하세요.")
    print(f"완료: 차트 {count}개 → {OUT_DIR}")


if __name__ == "__main__":
    main()
