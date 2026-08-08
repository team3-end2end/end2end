"""제출용 단일 HTML을 만든다 (브라우저에서 A4 PDF로 인쇄).

final_report.md + submission_appendix.md를 합쳐 HTML로 변환하고,
참조된 이미지를 전부 base64로 파일 안에 넣어 어디서 열어도 그림이 보이게 한다.

실행: python reports/build_submission_html.py
산출: reports/submission.html
"""
import base64
import mimetypes
import re
from pathlib import Path

import markdown

REPORTS_DIR = Path(__file__).resolve().parent
SOURCES = [REPORTS_DIR / "final_report.md", REPORTS_DIR / "submission_appendix.md"]
OUTPUT = REPORTS_DIR / "submission.html"

# A4 인쇄 기준 스타일. 표·이미지가 페이지 경계에서 잘리지 않도록 제어한다.
STYLE = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
  font-size: 10.5pt; line-height: 1.65; color: #1a1a1a;
  max-width: 178mm; margin: 0 auto; padding: 10mm;
}
h1 { font-size: 19pt; border-bottom: 2.5px solid #222; padding-bottom: 6px; margin: 0 0 18px; }
h2 { font-size: 14pt; margin: 26px 0 10px; padding-top: 6px; border-top: 1px solid #ddd; }
h3 { font-size: 11.5pt; margin: 18px 0 8px; color: #333; }
h2, h3 { break-after: avoid; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table {
  border-collapse: collapse; width: 100%; margin: 10px 0 16px;
  font-size: 9.5pt; break-inside: avoid; page-break-inside: avoid;
}
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; }
th { background: #f2f2f2; font-weight: 600; }
td:not(:first-child) { text-align: right; }
code {
  background: #f4f4f4; padding: 1px 4px; border-radius: 3px;
  font-family: "SF Mono", Menlo, monospace; font-size: 9pt;
}
blockquote {
  margin: 10px 0; padding: 6px 12px; border-left: 3px solid #bbb;
  background: #fafafa; color: #444; font-size: 9.5pt;
}
img {
  max-width: 100%; height: auto; display: block; margin: 10px auto;
  border: 1px solid #ddd; border-radius: 3px;
  break-inside: avoid; page-break-inside: avoid;
}
ul, ol { padding-left: 20px; }
li { margin: 3px 0; }
hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
em { color: #666; font-size: 9.5pt; }
@media print { body { padding: 0; max-width: none; } }
"""


def embed_images(html: str) -> str:
    """<img src="상대경로">를 base64 data URI로 바꿔 단일 파일로 만든다."""

    def replace(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = (REPORTS_DIR / src).resolve()
        if not path.is_file():
            print(f"  [경고] 이미지 없음: {src}")
            return match.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        print(f"  포함: {src} ({path.stat().st_size:,}B)")
        return f'src="data:{mime};base64,{encoded}"'

    return re.sub(r'src="([^"]+)"', replace, html)


def main() -> None:
    parts = []
    for source in SOURCES:
        if not source.is_file():
            raise SystemExit(f"원본 문서가 없습니다: {source}")
        parts.append(source.read_text(encoding="utf-8"))
    body = markdown.markdown(
        "\n\n".join(parts), extensions=["tables", "fenced_code", "sane_lists"]
    )

    print("이미지 포함 중...")
    body = embed_images(body)

    OUTPUT.write_text(
        "<!doctype html>\n<html lang=\"ko\">\n<head>\n"
        '<meta charset="utf-8">\n'
        "<title>NYC Yellow Taxi 결제수단 예측 보고서</title>\n"
        f"<style>{STYLE}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n",
        encoding="utf-8",
    )
    size = OUTPUT.stat().st_size
    if size == 0:
        raise RuntimeError(f"생성된 HTML이 비어 있습니다: {OUTPUT}")
    print(f"\n완료: {OUTPUT} ({size:,}B)")
    print("브라우저로 열어 Cmd+P → 대상을 'PDF로 저장'으로 인쇄하세요.")


if __name__ == "__main__":
    main()
