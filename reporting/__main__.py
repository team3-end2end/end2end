"""보고서 생성기를 터미널과 CI에서 실행하기 위한 명령행 진입점."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contracts import ContractError
from .generator import (
    DEFAULT_INPUT_DIR,
    EXAMPLE_INPUT_DIR,
    EXAMPLE_OUTPUT_DIR,
    generate_reports,
)


def build_parser() -> argparse.ArgumentParser:
    """지원하는 CLI 옵션과 도움말을 정의한다."""
    parser = argparse.ArgumentParser(description="Markdown과 HTML 파이프라인 보고서를 생성합니다")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="require every stage to be complete")
    parser.add_argument("--check", action="store_true", help="fail when committed reports are stale")
    parser.add_argument("--example", action="store_true", help="render tracked sample inputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 인자를 해석해 생성기를 실행하고 자동화용 종료 코드를 반환한다."""
    args = build_parser().parse_args(argv)
    # --example은 별도 경로를 기본값으로 사용하되 사용자가 지정한 경로를 우선한다.
    input_dir = args.input_dir or (EXAMPLE_INPUT_DIR if args.example else DEFAULT_INPUT_DIR)
    output_dir = args.output_dir or (EXAMPLE_OUTPUT_DIR if args.example else Path("."))
    try:
        reports = generate_reports(
            input_dir=input_dir,
            output_dir=output_dir,
            strict=args.strict,
            check=args.check,
            example=args.example,
        )
    except ContractError as exc:
        # 입력/최신성 오류는 CI가 구분할 수 있도록 일관되게 종료 코드 2를 사용한다.
        print(f"report generation failed: {exc}", file=sys.stderr)
        return 2
    action = "checked" if args.check else "generated"
    print(f"{action}: {reports.markdown}")
    if reports.html:
        print(f"{action}: {reports.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
