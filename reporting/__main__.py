"""Command-line entrypoint for report generation."""

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
    parser = argparse.ArgumentParser(description="Generate Markdown and HTML pipeline reports")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="require every stage to be complete")
    parser.add_argument("--check", action="store_true", help="fail when committed reports are stale")
    parser.add_argument("--example", action="store_true", help="render tracked sample inputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        print(f"report generation failed: {exc}", file=sys.stderr)
        return 2
    action = "checked" if args.check else "generated"
    print(f"{action}: {reports.markdown}")
    if reports.html:
        print(f"{action}: {reports.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
