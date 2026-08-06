"""Load validated pipeline results and render human-readable reports."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from .contracts import STAGES, ContractError, validate_stage_result

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = PACKAGE_DIR / "templates"
DEFAULT_INPUT_DIR = PACKAGE_DIR / "data"
EXAMPLE_INPUT_DIR = PACKAGE_DIR / "examples" / "data"
EXAMPLE_OUTPUT_DIR = PACKAGE_DIR / "examples"
STATUS_LABELS = {"pending": "준비 중", "complete": "완료", "failed": "실패"}


@dataclass(frozen=True)
class GeneratedReports:
    markdown: Path
    html: Path | None = None


def _load_stages(input_dir: Path, strict: bool) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        path = input_dir / f"{stage}.json"
        if not path.exists():
            raise ContractError(f"Missing stage result: {path}")
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ContractError(f"Invalid JSON in {path}: {exc}") from exc
        validate_stage_result(result)
        if result["stage"] != stage:
            raise ContractError(
                f"Stage mismatch in {path}: expected {stage!r}, got {result['stage']!r}"
            )
        if strict and result["status"] != "complete":
            raise ContractError(
                f"Strict mode requires complete stages: {stage} is {result['status']}"
            )
        result["status_label"] = STATUS_LABELS[result["status"]]
        results[stage] = result
    return results


def _display(value: Any, fallback: str = "준비 중") -> Any:
    if value is None or value == "":
        return fallback
    return value


def _integer(value: Any) -> str:
    return "준비 중" if value is None else f"{int(value):,}"


def _percent(value: Any, digits: int = 1) -> str:
    return "준비 중" if value is None else f"{float(value) * 100:.{digits}f}%"


def _seconds(value: Any) -> str:
    return "준비 중" if value is None else f"{float(value):,.2f}초"


def _environment(template_name: str) -> Environment:
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    environment.filters.update(
        display=_display, integer=_integer, percent=_percent, seconds=_seconds
    )
    return environment


def _context(stages: dict[str, dict[str, Any]], output_dir: Path, example: bool) -> dict[str, Any]:
    context: dict[str, Any] = {"stages": stages, "is_sample": example}
    for stage, result in stages.items():
        context[stage] = result["data"] if result["status"] == "complete" else {}

    figures = context["eda"].get("figures", [])
    normalized_figures = []
    for figure in figures:
        item = dict(figure)
        source = Path(item["path"])
        item["exists"] = source.is_file()
        item["markdown_path"] = os.path.relpath(source, output_dir)
        normalized_figures.append(item)
    if context["eda"]:
        context["eda"] = dict(context["eda"])
        context["eda"]["figures"] = normalized_figures
    return context


def _render(template_name: str, context: dict[str, Any]) -> str:
    return _environment(template_name).get_template(template_name).render(**context)


def _write_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise ContractError(f"Generated report is out of date: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def generate_reports(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = ".",
    *,
    strict: bool = False,
    check: bool = False,
    example: bool = False,
) -> GeneratedReports:
    """Generate reports from stage JSON files.

    HTML support is attached by the HTML work unit; the stable return type
    already reserves its path.
    """
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    stages = _load_stages(source_dir, strict=strict)
    context = _context(stages, destination_dir, example=example)
    markdown_name = "report.example.md" if example else "report.md"
    markdown_path = destination_dir / markdown_name
    _write_or_check(markdown_path, _render("report.md.j2", context), check)
    return GeneratedReports(markdown=markdown_path)
