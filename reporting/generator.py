"""Load validated pipeline results and render human-readable reports."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

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
    _validate_cross_stage(results)
    return results


def _validate_cross_stage(results: dict[str, dict[str, Any]]) -> None:
    model = results["model"]
    evaluation = results["evaluation"]
    if model["status"] != "complete" or evaluation["status"] != "complete":
        return
    candidate_ids = {item["id"] for item in model["data"]["candidates"]}
    result_ids = {item["model_id"] for item in evaluation["data"]["model_results"]}
    unknown = sorted(result_ids - candidate_ids)
    if unknown:
        raise ContractError(f"Evaluation references unknown model candidates: {unknown}")


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
        autoescape=template_name.endswith(".html.j2"),
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
        item["data_uri"] = _data_uri(source) if item["exists"] else None
        normalized_figures.append(item)
    if context["eda"]:
        context["eda"] = dict(context["eda"])
        context["eda"]["figures"] = normalized_figures
    return context


def _data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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
            os.fchmod(file.fileno(), 0o644)
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
    """Generate Markdown and standalone HTML reports from stage JSON files."""
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    stages = _load_stages(source_dir, strict=strict)
    context = _context(stages, destination_dir, example=example)
    markdown_name = "report.example.md" if example else "report.md"
    html_name = "report.example.html" if example else "report.html"
    markdown_path = destination_dir / markdown_name
    html_path = destination_dir / html_name
    _write_or_check(markdown_path, _render("report.md.j2", context), check)
    _write_or_check(html_path, _render("report.html.j2", context), check)
    return GeneratedReports(markdown=markdown_path, html=html_path)
