"""검증된 단계별 결과로 기술용 Markdown과 배포용 HTML을 생성한다."""

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
    """한 번의 생성 작업에서 만들어진 산출물 경로."""

    markdown: Path
    html: Path | None = None


def _load_stages(input_dir: Path, strict: bool) -> dict[str, dict[str, Any]]:
    """예상하는 다섯 단계 JSON을 읽고 개별·교차 검증 준비를 마친다."""
    results: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        path = input_dir / f"{stage}.json"
        if not path.exists():
            raise ContractError(f"Missing stage result: {path}")
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ContractError(f"Invalid JSON in {path}: {exc}") from exc
        # 파일명과 내부 stage가 다르면 잘못된 담당 파일에 결과를 쓴 것이므로 거부한다.
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
    """평가 결과가 실제 모델 단계에 등록된 후보만 참조하는지 확인한다."""
    model = results["model"]
    evaluation = results["evaluation"]
    if model["status"] != "complete" or evaluation["status"] != "complete":
        return
    model_id = model["data"]["model"]["id"]
    evaluation_model_id = evaluation["data"]["model_id"]
    if model_id != evaluation_model_id:
        raise ContractError(
            f"Evaluation model_id {evaluation_model_id!r} does not match model {model_id!r}"
        )

    preprocessing = results["preprocessing"]
    if preprocessing["status"] == "complete":
        split = model["data"]["split"]
        split_total = sum(
            split[key] for key in ("train_samples", "validation_samples", "test_samples")
        )
        output_rows = preprocessing["data"]["output_shape"]["rows"]
        if split_total != output_rows:
            raise ContractError(
                f"Model split total {split_total} does not match preprocessing rows {output_rows}"
            )

    test_samples = model["data"]["split"]["test_samples"]
    evaluated_samples = sum(item["support"] for item in evaluation["data"]["class_metrics"])
    if evaluated_samples != test_samples:
        raise ContractError(
            f"Evaluation support total {evaluated_samples} does not match test samples {test_samples}"
        )


def _display(value: Any, fallback: str = "준비 중") -> Any:
    """일반 값의 빈 상태를 보고서용 문구로 바꾼다."""
    if value is None or value == "":
        return fallback
    return value


def _integer(value: Any) -> str:
    """정수를 천 단위 구분 기호가 있는 문자열로 표시한다."""
    return "준비 중" if value is None else f"{int(value):,}"


def _percent(value: Any, digits: int = 1) -> str:
    """0~1 비율을 지정한 소수 자릿수의 백분율로 표시한다."""
    return "준비 중" if value is None else f"{float(value) * 100:.{digits}f}%"


def _seconds(value: Any) -> str:
    """학습 시간을 초 단위 문자열로 표시한다."""
    return "준비 중" if value is None else f"{float(value):,.2f}초"


def _environment(template_name: str) -> Environment:
    """공통 포맷터와 출력 형식별 이스케이프 정책을 가진 Jinja 환경을 만든다."""
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        # HTML은 사용자 입력을 이스케이프하고 Markdown은 원문 표기를 유지한다.
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
    """원본 단계 결과를 두 템플릿이 공통으로 읽는 표시용 데이터로 변환한다."""
    context: dict[str, Any] = {"stages": stages, "is_sample": example}
    for stage, result in stages.items():
        context[stage] = result["data"] if result["status"] == "complete" else {}

    # Markdown에는 상대 경로를, 단일 HTML에는 같은 이미지를 data URI로 제공한다.
    for stage_name in ("eda", "evaluation"):
        figures = context[stage_name].get("figures", [])
        normalized_figures = []
        for figure in figures:
            item = dict(figure)
            source = Path(item["path"])
            item["exists"] = source.is_file()
            item["markdown_path"] = os.path.relpath(source, output_dir)
            item["data_uri"] = _data_uri(source) if item["exists"] else None
            normalized_figures.append(item)
        if context[stage_name]:
            context[stage_name] = dict(context[stage_name])
            context[stage_name]["figures"] = normalized_figures

    # 혼동행렬 PNG는 행렬 수치와 함께 전달되므로 일반 평가 그림과 별도로 정규화한다.
    if context["evaluation"]:
        confusion_matrix = dict(context["evaluation"]["confusion_matrix"])
        figure_path = confusion_matrix.get("figure_path")
        if figure_path:
            source = Path(figure_path)
            confusion_matrix["figure"] = {
                "path": figure_path,
                "exists": source.is_file(),
                "markdown_path": os.path.relpath(source, output_dir),
                "data_uri": _data_uri(source) if source.is_file() else None,
            }
        else:
            confusion_matrix["figure"] = None
        context["evaluation"] = dict(context["evaluation"])
        context["evaluation"]["confusion_matrix"] = confusion_matrix
    return context


def _data_uri(path: Path) -> str:
    """로컬 이미지 파일을 단일 HTML에 포함할 Base64 data URI로 변환한다."""
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _render(template_name: str, context: dict[str, Any]) -> str:
    """선택한 템플릿을 표시용 데이터로 렌더링한다."""
    return _environment(template_name).get_template(template_name).render(**context)


def _write_or_check(path: Path, content: str, check: bool) -> None:
    """결과를 원자적으로 저장하거나 기존 파일이 최신인지 비교한다."""
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise ContractError(f"Generated report is out of date: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # 두 보고서 중 하나가 기록 도중 잘린 상태로 남지 않도록 임시 파일을 교체한다.
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
    """단계별 JSON에서 Markdown과 독립 실행형 HTML 보고서를 함께 생성한다.

    ``strict=False``는 진행 중인 단계를 "준비 중"으로 렌더링한다.
    최종 파이프라인과 CI에서는 ``strict=True``로 모든 단계 완료를 강제한다.
    ``check=True``는 파일을 쓰지 않고 커밋된 산출물의 최신 여부만 검사한다.
    """
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    stages = _load_stages(source_dir, strict=strict)
    context = _context(stages, destination_dir, example=example)
    # 예시와 운영 산출물은 이름과 경로를 분리해 실제 결과로 오인하지 않게 한다.
    markdown_name = "report.example.md" if example else "report.md"
    html_name = "report.example.html" if example else "report.html"
    markdown_path = destination_dir / markdown_name
    html_path = destination_dir / html_name
    _write_or_check(markdown_path, _render("report.md.j2", context), check)
    _write_or_check(html_path, _render("report.html.j2", context), check)
    return GeneratedReports(markdown=markdown_path, html=html_path)
