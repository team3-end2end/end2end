"""단계별 보고서 입력이 약속된 형식과 의미를 지키는지 검증한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

SCHEMA_VERSION = 1
STAGES = ("run", "eda", "preprocessing", "model", "evaluation")
STATUSES = ("pending", "complete", "failed")
PACKAGE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = PACKAGE_DIR / "schemas" / "stage-v1.schema.json"


class ContractError(ValueError):
    """보고서 입력 계약을 만족하지 못했을 때 발생하는 예외."""


def _validator() -> Draft202012Validator:
    """현재 스키마 버전에 대응하는 JSON Schema 검증기를 생성한다."""
    with SCHEMA_PATH.open(encoding="utf-8") as file:
        schema = json.load(file)
    return Draft202012Validator(schema)


def validate_stage_result(result: Mapping[str, Any]) -> None:
    """한 단계의 공통 봉투와 데이터를 검증하고 오류를 한 번에 보여준다.

    JSON Schema는 키·타입·필수값을 검사하고, 그 검사가 성공한 뒤
    ``_validate_semantics``가 합계나 참조 관계처럼 스키마만으로 표현하기
    어려운 규칙을 확인한다.
    """
    # 팀원이 여러 필드를 한 번에 수정할 수 있도록 첫 오류에서 멈추지 않고 모은다.
    errors = sorted(_validator().iter_errors(dict(result)), key=lambda error: list(error.path))
    if not errors:
        _validate_semantics(result)
        return
    details = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        details.append(f"{location}: {error.message}")
    raise ContractError("Invalid report stage result:\n- " + "\n- ".join(details))


def _validate_semantics(result: Mapping[str, Any]) -> None:
    """완료된 단계의 숫자 관계와 식별자 일관성을 검증한다."""
    # 준비 중·실패 단계는 값이 완성되지 않았으므로 구조 검증만 수행한다.
    if result["status"] != "complete":
        return
    stage = result["stage"]
    data = result["data"]
    problems: list[str] = []

    if stage == "eda" and data["class_distribution"]:
        # 반올림 오차는 허용하되 클래스 비율 전체는 100%여야 한다.
        total_ratio = sum(item["ratio"] for item in data["class_distribution"])
        if not 0.999 <= total_ratio <= 1.001:
            problems.append(f"class_distribution ratios must sum to 1 (got {total_ratio:.6f})")
        reported_missing = sum(item["count"] for item in data["missing_by_column"])
        if reported_missing != data["missing_cell_count"]:
            problems.append("missing_by_column counts must sum to missing_cell_count")

    if stage == "preprocessing":
        # 각 필터의 제거 수와 다음 필터로 이어지는 행 수가 맞아야 한다.
        for index, item in enumerate(data["filters"]):
            if item["before_rows"] - item["after_rows"] != item["removed_rows"]:
                problems.append(f"filters[{index}] row counts are inconsistent")
        filters = data["filters"]
        for index in range(1, len(filters)):
            if filters[index - 1]["after_rows"] != filters[index]["before_rows"]:
                problems.append(f"filters[{index}] does not continue from the previous filter")

    if stage == "evaluation":
        # 혼동행렬과 클래스별 지표가 같은 클래스 순서와 크기를 사용하는지 확인한다.
        labels = data["confusion_matrix"]["labels"]
        values = data["confusion_matrix"]["values"]
        if len(values) != len(labels) or any(len(row) != len(labels) for row in values):
            problems.append("confusion_matrix values must be square and match labels")
        metric_labels = [item["label"] for item in data["class_metrics"]]
        if metric_labels != labels:
            problems.append("class_metrics labels must match confusion_matrix labels and order")
        row_totals = [sum(row) for row in values]
        supports = [item["support"] for item in data["class_metrics"]]
        if len(row_totals) == len(supports) and row_totals != supports:
            problems.append("confusion_matrix row totals must match class_metrics support")

    if problems:
        raise ContractError("Invalid report stage result:\n- " + "\n- ".join(problems))
