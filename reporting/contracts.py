"""Versioned input contracts for report stages."""

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
    """Raised when a stage result does not satisfy the report contract."""


def _validator() -> Draft202012Validator:
    with SCHEMA_PATH.open(encoding="utf-8") as file:
        schema = json.load(file)
    return Draft202012Validator(schema)


def validate_stage_result(result: Mapping[str, Any]) -> None:
    """Validate a complete stage envelope and raise one readable error."""
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
    if result["status"] != "complete":
        return
    stage = result["stage"]
    data = result["data"]
    problems: list[str] = []

    if stage == "eda" and data["class_distribution"]:
        total_ratio = sum(item["ratio"] for item in data["class_distribution"])
        if not 0.999 <= total_ratio <= 1.001:
            problems.append(f"class_distribution ratios must sum to 1 (got {total_ratio:.6f})")

    if stage == "preprocessing":
        for index, item in enumerate(data["filters"]):
            if item["before_rows"] - item["after_rows"] != item["removed_rows"]:
                problems.append(f"filters[{index}] row counts are inconsistent")
        filters = data["filters"]
        for index in range(1, len(filters)):
            if filters[index - 1]["after_rows"] != filters[index]["before_rows"]:
                problems.append(f"filters[{index}] does not continue from the previous filter")

    if stage == "model":
        identifiers = [item["id"] for item in data["candidates"]]
        if len(identifiers) != len(set(identifiers)):
            problems.append("model candidate ids must be unique")

    if stage == "evaluation":
        identifiers = [item["model_id"] for item in data["model_results"]]
        if len(identifiers) != len(set(identifiers)):
            problems.append("model result ids must be unique")
        if data["selected_model_id"] not in identifiers:
            problems.append("selected_model_id must reference a model result")
        labels = data["confusion_matrix"]["labels"]
        values = data["confusion_matrix"]["values"]
        if len(values) != len(labels) or any(len(row) != len(labels) for row in values):
            problems.append("confusion_matrix values must be square and match labels")
        metric_labels = [item["label"] for item in data["class_metrics"]]
        if metric_labels != labels:
            problems.append("class_metrics labels must match confusion_matrix labels and order")

    if problems:
        raise ContractError("Invalid report stage result:\n- " + "\n- ".join(problems))
