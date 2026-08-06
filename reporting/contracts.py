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
        return
    details = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        details.append(f"{location}: {error.message}")
    raise ContractError("Invalid report stage result:\n- " + "\n- ".join(details))
