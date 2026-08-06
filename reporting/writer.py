"""Safe writer used by notebooks and pipeline stages."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import SCHEMA_VERSION, STAGES, ContractError, validate_stage_result


def write_stage_result(
    stage: str,
    data: Mapping[str, Any] | None = None,
    *,
    status: str = "complete",
    error: Mapping[str, Any] | None = None,
    output_dir: str | Path = "reporting/data",
    generated_at: str | None = None,
) -> Path:
    """Validate and atomically write one pipeline stage result.

    Each stage owns one JSON file, so independent team members never need to
    modify the same result document.
    """
    if stage not in STAGES:
        raise ContractError(f"Unknown stage {stage!r}; expected one of {STAGES}")

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": dict(data or {}),
        "error": dict(error) if error is not None else None,
    }
    validate_stage_result(envelope)

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{stage}.json"
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{stage}-", suffix=".json.tmp", dir=destination_dir
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(envelope, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination
