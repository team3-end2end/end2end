"""노트북과 파이프라인이 단계별 결과를 안전하게 저장하도록 돕는다."""

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
    """한 단계 결과를 검증한 뒤 대응하는 JSON 파일에 원자적으로 저장한다.

    단계마다 파일을 분리해 팀원 간 수정 충돌을 줄인다. 임시 파일을 완전히
    기록한 뒤 교체하므로 실행이 중단돼도 기존 정상 결과는 보존된다.

    Args:
        stage: 저장할 단계 이름(``run``, ``eda``, ``preprocessing``, ``model``, ``evaluation``).
        data: 단계별 스키마를 만족하는 실제 결과 데이터.
        status: ``pending``, ``complete``, ``failed`` 중 하나.
        error: 실패 상태일 때 보고서에 표시할 메시지와 상세 내용.
        output_dir: 단계별 JSON을 저장할 디렉터리.
        generated_at: 재현 가능한 테스트가 필요할 때 지정하는 ISO 8601 시각.

    Returns:
        최종 저장된 JSON 파일 경로.
    """
    if stage not in STAGES:
        raise ContractError(f"Unknown stage {stage!r}; expected one of {STAGES}")

    # 모든 단계가 동일한 로더를 사용할 수 있도록 공통 메타데이터로 감싼다.
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
    # 대상 파일과 같은 디렉터리에 임시 파일을 만들어 os.replace의 원자성을 보장한다.
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{stage}-", suffix=".json.tmp", dir=destination_dir
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            os.fchmod(file.fileno(), 0o644)
            json.dump(envelope, file, ensure_ascii=False, indent=2)
            file.write("\n")
            # 프로세스가 종료되기 전에 운영체제 버퍼까지 밀어 넣는다.
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination
