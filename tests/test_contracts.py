import json

import pytest

from reporting.contracts import ContractError, validate_stage_result
from reporting.writer import write_stage_result


def test_pending_stage_accepts_empty_data():
    validate_stage_result(
        {
            "schema_version": 1,
            "stage": "eda",
            "status": "pending",
            "generated_at": None,
            "data": {},
            "error": None,
        }
    )


def test_failed_stage_requires_error_message():
    with pytest.raises(ContractError, match="error"):
        validate_stage_result(
            {
                "schema_version": 1,
                "stage": "model",
                "status": "failed",
                "generated_at": None,
                "data": {},
                "error": None,
            }
        )


def test_complete_stage_requires_stage_fields():
    with pytest.raises(ContractError, match="project_name"):
        validate_stage_result(
            {
                "schema_version": 1,
                "stage": "run",
                "status": "complete",
                "generated_at": None,
                "data": {},
                "error": None,
            }
        )


def test_writer_creates_one_valid_stage_file(tmp_path):
    destination = write_stage_result(
        "run",
        {
            "project_name": "end2end",
            "report_title": "테스트",
            "run_id": "test-run",
            "source_revision": None,
            "dataset": {
                "name": "sample.parquet",
                "path": "data/sample.parquet",
                "period_start": None,
                "period_end": None,
                "shape": {"rows": 10, "columns": 2},
            },
            "target": {
                "name": "label",
                "problem_type": "multiclass_classification",
                "classes": ["a", "b"],
            },
        },
        output_dir=tmp_path,
        generated_at="2026-08-06T00:00:00+00:00",
    )
    result = json.loads(destination.read_text(encoding="utf-8"))
    assert destination == tmp_path / "run.json"
    assert result["status"] == "complete"
    validate_stage_result(result)


def test_writer_rejects_unknown_stage(tmp_path):
    with pytest.raises(ContractError, match="Unknown stage"):
        write_stage_result("unknown", {}, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
