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


def test_preprocessing_filter_counts_must_be_consistent():
    with pytest.raises(ContractError, match="row counts are inconsistent"):
        validate_stage_result(
            {
                "schema_version": 1,
                "stage": "preprocessing",
                "status": "complete",
                "generated_at": None,
                "error": None,
                "data": {
                    "input_shape": {"rows": 10, "columns": 2},
                    "output_shape": {"rows": 7, "columns": 2},
                    "retention_ratio": 0.7,
                    "filters": [{"name": "bad", "rule": "x", "before_rows": 10, "after_rows": 7, "removed_rows": 2}],
                    "selected_features": ["x"],
                    "excluded_features": [],
                    "derived_features": [],
                    "transformations": [],
                },
            }
        )


def test_evaluation_confusion_matrix_must_match_labels():
    data = {
        "model_id": "m1",
        "metrics": {"accuracy": 0.8, "macro_precision": 0.7, "macro_recall": 0.7, "macro_f1": 0.7, "weighted_f1": 0.8},
        "cross_validation": {"metric": "macro_f1", "mean": 0.7, "std": 0.01, "folds": 5},
        "class_metrics": [
            {"label": "a", "precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 5},
            {"label": "b", "precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 5},
        ],
        "confusion_matrix": {"labels": ["a", "b"], "values": [[4, 1]], "figure_path": None},
        "figures": [],
    }
    with pytest.raises(ContractError, match="must be square"):
        validate_stage_result(
            {"schema_version": 1, "stage": "evaluation", "status": "complete", "generated_at": None, "data": data, "error": None}
        )


def test_interpretation_fields_are_rejected():
    with pytest.raises(ContractError, match="Additional properties"):
        validate_stage_result(
            {
                "schema_version": 1,
                "stage": "eda",
                "status": "complete",
                "generated_at": None,
                "error": None,
                "data": {
                    "numeric_feature_count": 1,
                    "categorical_feature_count": 1,
                    "missing_cell_count": 0,
                    "duplicate_row_count": 0,
                    "missing_by_column": [],
                    "class_distribution": [{"label": "a", "count": 1, "ratio": 1.0}],
                    "figures": [],
                    "findings": [{"description": "human interpretation"}],
                },
            }
        )


def test_missing_column_counts_must_match_total():
    with pytest.raises(ContractError, match="must sum to missing_cell_count"):
        validate_stage_result(
            {
                "schema_version": 1,
                "stage": "eda",
                "status": "complete",
                "generated_at": None,
                "error": None,
                "data": {
                    "numeric_feature_count": 1,
                    "categorical_feature_count": 1,
                    "missing_cell_count": 2,
                    "duplicate_row_count": 0,
                    "missing_by_column": [{"column": "x", "count": 1, "ratio": 0.1}],
                    "class_distribution": [{"label": "a", "count": 10, "ratio": 1.0}],
                    "figures": [],
                },
            }
        )
