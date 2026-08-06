import json
from pathlib import Path

import pytest

from reporting.contracts import ContractError, STAGES
from reporting.generator import generate_reports


def _pending_inputs(directory: Path):
    directory.mkdir()
    for stage in STAGES:
        (directory / f"{stage}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": stage,
                    "status": "pending",
                    "generated_at": None,
                    "data": {},
                    "error": None,
                }
            ),
            encoding="utf-8",
        )


def test_pending_inputs_render_markdown(tmp_path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    _pending_inputs(inputs)

    result = generate_reports(inputs, outputs)

    content = result.markdown.read_text(encoding="utf-8")
    assert content.count("준비 중") >= 5
    assert "직접 수정하지 말고" in content


def test_strict_mode_rejects_pending_stage(tmp_path):
    inputs = tmp_path / "inputs"
    _pending_inputs(inputs)
    with pytest.raises(ContractError, match="Strict mode"):
        generate_reports(inputs, tmp_path / "outputs", strict=True)


def test_check_detects_stale_markdown(tmp_path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    _pending_inputs(inputs)
    generate_reports(inputs, outputs)
    generate_reports(inputs, outputs, check=True)
    (outputs / "report.md").write_text("stale", encoding="utf-8")
    with pytest.raises(ContractError, match="out of date"):
        generate_reports(inputs, outputs, check=True)
