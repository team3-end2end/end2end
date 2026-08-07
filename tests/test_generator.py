import json
import shutil
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
    html = result.html.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "외부" not in html
    assert "준비 중" in html


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


def test_html_has_no_external_runtime_dependencies(tmp_path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    _pending_inputs(inputs)
    result = generate_reports(inputs, outputs)
    html = result.html.read_text(encoding="utf-8")
    assert "<script" not in html
    assert "http://" not in html
    assert "https://" not in html
    assert "@media print" in html


def test_missing_stage_file_is_reported(tmp_path):
    inputs = tmp_path / "inputs"
    _pending_inputs(inputs)
    (inputs / "model.json").unlink()
    with pytest.raises(ContractError, match="Missing stage result"):
        generate_reports(inputs, tmp_path / "outputs")


def test_generation_is_deterministic(tmp_path):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    _pending_inputs(inputs)
    first = generate_reports(inputs, outputs)
    markdown = first.markdown.read_bytes()
    html = first.html.read_bytes()
    second = generate_reports(inputs, outputs)
    assert second.markdown.read_bytes() == markdown
    assert second.html.read_bytes() == html


def test_tracked_example_renders_complete_sample(tmp_path):
    result = generate_reports(
        Path("reporting/examples/data"), tmp_path, strict=True, example=True
    )
    markdown = result.markdown.read_text(encoding="utf-8")
    html = result.html.read_text(encoding="utf-8")
    assert "SAMPLE" in markdown
    assert "random_forest" in markdown
    assert "Macro F1" in markdown
    assert "SAMPLE" in html
    assert "data:image/png;base64," in html
    assert "reporting/examples/data" not in html
    for forbidden in ("선정 근거", "주요 발견", "한계 및 후속 작업", "결론"):
        assert forbidden not in markdown
        assert forbidden not in html


def test_evaluation_must_reference_the_trained_model(tmp_path):
    inputs = tmp_path / "inputs"
    shutil.copytree("reporting/examples/data", inputs)
    evaluation_path = inputs / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["data"]["model_id"] = "different-model"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ContractError, match="does not match model"):
        generate_reports(inputs, tmp_path / "outputs", strict=True)


def test_sample_marker_comes_from_input_data(tmp_path):
    result = generate_reports(Path("reporting/examples/data"), tmp_path, strict=True)
    assert "SAMPLE" in result.markdown.read_text(encoding="utf-8")
    assert "SAMPLE" in result.html.read_text(encoding="utf-8")
