from reporting.__main__ import main


# 파이프라인 결과가 채워진 뒤에는 모든 단계가 complete이므로 strict 모드가 성공해야 한다.
def test_cli_strict_succeeds_with_complete_inputs(capsys):
    exit_code = main(["--strict"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "generated: report.md" in captured.out


def test_cli_check_reports_generated_paths(capsys):
    exit_code = main(["--check"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "checked: report.md" in captured.out
    assert "checked: report.html" in captured.out
