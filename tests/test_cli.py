from reporting.__main__ import main


def test_cli_strict_returns_two_for_pending_inputs(capsys):
    exit_code = main(["--strict"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Strict mode requires complete stages" in captured.err


def test_cli_check_reports_generated_paths(capsys):
    exit_code = main(["--check"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "checked: report.md" in captured.out
    assert "checked: report.html" in captured.out
