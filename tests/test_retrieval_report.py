from __future__ import annotations

import pytest

from agenttrace.workflow.kb import report


def test_report_prints_table_and_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert report.main() == 0
    out = capsys.readouterr().out
    assert "recall@3" in out
    assert "dev" in out and "held-out" in out
    assert "PASS" in out
