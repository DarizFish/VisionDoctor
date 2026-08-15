from __future__ import annotations

import sys
from pathlib import Path

from visiondoctor.web import cli


def test_web_entrypoint_launches_streamlit_cli(
    monkeypatch,
) -> None:
    captured: list[list[str]] = []

    def fake_main() -> None:
        captured.append(list(sys.argv))

    monkeypatch.setattr("streamlit.web.cli.main", fake_main)
    monkeypatch.setattr(sys, "argv", ["visiondoctor-web", "--server.port", "8600"])

    cli.main()

    assert len(captured) == 1
    arguments = captured[0]
    assert arguments[:2] == ["streamlit", "run"]
    assert Path(arguments[2]).name == "app.py"
    assert arguments[3:11] == [
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    assert arguments[-2:] == ["--server.port", "8600"]
