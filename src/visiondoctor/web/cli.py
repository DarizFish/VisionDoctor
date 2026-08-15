from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch the packaged Streamlit app through Streamlit's supported CLI."""

    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("app.py").resolve()
    user_arguments = sys.argv[1:]
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        *user_arguments,
    ]
    streamlit_cli.main()


if __name__ == "__main__":
    main()
