"""Relay one request to the resident cell agent; needs no ROS environment.

The request JSON arrives on stdin.  Each line the agent sends is printed as it
arrives, so the caller can show progress; the process ends after the result.
"""

from __future__ import annotations

import json
import os
import socket
import sys

PORT = int(os.environ.get("PICK_CELL_AGENT_PORT", "18650"))


def main() -> int:
    request = sys.stdin.read().strip()
    try:
        conn = socket.create_connection(("127.0.0.1", PORT), timeout=3.0)
    except OSError as exc:
        print(json.dumps({"type": "unavailable", "error": str(exc)}), flush=True)
        return 2
    with conn:
        conn.settimeout(None)
        conn.sendall(request.encode("utf-8") + b"\n")
        for line in conn.makefile("r", encoding="utf-8"):
            sys.stdout.write(line)
            sys.stdout.flush()
            if json.loads(line).get("type") == "result":
                return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
