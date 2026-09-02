"""Confirm that the container can connect to the WSLg X11 display."""

from __future__ import annotations

import ctypes
import json
import sys


def main() -> int:
    try:
        x11 = ctypes.CDLL("libX11.so.6")
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.restype = ctypes.c_int
        display = x11.XOpenDisplay(None)
        if not display:
            raise RuntimeError("XOpenDisplay returned no display")
        x11.XCloseDisplay(display)
    except Exception as exc:
        print("PICK_CELL_X11=" + json.dumps({"connected": False, "error": str(exc)}))
        return 1
    print("PICK_CELL_X11=" + json.dumps({"connected": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
