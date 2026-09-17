"""Keep one interpreter warm for a project workspace's own programs.

Starting Python and importing NumPy, Pillow and YAML took over a second per
program on this host, while the perception algorithm itself runs in under a
tenth of that.  Each request names a module of the project and its arguments;
the module runs as ``__main__``, exactly as ``python -m`` would, in a fresh module
namespace.  Only the interpreter and third-party imports are reused, and the
controller starts a new worker whenever the project's code changes.
"""

from __future__ import annotations

import contextlib
import io
import json
import runpy
import sys
import traceback


def main() -> int:
    import numpy  # noqa: F401 - warmed once for every later request
    import yaml  # noqa: F401
    from PIL import Image  # noqa: F401

    reply = sys.stdout
    for line in sys.stdin:
        request = json.loads(line)
        captured_out, captured_err = io.StringIO(), io.StringIO()
        sys.argv = [request["module"], *request["args"]]
        code = 0
        try:
            with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
                runpy.run_module(request["module"], run_name="__main__", alter_sys=True)
        except SystemExit as exc:
            if isinstance(exc.code, int):
                code = exc.code
            elif exc.code is not None:
                code = 1
                captured_err.write(str(exc.code))
        except BaseException:  # noqa: BLE001 - the project's failure is reported, not fatal
            code = 1
            captured_err.write(traceback.format_exc())
        reply.write(json.dumps({
            "code": code,
            "stdout": captured_out.getvalue()[-4000:],
            "stderr": captured_err.getvalue()[-4000:],
        }) + "\n")
        reply.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
