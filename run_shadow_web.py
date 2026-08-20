"""Serve the validation-gated V27 shadow deployment on localhost."""

from __future__ import annotations

import os

from deployment_fusion import shadow_status
from web_viewer import app


def main() -> int:
    status = shadow_status()
    if status["state"] != "shadow" or not status["every_output_requires_yolo11"]:
        raise RuntimeError("V27 shadow artifacts failed startup validation")
    port = int(os.environ.get("YOLO_SHADOW_WEB_PORT", "5055"))
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
