"""Run the local production-like web viewer without Flask's debug reloader."""

from __future__ import annotations

import argparse

from web_viewer import app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

