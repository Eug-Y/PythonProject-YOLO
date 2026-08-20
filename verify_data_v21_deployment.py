"""Run V21 post-deployment inference, Flask, and rollback regression checks."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import verify_data_v20_deployment as _implementation


ROOT = Path(__file__).resolve().parent

_OVERRIDES = {
    "DEPLOYMENT_AUDIT": ROOT / "logs" / "data_v21_deployment.json",
    "EXPERIMENT_LABEL": "V21",
    "RELEASE_VERSION": "v21_sparse_context_half_v1",
}


@contextmanager
def _configured():
    previous = {name: getattr(_implementation, name) for name in _OVERRIDES}
    try:
        for name, value in _OVERRIDES.items():
            setattr(_implementation, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_implementation, name, value)


def validate_deployment_config(payload: dict[str, object]) -> Path:
    with _configured():
        return _implementation.validate_deployment_config(payload)


def main() -> int:
    with _configured():
        return _implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
