"""Publish an eligible V21 checkpoint as a reversible web deployment release."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import deploy_data_v20 as _implementation


ROOT = Path(__file__).resolve().parent

_OVERRIDES = {
    "DECISION": ROOT / "runs" / "final_evaluation" / "data_v21_locked_test" / "decision.json",
    "RELEASE_LOG": ROOT / "logs" / "data_v21_deployment.json",
    "EXPERIMENT_LABEL": "V21",
    "RELEASE_SERIES": "v21",
    "RELEASE_VERSION": "v21_sparse_context_half_v1",
    "ARTIFACT_PREFIX": "data_v21",
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


def release_name(seed: int) -> str:
    with _configured():
        return _implementation.release_name(seed)


def load_release_plan() -> dict[str, object]:
    with _configured():
        return _implementation.load_release_plan()


def apply_release(plan: dict[str, object]) -> dict[str, object]:
    with _configured():
        return _implementation.apply_release(plan)


def main() -> int:
    with _configured():
        return _implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
