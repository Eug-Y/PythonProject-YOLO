"""Evaluate the locked V19 candidate against production on test exactly once."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import run_data_v20_final_test as _implementation


ROOT = Path(__file__).resolve().parent
_OVERRIDES = {
    "V20_SUMMARY": ROOT / "logs" / "v19b_yolo11_boundary_nooverlap_multiseed.summary.json",
    "OUTPUT": ROOT / "runs" / "final_evaluation" / "data_v19_locked_test",
    "GUARD": ROOT / "logs" / "data_v19_final_test_guard.json",
    "STATUS": ROOT / "logs" / "data_v19_final_test.status.json",
    "LOG": ROOT / "logs" / "data_v19_final_test.log",
    "EXPERIMENT_LABEL": "V19",
    "ARTIFACT_PREFIX": "data_v19",
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


fixed_totals = _implementation.fixed_totals
upgrade_decision = _implementation.upgrade_decision
extract_frozen_candidate = _implementation.extract_frozen_candidate


def main() -> int:
    with _configured():
        return _implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
