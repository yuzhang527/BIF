from __future__ import annotations

import numpy as np

from bif.analysis.sweep_diagnostics import topk_overlap_metrics
from bif.analysis.sweep_runner import _values_from_spec


def test_topk_overlap_recall_and_jaccard() -> None:
    a = np.array([10.0, 9.0, 8.0, 1.0])
    b = np.array([10.0, 8.0, 9.0, 1.0])
    metrics = topk_overlap_metrics(a, b, [2, 3])
    assert metrics["top2_overlap_recall"] == 0.5
    assert metrics["top2_jaccard"] == 1.0 / 3.0
    assert metrics["top3_overlap_recall"] == 1.0
    assert metrics["top3_jaccard"] == 1.0


def test_values_from_spec_supports_values_and_logspace() -> None:
    assert _values_from_spec({"values": [1, 2]}, "lr") == [1.0, 2.0]
    vals = _values_from_spec({"min": 1e-6, "max": 1e-4, "num": 3, "scale": "log"}, "lr")
    assert len(vals) == 3
    assert np.isclose(vals[0], 1e-6)
    assert np.isclose(vals[-1], 1e-4)
