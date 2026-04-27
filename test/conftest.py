"""Pytest configuration: repo roots on sys.path for flat scripts."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (
    REPO / "0_simulation",
    REPO / "1_data_processing",
    REPO / "2_model" / "1_RNNSR",
    REPO / "3_metrics" / "0_lacey_mixing_index",
    REPO / "4_visualization",
):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
