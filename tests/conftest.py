"""Pytest configuration: script directories on sys.path for flat imports."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "bppm_dem_sm"
for p in (
    PKG / "data_gen_sim",
    PKG / "data_processing",
    PKG / "model" / "RNNSR",
    PKG / "metrics" / "0_lacey_mixing_index",
    PKG / "visualization",
):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
