"""Tests for chord geometry helpers (requires YADE for Vector3)."""
from __future__ import annotations

import pytest

pytest.importorskip("yade")

import ingress_func_v1 as ing


def test_chord_box_3d_geometry_and_corners():
    d, y, h, depth = 10.0, 0.0, 2.0, 5.0
    box = ing.chord_box_3d(d, y, h, depth)
    r = d / 2.0
    assert box["x_min"] == pytest.approx(-r)
    assert box["x_max"] == pytest.approx(r)
    assert box["y_bottom"] == pytest.approx(y)
    assert box["y_top"] == pytest.approx(y + h)
    assert box["z_min"] == 0.0
    assert box["z_max"] == depth
    assert box["width"] == pytest.approx(2 * r)
    assert box["height"] == h
    assert box["depth"] == depth
    mc, xc = box["min_corner"], box["max_corner"]
    assert float(mc[0]) == pytest.approx(box["x_min"])
    assert float(xc[1]) == pytest.approx(box["y_top"])


def test_chord_box_3d_clamps_y_to_radius():
    box = ing.chord_box_3d(4.0, 10.0, 1.0, 1.0)
    assert abs(box["y_bottom"]) <= 2.0 + 1e-9
