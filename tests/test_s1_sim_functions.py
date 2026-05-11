"""Light tests for simulation helpers (YADE optional)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("yade")

import s1_sim_functions as sim


def test_mat_label_empty_material():
    b = SimpleNamespace(material=None)
    assert sim._mat_label(b) == ""


def test_mat_label_with_label():
    b = SimpleNamespace(material=SimpleNamespace(label="rock"))
    assert sim._mat_label(b) == "rock"
