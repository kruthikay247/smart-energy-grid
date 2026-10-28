"""Run: python -m pytest tests -q   (or: python tests/test_controller.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from controller import decide, STORAGE, GRID


def test_edges():
    assert decide(160, 12)["mode"] == GRID          # exactly 160 -> not > 160
    assert decide(160.01, 12)["mode"] == STORAGE
    assert decide(100, 17)["mode"] == GRID
    assert decide(100, 18)["mode"] == STORAGE       # window start inclusive
    assert decide(100, 22)["mode"] == STORAGE       # window end inclusive
    assert decide(100, 23)["mode"] == GRID
    assert decide(199.99, 3)["overload"] is False
    assert decide(200, 3)["overload"] is True       # >= 200
    assert decide(210, 15)["mode"] == STORAGE and decide(210, 15)["overload"]


if __name__ == "__main__":
    test_edges(); print("controller tests passed")
