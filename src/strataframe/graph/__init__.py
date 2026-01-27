# src/strataframe/graph/__init__.py
from __future__ import annotations

"""
strataframe.graph

Grid-only graph/binning/selection utilities.

This package intentionally avoids eager imports to keep optional dependencies
and in-progress modules from breaking basic workflows (Step0/Step1/Step2).
"""

from importlib import import_module
from typing import Any

__all__ = [
    "GridBinningConfig",
    "bin_wells_grid",
    "RepSelectConfig",
    "select_representatives",
]


def __getattr__(name: str) -> Any:
    # Grid binning
    if name in ("GridBinningConfig", "bin_wells_grid"):
        m = import_module("strataframe.graph.bin_wells_grid")
        return getattr(m, name)

    # Rep selection
    if name in ("RepSelectConfig", "select_representatives"):
        m = import_module("strataframe.graph.select_representatives")
        return getattr(m, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
