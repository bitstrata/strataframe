# src/strataframe/correlation/__init__.py
from __future__ import annotations

"""
Correlation subpackage = DTW + framework selection.

Do NOT import strataframe.rgt.* here.
This __init__ executes whenever any submodule is imported (e.g. correlation.dtw),
so keep it import-light and avoid pulling in scipy-heavy modules.
"""

from .dtw import DtwConfig, dtw_path_and_cost, correlate_graph_edges_dtw
from .framework import FrameworkConfig, edge_similarity, add_similarity_scores, prune_to_framework

__all__ = [
    "DtwConfig",
    "dtw_path_and_cost",
    "correlate_graph_edges_dtw",
    "FrameworkConfig",
    "edge_similarity",
    "add_similarity_scores",
    "prune_to_framework",
]
