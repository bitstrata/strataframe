from __future__ import annotations

from typing import Set, Tuple

import numpy as np


def build_delaunay_edges(xy: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Build Delaunay triangulation edges from xy (shape [N,2]).
    Returns a set of (i,j) index pairs (i<j).
    """
    try:
        from scipy.spatial import Delaunay, QhullError  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scipy is required for Delaunay triangulation. Install with: pip install scipy") from e

    if xy.ndim != 2 or xy.shape[0] < 3:
        return set()

    try:
        tri = Delaunay(xy)
    except QhullError:
        return set()
    edges: Set[Tuple[int, int]] = set()
    for t in tri.simplices:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        edges.add((a, b) if a < b else (b, a))
        edges.add((b, c) if b < c else (c, b))
        edges.add((a, c) if a < c else (c, a))
    return edges
