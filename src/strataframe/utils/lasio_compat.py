# src/strataframe/utils/lasio_compat.py
from __future__ import annotations

# LAS IO (support either module name)
from strataframe.io.las import read_las_header_only, read_las_safely

__all__ = ["read_las_header_only", "read_las_safely"]
