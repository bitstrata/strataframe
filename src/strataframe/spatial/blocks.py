from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from strataframe.io.wells import Well


@dataclass(frozen=True)
class Block:
    block_id: str
    core_well_ids: List[str]
    halo_well_ids: List[str]
    bbox: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Adjacency:
    a: str
    b: str
    shared_well_ids: List[str]


def make_tiles_with_halo(
    wells: List[Well],
    *,
    tile_km: float,
    halo_km: float,
) -> Tuple[Dict[str, Block], List[Adjacency]]:
    """
    Deterministic tiling + halo overlap.
    Coordinates assumed in meters. If lat/lon, project first.
    """
    xy = np.array([(w.x, w.y) for w in wells], dtype=float)
    xmin, ymin = xy.min(axis=0)

    tile_m = tile_km * 1000.0
    halo_m = halo_km * 1000.0

    ix = np.floor((xy[:, 0] - xmin) / tile_m).astype(int)
    iy = np.floor((xy[:, 1] - ymin) / tile_m).astype(int)

    tile_to_wells: Dict[Tuple[int, int], List[str]] = {}
    for w, tx, ty in zip(wells, ix, iy):
        tile_to_wells.setdefault((tx, ty), []).append(w.well_id)

    blocks: Dict[str, Block] = {}
    for (tx, ty), core_ids in tile_to_wells.items():
        bx0 = xmin + tx * tile_m
        by0 = ymin + ty * tile_m
        bx1 = bx0 + tile_m
        by1 = by0 + tile_m
        bbox = (bx0, by0, bx1, by1)

        ex0, ey0, ex1, ey1 = (bx0 - halo_m, by0 - halo_m, bx1 + halo_m, by1 + halo_m)
        halo_ids: List[str] = []
        for w in wells:
            if ex0 <= w.x <= ex1 and ey0 <= w.y <= ey1:
                halo_ids.append(w.well_id)

        block_id = f"tile_{tx}_{ty}"
        core_set = set(core_ids)
        halo_set = set(halo_ids) - core_set
        blocks[block_id] = Block(
            block_id=block_id,
            core_well_ids=sorted(core_set),
            halo_well_ids=sorted(halo_set),
            bbox=bbox,
        )

    # adjacency: 4-neighborhood
    blocks_by_xy: Dict[Tuple[int, int], Block] = {}
    for bid, b in blocks.items():
        _, tx, ty = bid.split("_")
        blocks_by_xy[(int(tx), int(ty))] = b

    adj: List[Adjacency] = []
    for (tx, ty), b in blocks_by_xy.items():
        for dx, dy in [(1, 0), (0, 1)]:
            nb = blocks_by_xy.get((tx + dx, ty + dy))
            if nb is None:
                continue
            shared = sorted(
                (set(b.core_well_ids) | set(b.halo_well_ids)) &
                (set(nb.core_well_ids) | set(nb.halo_well_ids))
            )
            adj.append(Adjacency(a=b.block_id, b=nb.block_id, shared_well_ids=shared))

    return blocks, adj
