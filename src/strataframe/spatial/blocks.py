# src/strataframe/spatial/blocks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from strataframe.io.wells import Well
from strataframe.curves.normalize_header import norm_mnemonic  # canonical header normalization


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


# -----------------------------------------------------------------------------
# Canonicalization (curve header / mnemonic keys)
# -----------------------------------------------------------------------------

def _finite_count(arr: np.ndarray) -> int:
    try:
        a = np.asarray(arr, dtype="float64")
        return int(np.isfinite(a).sum())
    except Exception:
        return 0


def canonicalize_well_logs_inplace(wells: List[Well]) -> None:
    """
    Ensure Well.logs keys are canonical mnemonics (e.g., 'GAMMA' -> 'GR').

    Collision policy (if two raw keys map to the same canonical key):
      keep the array with more finite samples; tie-break: keep existing.
    """
    for w in wells:
        logs = getattr(w, "logs", None)
        if not isinstance(logs, dict) or not logs:
            continue

        new_logs: Dict[str, np.ndarray] = {}
        for raw_key, arr in list(logs.items()):
            ck = norm_mnemonic(raw_key)
            if not ck:
                continue

            if ck not in new_logs:
                new_logs[ck] = arr
                continue

            # collision: choose "better" payload
            a0 = new_logs[ck]
            if _finite_count(np.asarray(arr)) > _finite_count(np.asarray(a0)):
                new_logs[ck] = arr

        # mutate in place: downstream sees canonical-only keys
        w.logs = new_logs  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Spatial tiling + halo
# -----------------------------------------------------------------------------

def make_tiles_with_halo(
    wells: List[Well],
    *,
    tile_km: float,
    halo_km: float,
) -> Tuple[Dict[str, Block], List[Adjacency]]:
    """
    Deterministic tiling + halo overlap.

    IMPORTANT:
      - Coordinates assumed in meters (project first if starting from lat/lon).
      - This function now canonicalizes Well.logs keys in-place so downstream
        code can request curves by canonical mnemonic only.
    """
    # Canonicalize curve keys (harmless if wells have empty/no logs)
    canonicalize_well_logs_inplace(wells)

    xy = np.array([(w.x, w.y) for w in wells], dtype=float)
    xmin, ymin = xy.min(axis=0)

    tile_m = float(tile_km) * 1000.0
    halo_m = float(halo_km) * 1000.0
    if tile_m <= 0.0:
        raise ValueError("tile_km must be > 0")
    if halo_m < 0.0:
        raise ValueError("halo_km must be >= 0")

    ix = np.floor((xy[:, 0] - xmin) / tile_m).astype(int)
    iy = np.floor((xy[:, 1] - ymin) / tile_m).astype(int)

    tile_to_wells: Dict[Tuple[int, int], List[str]] = {}
    for w, tx, ty in zip(wells, ix, iy):
        tile_to_wells.setdefault((int(tx), int(ty)), []).append(w.well_id)

    blocks: Dict[str, Block] = {}
    for (tx, ty), core_ids in tile_to_wells.items():
        bx0 = xmin + tx * tile_m
        by0 = ymin + ty * tile_m
        bx1 = bx0 + tile_m
        by1 = by0 + tile_m
        bbox = (float(bx0), float(by0), float(bx1), float(by1))

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
        _, tx_s, ty_s = bid.split("_")
        blocks_by_xy[(int(tx_s), int(ty_s))] = b

    adj: List[Adjacency] = []
    for (tx, ty), b in blocks_by_xy.items():
        for dx, dy in [(1, 0), (0, 1)]:
            nb = blocks_by_xy.get((tx + dx, ty + dy))
            if nb is None:
                continue
            shared = sorted(
                (set(b.core_well_ids) | set(b.halo_well_ids))
                & (set(nb.core_well_ids) | set(nb.halo_well_ids))
            )
            adj.append(Adjacency(a=b.block_id, b=nb.block_id, shared_well_ids=shared))

    return blocks, adj
