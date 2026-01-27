# src/strataframe/correlation/paths_npz.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class PackedPaths:
    """
    Packed warping paths for fast disk IO.

    Format:
      src_ids: (E,) str
      dst_ids: (E,) str
      ptr:     (E+1,) int64 prefix sums into ii/jj
      ii:      (K,) int32
      jj:      (K,) int32

    For edge e in [0..E-1], its path is:
      k0 = ptr[e], k1 = ptr[e+1]
      path = stack([ii[k0:k1], jj[k0:k1]], axis=1)
    """
    src_ids: np.ndarray
    dst_ids: np.ndarray
    ptr: np.ndarray
    ii: np.ndarray
    jj: np.ndarray

    def to_dict(self) -> Dict[Tuple[str, str], np.ndarray]:
        out: Dict[Tuple[str, str], np.ndarray] = {}
        E = int(self.src_ids.size)
        for e in range(E):
            k0 = int(self.ptr[e])
            k1 = int(self.ptr[e + 1])
            path = np.stack([self.ii[k0:k1], self.jj[k0:k1]], axis=1).astype(np.int64)
            out[(str(self.src_ids[e]), str(self.dst_ids[e]))] = path
        return out


def _as_str_array(x: Iterable[str]) -> np.ndarray:
    xs = [str(v) for v in x]
    # Use unicode dtype; npz handles this well
    return np.asarray(xs, dtype=np.str_)


def pack_paths(
    edge_ids: Iterable[Tuple[str, str]],
    paths: Iterable[np.ndarray],
    *,
    canonicalize: bool = True,
    dtype_index: np.dtype = np.int32,
) -> PackedPaths:
    """
    Pack (edge_id -> path) into a compact structure.

    If canonicalize=True: edges are stored with (min(src,dst), max(src,dst)).
    Caller must ensure the stored path corresponds to that direction.
    """
    e_list: List[Tuple[str, str]] = []
    p_list: List[np.ndarray] = []

    for (u0, v0), p in zip(edge_ids, paths):
        u = str(u0)
        v = str(v0)
        path = np.asarray(p, dtype=np.int64)
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError("Each DTW path must be shape (K,2).")

        if canonicalize and v < u:
            # If we flip the edge ordering, we must also swap path columns (i,j) -> (j,i)
            u, v = v, u
            path = path[:, [1, 0]]

        e_list.append((u, v))
        p_list.append(path)

    src_ids = _as_str_array([u for u, _ in e_list])
    dst_ids = _as_str_array([v for _, v in e_list])

    lengths = np.asarray([int(p.shape[0]) for p in p_list], dtype=np.int64)
    ptr = np.zeros((lengths.size + 1,), dtype=np.int64)
    ptr[1:] = np.cumsum(lengths)

    K = int(ptr[-1])
    ii = np.zeros((K,), dtype=dtype_index)
    jj = np.zeros((K,), dtype=dtype_index)

    k = 0
    for p in p_list:
        n = int(p.shape[0])
        ii[k : k + n] = p[:, 0].astype(dtype_index, copy=False)
        jj[k : k + n] = p[:, 1].astype(dtype_index, copy=False)
        k += n

    return PackedPaths(src_ids=src_ids, dst_ids=dst_ids, ptr=ptr, ii=ii, jj=jj)


def save_paths_npz(packed: PackedPaths, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        src_ids=packed.src_ids,
        dst_ids=packed.dst_ids,
        ptr=packed.ptr,
        ii=packed.ii,
        jj=packed.jj,
    )


def load_paths_npz(path: str | Path) -> PackedPaths:
    z = np.load(Path(path), allow_pickle=False)
    return PackedPaths(
        src_ids=np.asarray(z["src_ids"], dtype=np.str_),
        dst_ids=np.asarray(z["dst_ids"], dtype=np.str_),
        ptr=np.asarray(z["ptr"], dtype=np.int64),
        ii=np.asarray(z["ii"], dtype=np.int32),
        jj=np.asarray(z["jj"], dtype=np.int32),
    )


def get_path(packed: PackedPaths, src: str, dst: str) -> Optional[np.ndarray]:
    """
    Fetch a single path by (src,dst). Tries both orientations and returns path
    in the requested orientation (i,j for src->dst).
    """
    src = str(src)
    dst = str(dst)

    # Fast path: scan (E is usually manageable). If you need O(1), build an index map once.
    E = int(packed.src_ids.size)
    for e in range(E):
        u = str(packed.src_ids[e])
        v = str(packed.dst_ids[e])
        if u == src and v == dst:
            k0, k1 = int(packed.ptr[e]), int(packed.ptr[e + 1])
            return np.stack([packed.ii[k0:k1], packed.jj[k0:k1]], axis=1).astype(np.int64)

        if u == dst and v == src:
            # stored opposite direction; swap columns to match requested src->dst
            k0, k1 = int(packed.ptr[e]), int(packed.ptr[e + 1])
            return np.stack([packed.jj[k0:k1], packed.ii[k0:k1]], axis=1).astype(np.int64)

    return None
