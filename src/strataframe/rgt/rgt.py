# src/strataframe/rgt/rgt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

try:
    from scipy.sparse import csr_matrix  # type: ignore
    from scipy.sparse.linalg import LinearOperator, cg  # type: ignore
except Exception:  # pragma: no cover
    csr_matrix = None  # type: ignore
    LinearOperator = None  # type: ignore
    cg = None  # type: ignore


@dataclass(frozen=True)
class Anchor:
    """
    Anchor constraint applied to one unknown s[node, sample_idx].

    Equation (soft constraint):
      lambda_anchor * w^2 * ( s[idx] - target_shift ) = 0

    Which contributes:
      A[idx,idx] += lambda_anchor * w^2
      b[idx]     += lambda_anchor * w^2 * target_shift
    """
    node_id: str
    sample_idx: int
    target_shift: float
    weight: float = 1.0


@dataclass(frozen=True)
class RgtConfig:
    # Damping removes gauge freedom (null space) in A = B^T B
    damping: float = 1e-2
    # Max cg iterations
    maxiter: int = 500
    # Convergence tolerance (maps to rtol when supported)
    tol: float = 1e-6
    # If True: map constraints onto same index i (ChronoLog-style simplification).
    # If False: use precise DTW indices (i,j).
    simplified_indexing: bool = True
    # Optional anchor strength (0 => off)
    lambda_anchor: float = 0.0


def _require() -> None:
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")
    if csr_matrix is None or LinearOperator is None or cg is None:
        raise RuntimeError("scipy is required. Install with: pip install scipy")


def _check_uniform_nS(G: "nx.Graph", *, z_key: str) -> int:
    nodes = list(G.nodes)
    if not nodes:
        raise ValueError("Graph has no nodes.")
    nS0 = int(np.asarray(G.nodes[nodes[0]][z_key]).size)
    if nS0 <= 0:
        raise ValueError(f"Node {nodes[0]} has empty {z_key}.")
    for n in nodes[1:]:
        nS = int(np.asarray(G.nodes[n][z_key]).size)
        if nS != nS0:
            raise ValueError(f"Non-uniform resampled length: node {n} has {nS} != {nS0} for {z_key}.")
    return nS0


def _median_per_i(i_vals: np.ndarray, j_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(i_vals, kind="mergesort")
    i_sorted = i_vals[order]
    j_sorted = j_vals[order]

    change = np.ones_like(i_sorted, dtype=bool)
    change[1:] = i_sorted[1:] != i_sorted[:-1]
    starts = np.where(change)[0]
    ends = np.r_[starts[1:], i_sorted.size]

    i_unique = i_sorted[starts]
    j_med = np.empty_like(i_unique, dtype=np.int64)

    for k, (s, e) in enumerate(zip(starts, ends)):
        js = j_sorted[s:e]
        j_med[k] = int(np.median(js))

    return i_unique.astype(np.int64, copy=False), j_med


def _build_constraints_arrays(
    G: "nx.Graph",
    node_index: Dict[str, int],
    *,
    nS: int,
    z_key: str,
    path_key: str,
    simplified: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_list: List[int] = []
    q_list: List[int] = []
    dz_list: List[float] = []

    for u, v, ed in G.edges(data=True):
        wp = ed.get(path_key, None)
        if wp is None:
            continue

        wp = np.asarray(wp, dtype=np.int64)
        if wp.size == 0 or wp.ndim != 2 or wp.shape[1] != 2:
            continue

        iu = int(node_index[u])
        iv = int(node_index[v])

        zu = np.asarray(G.nodes[u][z_key], dtype="float64")
        zv = np.asarray(G.nodes[v][z_key], dtype="float64")
        if zu.size != nS or zv.size != nS:
            continue

        i_vals = wp[:, 0].astype(np.int64, copy=False)
        j_vals = wp[:, 1].astype(np.int64, copy=False)

        if simplified:
            i_u, j_med = _median_per_i(i_vals, j_vals)
            m = (i_u >= 0) & (i_u < nS) & (j_med >= 0) & (j_med < nS)
            if not np.any(m):
                continue
            i_u = i_u[m]
            j_med = j_med[m]

            p = iu * nS + i_u
            q = iv * nS + i_u
            dz = zv[j_med] - zu[i_u]

            p_list.extend(p.tolist())
            q_list.extend(q.tolist())
            dz_list.extend(dz.astype("float64", copy=False).tolist())

        else:
            m = (i_vals >= 0) & (i_vals < nS) & (j_vals >= 0) & (j_vals < nS)
            if not np.any(m):
                continue
            i2 = i_vals[m]
            j2 = j_vals[m]

            p = iu * nS + i2
            q = iv * nS + j2
            dz = zv[j2] - zu[i2]

            p_list.extend(p.tolist())
            q_list.extend(q.tolist())
            dz_list.extend(dz.astype("float64", copy=False).tolist())

    if not p_list:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype="float64"),
        )

    return (
        np.asarray(p_list, dtype=np.int64),
        np.asarray(q_list, dtype=np.int64),
        np.asarray(dz_list, dtype="float64"),
    )


def _cg_solve(A: "LinearOperator", b: np.ndarray, *, x0: np.ndarray, maxiter: int, tol: float) -> Tuple[np.ndarray, int]:
    try:
        sol, info = cg(A, b, x0=x0, maxiter=int(maxiter), rtol=float(tol), atol=0.0)  # type: ignore[call-arg]
        return np.asarray(sol, dtype="float64"), int(info)
    except TypeError:
        sol, info = cg(A, b, x0=x0, maxiter=int(maxiter), tol=float(tol))  # type: ignore[call-arg]
        return np.asarray(sol, dtype="float64"), int(info)


def _build_anchor_terms(
    anchors: Optional[List[Anchor]],
    *,
    node_index: Dict[str, int],
    nS: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      a_idx  (K,) int64 indices into the flattened unknown vector
      a_diag (K,) float64 diagonal weights (w^2) accumulated per index
      a_rhs  (K,) float64 diagonal*w^2 * target_shift accumulated per index
    """
    if not anchors:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype="float64"),
            np.zeros((0,), dtype="float64"),
        )

    diag: Dict[int, float] = {}
    rhs: Dict[int, float] = {}

    for a in anchors:
        nid = str(a.node_id)
        if nid not in node_index:
            raise ValueError(f"Anchor node_id not in graph: {nid}")
        i = int(a.sample_idx)
        if i < 0 or i >= int(nS):
            raise ValueError(f"Anchor sample_idx out of range: {nid} idx={i} (nS={nS})")

        w2 = float(a.weight) * float(a.weight)
        if not np.isfinite(w2) or w2 <= 0.0:
            continue

        idx = int(node_index[nid]) * int(nS) + i
        diag[idx] = float(diag.get(idx, 0.0) + w2)
        rhs[idx] = float(rhs.get(idx, 0.0) + w2 * float(a.target_shift))

    if not diag:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype="float64"),
            np.zeros((0,), dtype="float64"),
        )

    a_idx = np.asarray(sorted(diag.keys()), dtype=np.int64)
    a_diag = np.asarray([diag[int(k)] for k in a_idx], dtype="float64")
    a_rhs = np.asarray([rhs[int(k)] for k in a_idx], dtype="float64")
    return a_idx, a_diag, a_rhs


def solve_rgt_shifts(
    G: "nx.Graph",
    *,
    cfg: RgtConfig,
    z_key: str = "depth_rs",
    path_key: str = "dtw_path",
    anchors: Optional[List[Anchor]] = None,
) -> Dict[str, np.ndarray]:
    """
    Solves for per-node shift vector s on the resampled grid:
      RGT(z) = z + s

    Normal equations:
      (B^T B + damping*I + lambda_anchor*A_anchor) s = B^T dz + lambda_anchor*b_anchor
    """
    _require()

    nodes = list(G.nodes)
    if not nodes:
        raise ValueError("Graph has no nodes.")

    nS = _check_uniform_nS(G, z_key=z_key)
    nW = int(len(nodes))
    n_unknowns = int(nW * nS)

    node_index = {n: i for i, n in enumerate(nodes)}

    p_idx, q_idx, dz = _build_constraints_arrays(
        G,
        node_index,
        nS=nS,
        z_key=z_key,
        path_key=path_key,
        simplified=cfg.simplified_indexing,
    )

    if dz.size == 0:
        raise RuntimeError("No DTW constraints found on graph edges. Ensure edges have dtw_path.")

    M = int(dz.size)

    rows = np.repeat(np.arange(M, dtype=np.int64), 2)
    cols = np.concatenate([p_idx, q_idx]).astype(np.int64, copy=False)
    data = np.concatenate([np.ones(M, dtype="float64"), -np.ones(M, dtype="float64")])

    B = csr_matrix((data, (rows, cols)), shape=(M, n_unknowns), dtype="float64")

    b = (B.T @ dz).astype("float64", copy=False)

    damping = float(cfg.damping)
    lam_a = float(getattr(cfg, "lambda_anchor", 0.0) or 0.0)

    a_idx, a_diag, a_rhs = _build_anchor_terms(anchors, node_index=node_index, nS=nS)
    if lam_a > 0.0 and a_idx.size > 0:
        b = b.copy()
        b[a_idx] += lam_a * a_rhs

    def matvec(x: np.ndarray) -> np.ndarray:
        y = B.T @ (B @ x)
        if damping > 0.0:
            y = y + damping * x
        if lam_a > 0.0 and a_idx.size > 0:
            # diagonal add at anchored indices
            y = np.asarray(y, dtype="float64", copy=False)
            y[a_idx] += lam_a * a_diag * x[a_idx]
        return np.asarray(y, dtype="float64")

    A = LinearOperator((n_unknowns, n_unknowns), matvec=matvec, dtype=np.float64)

    x0 = np.zeros(n_unknowns, dtype="float64")
    sol, info = _cg_solve(A, np.asarray(b, dtype="float64"), x0=x0, maxiter=int(cfg.maxiter), tol=float(cfg.tol))

    if info != 0:
        raise RuntimeError(
            f"Conjugate gradient did not converge (info={info}). "
            f"Try increasing maxiter, increasing damping, or reducing constraints."
        )

    out: Dict[str, np.ndarray] = {}
    for n in nodes:
        i = node_index[n]
        out[n] = sol[i * nS : (i + 1) * nS].copy()

    return out
