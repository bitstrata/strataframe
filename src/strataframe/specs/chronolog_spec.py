# src/strataframe/correlation/chronolog_spec.py
from __future__ import annotations

"""
ChronoLog methods tracking (scaffold notes)
-------------------------------------------

This module is *documentation-only* (no runtime code). It exists to keep the
pipeline aligned to the ChronoLog-style method summary while reflecting the
current implementation choices in this repo.

Key repo-specific policy: CANONICAL CURVE HEADERS
-------------------------------------------------
We treat LAS header mnemonics/units as messy inputs and enforce a canonical
namespace for downstream logic.

- Canonical mnemonic keys (examples): "DEPT", "GR", "RHOB", "DT", ...
- Alias families exist for ingest/resolution only (e.g., GAMMA, GAMMA_RAY, GR:1).
- Downstream steps should request canonical mnemonics only. If an ingest step
  needs to fetch a curve from a LAS file, it must:
    1) canonicalize request (e.g., "GR")
    2) resolve an available raw header via alias family
    3) index las[...] by the resolved raw mnemonic
    4) record both:
         - requested canonical mnemonic (e.g., "GR")
         - resolved raw mnemonic (e.g., "GAMMA_RAY" or "GR:1")
  This avoids brittle string compares and stabilizes cross-corpus behavior.

(See: strataframe/curves/normalize_header.py for norm_mnemonic(), aliases_for(), etc.)

---------------------------------------------------------------------
From the provided Methods excerpt (adapted to current implementation):
---------------------------------------------------------------------

2.1 Well database / well graph
- Well object holds: id/name, logs, depth, geographic coordinates
- Wells are nodes in a graph (currently materialized as CSV edges/nodes; networkx is optional)
- Edges represent well pairs to be correlated by DTW; DTW results stored as edge attributes

- On curve extraction: values are resampled to a uniform depth grid and normalized to [0,1]
  via percentile scaling:
      x_n = clip( (x - p_lo) / (p_hi - p_lo), 0, 1 )
  where p_lo ~ 1st percentile and p_hi ~ 99th percentile.

- Log-type selection:
  Prefer logs with broad availability/coverage (often GR).
  IMPORTANT: selection must be canonical-first:
    - request "GR"
    - resolve raw header via aliases_for("GR")
    - persist picked_gr as the resolved raw mnemonic (optional), but treat "GR" as the key.

- Interval of interest:
  Current implementation uses full curve depth extents (z_top, z_base) per well curve.
  (Future: constrain to a project interval; interpolate top/base if missing.)

- Missing upper/lower segments:
  Methods describe imputation from nearby wells (horizontal assumption) and flagging imputed
  segments. Not implemented yet in this repo (planned extension).

2.2 Selection of well pairs
- All pairs is too expensive; use spatial pruning.
- Current implementation (graph/build_sparse_dtw_edges.py):
    - include all pairs within max distance threshold inside local clusters
    - add sparse "bridge" edges via k-nearest neighbors (kNN) capped by a larger max distance
  (Methods mention Delaunay triangulation for bridges; this repo uses kNN bridges for simplicity.
   Delaunay can be added later if needed.)

2.3 DTW correlation
- Uses librosa.sequence.dtw (numba-accelerated) with robustified pointwise cost:
      d = |l2 - l1|^alpha
  with alpha default 0.15 (de-emphasizes outliers / "layercake bias")

- Endpoint-constrained:
  Standard DTW in librosa aligns endpoints by construction (path from (0,0) to (N-1,M-1)).
  If additional hard constraints are needed later, add them explicitly.

- Store DTW cost + path:
  Current implementation stores dtw_cost and dtw_cost_per_step per edge. Optionally stores
  a downsampled set of tiepoints from the DTW path for compact persistence.

2.4 Conflicting correlation optimization (RGT / chronostrat diagram)
- For each correlated depth pair z1~z2:
      s1 - s2 = z2 - z1
  Build an overdetermined system and solve least squares (sparse, iterative), then interpolate
  shifts back to original depth sampling.

- Not implemented yet in this repo; this file tracks the intended math + future work.

Practical invariant for implementation reviews:
-----------------------------------------------
Any step that:
  - compares mnemonics OR
  - chooses a "preferred curve" OR
  - indexes las[...] by a curve name
must be canonical-first and must not assume raw header strings are stable.
"""
