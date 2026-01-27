# src/strataframe/qc/gr_features.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from strataframe.graph.las_utils import read_las_normal, extract_depth_and_curve, resample_and_normalize_curve


@dataclass(frozen=True)
class GrFeatureConfig:
    """
    Feature extraction for type-well selection and QC.

    n_lowres:
      - low-res GR vector length used for shape comparison within a kernel
      - independent of Step 3 n_samples

    min_finite_raw:
      - minimum finite samples in raw GR required to consider the well usable
    """
    n_lowres: int = 128
    p_lo: float = 1.0
    p_hi: float = 99.0
    min_finite_raw: int = 50

    # Invariance heuristic in *raw* units (kept permissive; final outliering is kernel-relative)
    min_iqr_raw: float = 0.5
    min_range_raw: float = 2.0

    # Sand fraction threshold on normalized [0,1] curve (simple proxy)
    sand_thr_norm: float = 0.35


@dataclass
class GrFeatures:
    status: str
    error: str

    z_top: float
    z_base: float
    thickness: float

    n: int
    finite_frac: float

    p05: float
    p50: float
    p95: float
    iqr: float
    rng: float
    std: float

    sand_frac: float

    # Shape vector for kernel comparisons (normalized 0..1)
    gr_lowres: Optional[np.ndarray]


def _nanpct(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, dtype="float64")
    fin = np.isfinite(x)
    if not np.any(fin):
        return float("nan")
    return float(np.percentile(x[fin], float(p)))


def compute_gr_features(las_path: Path, *, cfg: GrFeatureConfig) -> GrFeatures:
    """
    Loads full GR curve and derives QC + typewell selection features.
    """
    try:
        las = read_las_normal(las_path)
        depth, gr = extract_depth_and_curve(las, curve_mnemonic="GR", depth_preferred=("DEPT", "MD"))
    except Exception as e:
        return GrFeatures(
            status="FAIL",
            error=f"{type(e).__name__}: {e}",
            z_top=float("nan"),
            z_base=float("nan"),
            thickness=float("nan"),
            n=0,
            finite_frac=0.0,
            p05=float("nan"),
            p50=float("nan"),
            p95=float("nan"),
            iqr=float("nan"),
            rng=float("nan"),
            std=float("nan"),
            sand_frac=float("nan"),
            gr_lowres=None,
        )

    depth = np.asarray(depth, dtype="float64").reshape(-1)
    gr = np.asarray(gr, dtype="float64").reshape(-1)

    if depth.size < 2 or gr.size < 2:
        return GrFeatures(
            status="FAIL",
            error="too_short",
            z_top=float("nan"),
            z_base=float("nan"),
            thickness=float("nan"),
            n=int(min(depth.size, gr.size)),
            finite_frac=0.0,
            p05=float("nan"),
            p50=float("nan"),
            p95=float("nan"),
            iqr=float("nan"),
            rng=float("nan"),
            std=float("nan"),
            sand_frac=float("nan"),
            gr_lowres=None,
        )

    # sort by depth
    o = np.argsort(depth)
    depth = depth[o]
    gr = gr[o]

    fin = np.isfinite(gr) & np.isfinite(depth)
    n = int(depth.size)
    nfin = int(np.count_nonzero(fin))
    if nfin < int(cfg.min_finite_raw):
        return GrFeatures(
            status="FAIL",
            error=f"too_few_finite:{nfin}",
            z_top=float(np.nanmin(depth)) if np.isfinite(depth).any() else float("nan"),
            z_base=float(np.nanmax(depth)) if np.isfinite(depth).any() else float("nan"),
            thickness=float("nan"),
            n=n,
            finite_frac=float(nfin / max(1, n)),
            p05=float("nan"),
            p50=float("nan"),
            p95=float("nan"),
            iqr=float("nan"),
            rng=float("nan"),
            std=float("nan"),
            sand_frac=float("nan"),
            gr_lowres=None,
        )

    z_top = float(np.nanmin(depth[fin]))
    z_base = float(np.nanmax(depth[fin]))
    th = float(z_base - z_top)

    p05 = _nanpct(gr, 5.0)
    p50 = _nanpct(gr, 50.0)
    p95 = _nanpct(gr, 95.0)
    p25 = _nanpct(gr, 25.0)
    p75 = _nanpct(gr, 75.0)

    iqr = float(p75 - p25) if np.isfinite(p75) and np.isfinite(p25) else float("nan")
    rng = float(p95 - p05) if np.isfinite(p95) and np.isfinite(p05) else float("nan")
    std = float(np.nanstd(gr[fin]))

    # low-res normalized vector across its *own* interval (shape-only proxy)
    try:
        gr_rs, _zt, _zb = resample_and_normalize_curve(depth, gr, n_samples=int(cfg.n_lowres), p_lo=float(cfg.p_lo), p_hi=float(cfg.p_hi))
        gr_lowres = np.asarray(gr_rs, dtype="float64")
        sand_frac = float(np.mean(gr_lowres < float(cfg.sand_thr_norm)))
    except Exception:
        gr_lowres = None
        sand_frac = float("nan")

    # permissive invariance check (final decision is kernel-relative)
    if np.isfinite(iqr) and np.isfinite(rng):
        if iqr < float(cfg.min_iqr_raw) and rng < float(cfg.min_range_raw):
            status = "WARN"
            err = "low_variance"
        else:
            status = "OK"
            err = ""
    else:
        status = "WARN"
        err = "bad_stats"

    return GrFeatures(
        status=status,
        error=err,
        z_top=z_top,
        z_base=z_base,
        thickness=th,
        n=n,
        finite_frac=float(nfin / max(1, n)),
        p05=p05,
        p50=p50,
        p95=p95,
        iqr=iqr,
        rng=rng,
        std=std,
        sand_frac=sand_frac,
        gr_lowres=gr_lowres,
    )
