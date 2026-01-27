# src/strataframe/viz/step3a_candidates_map.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .step3_common import line_collection_from_edges

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("pandas is required. pip install pandas") from e


def plot_step3a_candidates_map(
    *,
    reps: "pd.DataFrame",
    candidate_edges: Optional["pd.DataFrame"],
    out_png: Path,
    max_edges: int = 80_000,
    title: str = "Step 3a — Candidate graph (nodes + candidate edges)",
) -> None:
    """
    Plot candidate graph:
      - reps plotted as points (lon/lat)
      - candidate edges plotted as LineCollection (subset up to max_edges)

    Drop-in behavior:
      - Writes PNG to out_png (creating parent dirs)
      - Never raises on missing/empty candidate_edges
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Normalize/validate minimal required columns
    if "lon" not in reps.columns or "lat" not in reps.columns:
        # allow common aliases
        lon_c = None
        lat_c = None
        for c in reps.columns:
            lc = str(c).strip().lower()
            if lon_c is None and lc in ("lon", "longitude", "long"):
                lon_c = c
            if lat_c is None and lc in ("lat", "latitude"):
                lat_c = c
        if lon_c is not None and lon_c != "lon":
            reps = reps.rename(columns={lon_c: "lon"})
        if lat_c is not None and lat_c != "lat":
            reps = reps.rename(columns={lat_c: "lat"})
    if "lon" not in reps.columns or "lat" not in reps.columns:
        raise ValueError("reps must contain lon/lat columns (or longitude/latitude aliases).")

    # Coerce to numeric; filter finite for extents/scatter (LineCollection already filters internally)
    lon = pd.to_numeric(reps["lon"], errors="coerce").to_numpy(dtype="float64")
    lat = pd.to_numeric(reps["lat"], errors="coerce").to_numpy(dtype="float64")
    ok = np.isfinite(lon) & np.isfinite(lat)

    fig, ax = plt.subplots(figsize=(11, 9))

    if candidate_edges is not None and len(candidate_edges) > 0:
        lc, n_used = line_collection_from_edges(reps, candidate_edges, max_edges=int(max_edges))
        # keep a neutral gray; use set_color string to avoid style dependencies
        lc.set_color("0.65")
        ax.add_collection(lc)
        ax.text(
            0.01,
            0.01,
            f"candidate edges plotted: {n_used:,} (max={int(max_edges):,})",
            transform=ax.transAxes,
            fontsize=9,
        )
    else:
        ax.text(0.01, 0.01, "candidate edges: (not found)", transform=ax.transAxes, fontsize=9)

    ax.scatter(lon[ok], lat[ok], s=10, alpha=0.85)

    ax.set_title(str(title))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Use equal scaling but keep a sane view if data is tiny/degenerate
    if ok.any():
        ax.set_xlim(float(np.nanmin(lon[ok])), float(np.nanmax(lon[ok])))
        ax.set_ylim(float(np.nanmin(lat[ok])), float(np.nanmax(lat[ok])))
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(out_png, dpi=175)
    plt.close(fig)
