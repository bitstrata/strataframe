# src/strataframe/viz/step3c_dtw_map.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from .step3_common import line_collection_from_edges

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("pandas is required. pip install pandas") from e


def plot_step3c_dtw_map(
    *,
    reps: "pd.DataFrame",
    candidate_edges: Optional["pd.DataFrame"],
    dtw_edges: "pd.DataFrame",
    out_png: Path,
    max_edges: int = 80_000,
    title: str = "Step 3c — DTW edges (overlay on candidates)",
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))

    # candidates: light grey
    if candidate_edges is not None and len(candidate_edges) > 0:
        lc_c, n_used = line_collection_from_edges(reps, candidate_edges, max_edges=int(max_edges))
        lc_c.set_color("0.80")
        lc_c.set_alpha(0.25)
        ax.add_collection(lc_c)
        ax.text(0.01, 0.01, f"candidate edges plotted: {n_used:,}", transform=ax.transAxes, fontsize=9)

    # dtw edges: darker
    lc_d, n_used_d = line_collection_from_edges(reps, dtw_edges, max_edges=int(max_edges))
    lc_d.set_color("0.20")
    lc_d.set_alpha(0.65)
    try:
        lc_d.set_linewidth(0.9)
    except Exception:
        # Some matplotlib versions/LineCollection wrappers may not expose set_linewidth
        pass
    ax.add_collection(lc_d)
    ax.text(0.01, 0.04, f"dtw edges plotted: {n_used_d:,}", transform=ax.transAxes, fontsize=9)

    # nodes
    ax.scatter(reps["lon"], reps["lat"], s=10, alpha=0.85)

    ax.set_title(str(title))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(out_png, dpi=175)
    plt.close(fig)
