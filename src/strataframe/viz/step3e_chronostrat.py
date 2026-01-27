# src/strataframe/viz/step3e_chronostrat.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .step3_common import load_npz
from .step3_colors import yellow_brown_perceptual_cmap


def _coerce_diag_rgt(
    diag: np.ndarray,
    rgt: np.ndarray,
    node_order: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Ensure diag is shaped (W, T) where:
      - W = number of wells (rows)
      - T = number of RGT samples (cols)
      - rgt is length T

    Accepts either (W, T) or (T, W) and transposes if required.
    """
    diag = np.asarray(diag, dtype="float64")
    rgt = np.asarray(rgt, dtype="float64").reshape(-1)

    if diag.ndim != 2:
        raise ValueError(f"chronostrat 'diag' must be 2D; got shape={diag.shape}.")

    W, T = int(diag.shape[0]), int(diag.shape[1])
    n_rgt = int(rgt.size)

    if n_rgt == T:
        # already (W, T)
        return diag, rgt, node_order

    if n_rgt == W:
        # likely transposed; make (W, T)
        diag2 = diag.T
        W2, T2 = int(diag2.shape[0]), int(diag2.shape[1])
        if node_order is not None:
            node_order2 = np.asarray(node_order, dtype=np.str_).reshape(-1)
            if node_order2.size == T:  # node_order was along the other axis
                node_order2 = None
            elif node_order2.size != W2:
                node_order2 = None
        else:
            node_order2 = None
        return diag2, rgt, node_order2

    raise ValueError(
        f"RGT length ({n_rgt}) does not match diag shape ({W},{T}) in either axis. "
        "Expected rgt.size == diag.shape[1] (preferred) or == diag.shape[0] (transposed)."
    )


def plot_step3e_chronostrat(
    *,
    chronostrat_npz: Path,
    out_png: Path,
    title: str = "Step 3e — Chronostrat diagram (log in RGT space)",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    z = load_npz(chronostrat_npz)

    if "diag" not in z or "rgt_grid" not in z:
        raise ValueError(f"{chronostrat_npz} missing required keys 'diag' and/or 'rgt_grid'.")

    diag = np.asarray(z["diag"], dtype="float64")
    rgt = np.asarray(z["rgt_grid"], dtype="float64")
    node_order = np.asarray(z["node_order"], dtype=np.str_) if "node_order" in z else None

    diag, rgt, node_order = _coerce_diag_rgt(diag, rgt, node_order)

    # Colormap consistent with Step 3b/3d
    cmap = yellow_brown_perceptual_cmap()
    try:
        cmap = cmap.copy()
        cmap.set_bad(color="#ffffff")
    except Exception:
        pass

    W, T = diag.shape
    fig_h = 10 if W <= 200 else 14
    fig, ax = plt.subplots(figsize=(14, fig_h))

    # extent: x=rgt, y=well index; depth-like axis increasing downward for readability
    im = ax.imshow(
        diag,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        extent=[float(rgt[0]), float(rgt[-1]), float(W), 0.0],
    )

    ax.set_title(title)
    ax.set_xlabel("RGT")
    ax.set_ylabel("Wells (order)")

    # Optional y labels
    if node_order is not None:
        node_order = np.asarray(node_order, dtype=np.str_).reshape(-1)
        if node_order.size == W:
            step = max(1, W // 20)
            yt = np.arange(0, W, step)
            ax.set_yticks(yt)
            ax.set_yticklabels([node_order[i] for i in yt], fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Normalized log value")

    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=175)
    plt.close(fig)
