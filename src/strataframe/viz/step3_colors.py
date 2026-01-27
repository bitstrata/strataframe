# src/strataframe/viz/step3_colors.py
from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap

# -----------------------------------------------------------------------------
# Perceptual 2-color ramp (CIE Lab interpolation, sRGB D65)
# -----------------------------------------------------------------------------

def _hex_to_rgb01(h: str) -> np.ndarray:
    h = (h or "").strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"bad hex: {h!r}")
    return np.array(
        [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)],
        dtype="float64",
    ) / 255.0


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype="float64")
    a = 0.055
    # sRGB electro-optical transfer function (inverse)
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1.0 + a)) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype="float64")
    a = 0.055
    # sRGB electro-optical transfer function
    return np.where(c <= 0.0031308, 12.92 * c, (1.0 + a) * np.power(c, 1.0 / 2.4) - a)


def _linear_rgb_to_xyz(rgb_lin: np.ndarray) -> np.ndarray:
    # sRGB (D65) to XYZ (D65)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype="float64",
    )
    rgb_lin = np.asarray(rgb_lin, dtype="float64")
    return rgb_lin @ M.T


def _xyz_to_linear_rgb(xyz: np.ndarray) -> np.ndarray:
    # XYZ (D65) to sRGB (D65)
    M = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype="float64",
    )
    xyz = np.asarray(xyz, dtype="float64")
    return xyz @ M.T


def _f_lab(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype="float64")
    d = 6.0 / 29.0
    return np.where(t > d**3, np.cbrt(t), (t / (3.0 * d * d) + 4.0 / 29.0))


def _f_lab_inv(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype="float64")
    d = 6.0 / 29.0
    return np.where(t > d, t**3, 3.0 * d * d * (t - 4.0 / 29.0))


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    # Reference white D65
    xyz = np.asarray(xyz, dtype="float64")
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn
    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def _lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    # Reference white D65
    lab = np.asarray(lab, dtype="float64")
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    x = _f_lab_inv(fx) * Xn
    y = _f_lab_inv(fy) * Yn
    z = _f_lab_inv(fz) * Zn
    return np.stack([x, y, z], axis=-1)


def yellow_brown_perceptual_cmap(
    low_hex: str = "#F6E8A5",   # pale yellow
    high_hex: str = "#4B2E13",  # dark brown
    n: int = 256,
    name: str = "gr_yellow_brown_lab",
) -> ListedColormap:
    """
    Two-color perceptual ramp by linear interpolation in CIE Lab.

    Notes:
      - Uses sRGB D65 <-> XYZ D65 conversions.
      - Output is clipped to [0,1] in linear RGB prior to sRGB encoding, and again
        clipped post-encoding (defensive against small numeric overshoots).
      - n must be >= 2.
    """
    n_i = int(n)
    if n_i < 2:
        raise ValueError(f"n must be >= 2 (got {n!r})")

    c0 = _hex_to_rgb01(low_hex)
    c1 = _hex_to_rgb01(high_hex)

    # Ensure shape (...,3) consistently
    c0 = np.asarray(c0, dtype="float64").reshape(3)
    c1 = np.asarray(c1, dtype="float64").reshape(3)

    lab0 = _xyz_to_lab(_linear_rgb_to_xyz(_srgb_to_linear(c0)))
    lab1 = _xyz_to_lab(_linear_rgb_to_xyz(_srgb_to_linear(c1)))

    t = np.linspace(0.0, 1.0, n_i, dtype="float64")[:, None]
    lab = (1.0 - t) * lab0 + t * lab1

    xyz = _lab_to_xyz(lab)
    rgb_lin = _xyz_to_linear_rgb(xyz)
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)
    rgb = _linear_to_srgb(rgb_lin)
    rgb = np.clip(rgb, 0.0, 1.0)

    return ListedColormap(rgb, name=str(name))
