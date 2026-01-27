# src/strataframe/spatial/geodesy.py
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

# Mean Earth radius in km (IUGG)
EARTH_RADIUS_KM: float = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two WGS84-ish lat/lon points in kilometers.

    Defensive features:
      - clamps haversine 'a' term to [0,1] to avoid asin domain errors
      - returns NaN if any input is non-finite
    """
    vals = (lat1, lon1, lat2, lon2)
    if any(not math.isfinite(float(v)) for v in vals):
        return float("nan")

    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))

    sin_dphi = math.sin(dphi / 2.0)
    sin_dlmb = math.sin(dlmb / 2.0)

    a = sin_dphi * sin_dphi + math.cos(p1) * math.cos(p2) * sin_dlmb * sin_dlmb

    if a < 0.0:
        a = 0.0
    elif a > 1.0:
        a = 1.0

    return float(2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a)))


def haversine_km_vec(
    lat1: Union[np.ndarray, float],
    lon1: Union[np.ndarray, float],
    lat2: Union[np.ndarray, float],
    lon2: Union[np.ndarray, float],
) -> np.ndarray:
    """
    Vectorized haversine distance (km). Inputs are degrees.

    Broadcasting rules apply (e.g., lat1[:,None] vs lat2[None,:] yields a matrix).

    Returns:
      np.ndarray (float64) of broadcasted shape.

    Notes:
      - Non-finite inputs propagate as NaN (numpy semantics).
      - Uses the same EARTH_RADIUS_KM constant as scalar haversine_km().
    """
    lat1 = np.asarray(lat1, dtype="float64")
    lon1 = np.asarray(lon1, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")

    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlmb = np.deg2rad(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2

    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-18, 1.0 - a)))
    return (EARTH_RADIUS_KM * c).astype("float64", copy=False)


def pairwise_haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Pairwise distance matrix (N,N) using haversine (km).

    Intended for moderate N (e.g., <= ~2000). For huge N, prefer kNN with spatial
    indexing (BallTree/etc.) to avoid O(N^2) memory/time.

    Returns:
      D: (N,N) float64 with zeros on diagonal.
    """
    lat = np.asarray(lat, dtype="float64").reshape(-1)
    lon = np.asarray(lon, dtype="float64").reshape(-1)
    D = haversine_km_vec(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(D, 0.0)
    return D
