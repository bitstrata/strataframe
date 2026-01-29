# src/strataframe/chronolog/geo.py
from __future__ import annotations

from typing import Tuple

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def latlon_to_xy_km(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    lat0: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equirectangular projection (good for regional scales).
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    if lat0 is None:
        lat0 = float(np.nanmedian(lat)) if lat.size else 0.0
    lat0_rad = np.deg2rad(lat0)
    x = EARTH_RADIUS_KM * np.deg2rad(lon) * np.cos(lat0_rad)
    y = EARTH_RADIUS_KM * np.deg2rad(lat)
    return x, y


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized haversine distance in km.
    """
    lat1 = np.asarray(lat1, dtype="float64")
    lon1 = np.asarray(lon1, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")

    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = phi2 - phi1
    dlambda = np.deg2rad(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
