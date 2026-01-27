# src/strataframe/spatial/proj.py
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Tuple

try:
    from pyproj import Transformer  # type: ignore
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore


DEFAULT_EPSG = 5070  # NAD83 / Conus Albers


def require_pyproj() -> None:
    if Transformer is None:
        raise RuntimeError("pyproj is required. Install with: pip install pyproj")


@lru_cache(maxsize=32)
def _xfm(src_epsg: int, dst_epsg: int):
    require_pyproj()
    # always_xy=True => input/output order is (lon,lat) and (x,y)
    return Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)


def lonlat_to_xy(lon: float, lat: float, *, epsg: int = DEFAULT_EPSG) -> Tuple[float, float]:
    t = _xfm(4326, int(epsg))
    x, y = t.transform(float(lon), float(lat))
    return float(x), float(y)


def xy_to_lonlat(x: float, y: float, *, epsg: int = DEFAULT_EPSG) -> Tuple[float, float]:
    t = _xfm(int(epsg), 4326)
    lon, lat = t.transform(float(x), float(y))
    return float(lon), float(lat)


def lonlat_to_xy_many(lons: Iterable[float], lats: Iterable[float], *, epsg: int = DEFAULT_EPSG) -> List[Tuple[float, float]]:
    t = _xfm(4326, int(epsg))
    xs, ys = t.transform(list(lons), list(lats))
    return [(float(x), float(y)) for x, y in zip(xs, ys)]
