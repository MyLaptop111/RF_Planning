"""Geodesy helpers: equal-area projection, areas, distances, GeoJSON handling."""
from __future__ import annotations
import math
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import transform, unary_union


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class Projector:
    """Local Lambert azimuthal equal-area projection (WGS84): exact areas, ~exact short distances."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self.crs = CRS.from_proj4(f"+proj=laea +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 "
                                  f"+datum=WGS84 +units=m +no_defs")
        self._fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)

    def geom_to_xy(self, geom):
        return transform(self._fwd.transform, geom)

    def geom_to_lonlat(self, geom):
        return transform(self._inv.transform, geom)


def projector_for(geom_lonlat) -> Projector:
    c = geom_lonlat.centroid
    return Projector(c.y, c.x)


def polygon_area_m2(geom_lonlat) -> float:
    return projector_for(geom_lonlat).geom_to_xy(geom_lonlat).area


def polygon_area_km2(geom_lonlat) -> float:
    return polygon_area_m2(geom_lonlat) / 1e6


def circle_polygon(lat, lon, radius_km):
    p = Projector(lat, lon)
    return p.geom_to_lonlat(Point(0, 0).buffer(radius_km * 1000.0, 128))


def geojson_to_geometry(gj):
    """Feature / FeatureCollection / geometry dict -> shapely Polygon/MultiPolygon (lon, lat)."""
    if gj is None:
        return None
    t = gj.get("type")
    if t == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in gj["features"] if f.get("geometry")]
    elif t == "Feature":
        geoms = [shape(gj["geometry"])]
    else:
        geoms = [shape(gj)]
    polys = []
    for g in geoms:
        if isinstance(g, Polygon):
            polys.append(g)
        elif isinstance(g, MultiPolygon):
            polys.extend(g.geoms)
    if not polys:
        return None
    return unary_union([p.buffer(0) for p in polys])
