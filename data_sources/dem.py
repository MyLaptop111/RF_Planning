"""SRTM 30 m elevations through OpenTopoData (public API, ~1 req/s, 100 points/request)."""
from __future__ import annotations
import hashlib, json, time
import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator
from config import settings as S


class DEM:
    def __init__(self, lats, lons, z):
        self.lats = np.asarray(lats, float); self.lons = np.asarray(lons, float)
        self.z = np.asarray(z, float)
        self._f = RegularGridInterpolator((self.lats, self.lons), self.z, bounds_error=False, fill_value=None)

    def elevation(self, lat, lon):
        lat = np.asarray(lat, float); lon = np.asarray(lon, float)
        pts = np.column_stack([lat.ravel(), lon.ravel()])
        return self._f(pts).reshape(lat.shape)

    @property
    def mean(self):
        return float(np.nanmean(self.z))


def fetch_dem(bbox, step_m=500.0, max_axis=40, progress=None):
    s, w, n, e = bbox
    pad = 0.002
    s, w, n, e = s - pad, w - pad, n + pad, e + pad
    ny = int(np.clip(np.ceil((n - s) * 111320 / step_m) + 1, 5, max_axis))
    nx = int(np.clip(np.ceil((e - w) * 111320 * np.cos(np.radians((s + n) / 2)) / step_m) + 1, 5, max_axis))
    lats = np.linspace(s, n, ny); lons = np.linspace(w, e, nx)
    key = hashlib.md5(f"{s:.4f},{w:.4f},{n:.4f},{e:.4f},{ny},{nx}".encode()).hexdigest()
    cache = S.CACHE_DIR / f"dem_{key}.json"
    if cache.exists():
        z = np.array(json.loads(cache.read_text()), float)
        return DEM(lats, lons, z)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    pts = list(zip(LA.ravel(), LO.ravel()))
    vals = []
    for i in range(0, len(pts), 100):
        chunk = pts[i:i + 100]
        loc = "|".join(f"{a:.6f},{b:.6f}" for a, b in chunk)
        r = requests.get(S.OPENTOPODATA_URL, params={"locations": loc, "interpolation": "bilinear"},
                         headers={"User-Agent": S.USER_AGENT}, timeout=60)
        r.raise_for_status()
        vals += [x["elevation"] if x["elevation"] is not None else np.nan for x in r.json()["results"]]
        if progress:
            progress(min(1.0, (i + 100) / len(pts)))
        time.sleep(1.1)
    z = np.array(vals, float).reshape(ny, nx)
    z = np.where(np.isnan(z), np.nanmean(z), z)
    cache.write_text(json.dumps(z.tolist()))
    return DEM(lats, lons, z)
