"""OSM buildings via Overpass (centroid + optional height/levels)."""
from __future__ import annotations
import json, hashlib, re
import numpy as np
import pandas as pd
import requests
from config import settings as S


def _num(s):
    if s is None:
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+", str(s))
    return float(m.group()) if m else np.nan


def fetch_buildings(bbox, timeout=180):
    """bbox = (south, west, north, east) -> DataFrame[lat, lon, height_m]."""
    s, w, n, e = bbox
    key = hashlib.md5(f"{s:.4f},{w:.4f},{n:.4f},{e:.4f}".encode()).hexdigest()
    cache = S.CACHE_DIR / f"buildings_{key}.parquet"
    csv = S.CACHE_DIR / f"buildings_{key}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    q = f'[out:json][timeout:{timeout}];way["building"]({s},{w},{n},{e});out tags center;'
    last = None
    for url in S.OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": q}, headers={"User-Agent": S.USER_AGENT}, timeout=timeout + 30)
            r.raise_for_status()
            els = r.json().get("elements", [])
            break
        except Exception as ex:      # try next mirror
            last = ex
    else:
        raise RuntimeError(f"Overpass request failed: {last}")
    rows = []
    for el in els:
        c = el.get("center")
        if not c:
            continue
        t = el.get("tags", {})
        h = _num(t.get("height"))
        if np.isnan(h):
            lv = _num(t.get("building:levels"))
            h = lv * 3.0 + 1.5 if not np.isnan(lv) else np.nan     # 3 m/level (assumption)
        rows.append((c["lat"], c["lon"], h))
    df = pd.DataFrame(rows, columns=["lat", "lon", "height_m"])
    df.to_csv(csv, index=False)
    return df
