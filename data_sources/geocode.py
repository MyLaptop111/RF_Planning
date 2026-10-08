"""Place name -> administrative boundary polygon (OSM Nominatim)."""
from __future__ import annotations
import json, hashlib
import requests
from config import settings as S
from rf_engine.geometry import geojson_to_geometry


def search_place(query: str, limit: int = 5, country_codes: str = None, timeout: int = 30):
    key = hashlib.md5(f"{query}|{limit}|{country_codes}".encode()).hexdigest()
    cache = S.CACHE_DIR / f"geocode_{key}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        params = dict(q=query, format="jsonv2", polygon_geojson=1, limit=limit, addressdetails=0)
        if country_codes:
            params["countrycodes"] = country_codes
        r = requests.get(S.NOMINATIM_URL, params=params, headers={"User-Agent": S.USER_AGENT}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
    out = []
    for d in data:
        geom = geojson_to_geometry(d.get("geojson")) if d.get("geojson") else None
        out.append(dict(name=d.get("display_name", ""), cls=f"{d.get('category','')}/{d.get('type','')}",
                        lat=float(d["lat"]), lon=float(d["lon"]), geometry=geom,
                        has_polygon=geom is not None and not geom.is_empty))
    return out
