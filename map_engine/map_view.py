"""Folium maps: area/draw map and result map with RSRP / SINR overlays."""
from __future__ import annotations
import io, base64
import numpy as np
import folium
from folium.plugins import MeasureControl, Draw
import matplotlib
from matplotlib import colors
from PIL import Image


def base_map(center=(30.5877, 31.5020), zoom=12):
    return folium.Map(location=list(center), zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)


def add_draw_tools(m):
    MeasureControl(primary_length_unit="kilometers", primary_area_unit="sqkilometers").add_to(m)
    Draw(export=True, draw_options={"polyline": True, "polygon": True, "rectangle": True, "circle": False,
                                    "marker": True, "circlemarker": False}).add_to(m)


def _raster_overlay(raster, values, vmin, vmax, cmap, name, opacity=0.6, show=True):
    img = np.full((raster["ny"], raster["nx"]), np.nan)
    img[raster["iy"], raster["ix"]] = values
    img = img[::-1]                                             # north-up
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    rgba = matplotlib.colormaps[cmap](norm(np.ma.masked_invalid(img)))
    rgba[np.isnan(img)] = (0, 0, 0, 0)
    buf = io.BytesIO()
    Image.fromarray((rgba * 255).astype(np.uint8)).resize((raster["nx"] * 4, raster["ny"] * 4), Image.NEAREST).save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return folium.raster_layers.ImageOverlay(url, bounds=raster["bounds"], opacity=opacity, name=name, show=show)


def build_result_map(result, user_sites=None):
    g = result.grid
    b = result.raster["bounds"]
    m = base_map(((b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2), 12)
    folium.GeoJson(result.boundary_geojson, name="Boundary",
                   style_function=lambda f: dict(color="#1f3b73", weight=2, fillOpacity=0)).add_to(m)
    _raster_overlay(result.raster, g["rsrp_dbm"].values, -120, -70, "RdYlGn", "RSRP (dBm)").add_to(m)
    if g["sinr_db"].notna().any():
        _raster_overlay(result.raster, g["sinr_db"].values, -5, 25, "viridis", "SINR (dB)", show=False).add_to(m)
    _raster_overlay(result.raster, g["p_cov"].values, 0, 1, "Blues", "Coverage probability", show=False).add_to(m)
    fc = folium.FeatureGroup(name="Recommended sites")
    for _, r in result.sites.iterrows():
        color = "red" if r["Type"] == "Capacity" else "blue"
        folium.Marker([r["Latitude"], r["Longitude"]],
                      tooltip=f'{r["Site"]} ({r["Type"]})',
                      popup=folium.Popup(f'<b>{r["Site"]}</b><br>{r["Reason"]}<br>Azimuths: {r["Azimuths (deg)"]}<br>'
                                         f'Peak load: {r["Peak sector load (%)"]}%', max_width=300),
                      icon=folium.Icon(color=color, icon="signal", prefix="fa")).add_to(fc)
    fc.add_to(m)
    for i, (lat, lon) in enumerate(user_sites or [], 1):
        folium.CircleMarker([lat, lon], radius=5, color="black", tooltip=f"User candidate {i}").add_to(m)
    folium.LayerControl().add_to(m)
    return m
