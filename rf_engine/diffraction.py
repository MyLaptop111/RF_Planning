"""Terrain obstruction loss: single dominant knife-edge (ITU-R P.526) from a DEM."""
import numpy as np


def knife_edge_loss_db(v):
    v = np.asarray(v, float)
    out = np.zeros_like(v)
    m = v > -0.78
    out[m] = 6.9 + 20 * np.log10(np.sqrt((v[m] - 0.1) ** 2 + 1) + v[m] - 0.1)
    return out


def terrain_loss_db(dem, projector, tx_xy, tx_h_agl, rx_xy, rx_h_agl, f_mhz, k_samples=24, cap_db=40.0):
    """tx_xy/rx_xy: (N,2) metres; heights above local ground. Returns (N,) extra loss in dB."""
    tx_xy = np.asarray(tx_xy, float); rx_xy = np.asarray(rx_xy, float)
    tx_h_agl = np.broadcast_to(np.asarray(tx_h_agl, float), (len(tx_xy),))
    t = np.linspace(0, 1, k_samples + 2)[1:-1]
    px = tx_xy[:, 0:1] + (rx_xy[:, 0:1] - tx_xy[:, 0:1]) * t[None, :]
    py = tx_xy[:, 1:2] + (rx_xy[:, 1:2] - tx_xy[:, 1:2]) * t[None, :]
    lon, lat = projector.to_lonlat(px, py)
    z = dem.elevation(lat, lon)
    lon_t, lat_t = projector.to_lonlat(tx_xy[:, 0], tx_xy[:, 1])
    lon_r, lat_r = projector.to_lonlat(rx_xy[:, 0], rx_xy[:, 1])
    g_t = dem.elevation(lat_t, lon_t); g_r = dem.elevation(lat_r, lon_r)
    d = np.maximum(np.hypot(rx_xy[:, 0] - tx_xy[:, 0], rx_xy[:, 1] - tx_xy[:, 1]), 1.0)
    d1 = d[:, None] * t[None, :]; d2 = d[:, None] * (1 - t[None, :])
    bulge = d1 * d2 / (2 * (4 / 3) * 6371008.8)              # effective earth radius k = 4/3
    h_t = g_t + tx_h_agl; h_r = g_r + rx_h_agl
    los = h_t[:, None] + (h_r - h_t)[:, None] * t[None, :]
    h = z + bulge - los
    lam = 299.792458 / f_mhz
    v = h * np.sqrt(2 * (d1 + d2) / (lam * d1 * d2))
    return np.minimum(knife_edge_loss_db(v.max(axis=1)), cap_db)
